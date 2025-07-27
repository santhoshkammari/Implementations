import torch
import torch.nn.functional as F
from torch import Tensor
import time
import wandb
import datetime
from collections import defaultdict


class GRPO:
    def __init__(
        self,
        model,
        ref_model,
        tokenizer=None,
        group_size=8,
        micro_group_size=2,
        batch_size=1,
        max_iterations=1000,
        dataset=None,
        reward_functions=None,
        log_wandb=False,
        dtype=None,
        lr=5e-6,
        weight_decay=0.0,
        beta=0.0,
        epsilon=0.1
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.dataset = dataset.shuffle(seed=42)
        self.data_loader_iter = iter(self.dataset)
        self.group_size = group_size
        self.micro_group_size = micro_group_size
        self.batch_size = batch_size
        self.max_iterations = max_iterations
        self.dtype = dtype if dtype is not None else (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
        self.beta = beta
        self.epsilon = epsilon
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        assert reward_functions is not None, "Must pass reward_functions"
        self.reward_functions: list = reward_functions

        assert self.ref_model is not None, "Reference model must be provided"
        
        self.distributed = False
        self.log_wandb = log_wandb
        if self.log_wandb:
            wandb.init(project="nanoGRPO")

        self.metrics = defaultdict(list)

        self.model.to(self.device).to(dtype)
        self.ref_model.to(self.device).to(dtype)

    def get_per_token_logps(self, model, input_ids) -> Tensor:
        logits = model(input_ids=input_ids).logits
        logits = logits[:, :-1, :]  # Shape: [2, 660, 128256]
        input_ids = input_ids[:, 1:]
        logps = F.log_softmax(logits, dim=-1)
        return torch.gather(logps, -1, input_ids.unsqueeze(-1)).squeeze(-1)

    def compute_loss(self, inputs, old_policy_log_probs, reward, mean_rewards, std_rewards, loss_mask) -> Tensor:
        policy_log_probs = self.get_per_token_logps(self.model, inputs)

        with torch.no_grad():
            ref_policy_log_probs = self.get_per_token_logps(self.ref_model, inputs)

        # advantage calculation
        advantage: Tensor = (reward - mean_rewards) / (std_rewards + 1e-6)
        advantage = advantage.reshape(-1, 1)

        # kl divergence calculation
        log_ratios = ref_policy_log_probs - policy_log_probs
        kld = torch.exp(log_ratios) - log_ratios - 1

        policy_ratio = torch.exp(policy_log_probs - old_policy_log_probs.detach())

        loss1 = policy_ratio * advantage
        loss2 = torch.clamp(policy_ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage
        loss = -torch.min(loss1, loss2)
        loss = (loss * loss_mask).sum(dim=-1) / (loss_mask.sum(dim=-1) + 1e-6)
        kld = (kld * loss_mask).sum(dim=-1) / (loss_mask.sum(dim=-1) + 1e-6)
        loss += kld * self.beta
        if self.log_wandb:
            for _kd in kld:
                self.metrics["kld"].append(_kd.mean().item())
        return loss.mean()

    def sample_batch(self):
        inputs_texts = []
        samples = []
        for _ in range(self.batch_size):
            item = next(self.data_loader_iter)
            samples.append(item)
            prompt = item["prompt"]
            formatted = self.tokenizer.apply_chat_template(prompt, tokenize=False)
            inputs_texts.append(formatted)

        encoded = self.tokenizer(inputs_texts, padding=True, return_tensors="pt")
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        prompt_length = input_ids.shape[1]

        input_ids = torch.repeat_interleave(input_ids, self.group_size, dim=0)
        samples = [sample for _ in range(self.group_size) for sample in samples]

        start_time = time.time()
        max_new_tokens = 512
        outputs = self.model.generate(
            input_ids.to(self.device),
            # min_new_tokens=512,
            max_new_tokens=max_new_tokens,
            temperature=0.9,
            # repetition_penalty=1.1,
        )
        end_time = time.time()
        print(f"Time for generation: {end_time - start_time} seconds")

        decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)

        rewards = self.compute_rewards(samples, decoded_outputs)

        loss_mask = torch.zeros(outputs.shape, dtype=torch.bool)

        gen_tokens = outputs[:, prompt_length:]
        valid_gen_mask = gen_tokens != self.tokenizer.pad_token_id
        loss_mask[:, prompt_length:] = valid_gen_mask

        return outputs, rewards.float(), loss_mask[:, 1:]

    def compute_rewards(self, samples, responces) -> torch.Tensor:
        rewards = [[[] for _ in range(self.batch_size)] for _ in range(len(self.reward_functions))]

        for idx, (sample, resp) in enumerate(zip(samples, responces)):
            reward = 0
            for func_idx, func in enumerate(self.reward_functions):
                reward += func(sample, resp)
                # print(f"{func.__name__} reward: {reward}")
                rewards[func_idx][idx % self.batch_size].append(reward)

        rewards = torch.tensor(rewards, dtype=self.dtype).to(self.device)

        # print(f"rewards: {rewards.shape}")
        for func_idx, func in enumerate(self.reward_functions):
            rwds = rewards[func_idx].mean(dim=-1)
            for r in rwds:
                self.metrics[f"reward_{func.__name__}"].append(r.item())

        prompt_lenghts = [[] for _ in range(self.batch_size)]
        for idx, sample in enumerate(samples):
            prompt_lenghts[idx % self.batch_size].append(len(sample["prompt"]))

        for idx, pl in enumerate(prompt_lenghts):
            self.metrics[f"prompt_length"].append(sum(pl) / len(pl))

        return rewards.sum(dim=0)

    def log_metrics(self):
        if self.log_wandb:
            idx = self.metrics["idx"][-1] - 1
            metrics = {}
            for k, v in self.metrics.items():
                metrics[f"train/{k}"] = v[idx] if len(v) >= idx else v[-1]

            wandb.log(metrics)

    def train(self, epochs=1, max_iterations=1000):
        idx = 0
        start_time = time.perf_counter()
        while idx < max_iterations:

            x_batch_inputs, rewards, loss_mask = self.sample_batch()
            torch.cuda.empty_cache()

            batch_inputs = x_batch_inputs.reshape(self.batch_size, self.group_size, *x_batch_inputs.shape[1:])
            loss_mask = loss_mask.reshape(self.batch_size, self.group_size, *loss_mask.shape[1:])
            torch.cuda.empty_cache()  # gpu poor hack

            # offload to cpu to save vram
            batch_inputs = batch_inputs.cpu()
            rewards = rewards.cpu()
            loss_mask = loss_mask.cpu()
            torch.cuda.empty_cache()  # gpu poor hack

            pi_old = []
            for _, (b_inputs) in enumerate(batch_inputs):
                with torch.no_grad():
                    b_old_policy_log_probs = self.get_per_token_logps(self.model, b_inputs.to(self.device)).cpu()
                    torch.cuda.empty_cache()
                    pi_old.append(b_old_policy_log_probs)

            for _, (b_inputs, b_old_policy_log_probs, b_reward, b_loss_mask) in enumerate(
                zip(batch_inputs, pi_old, rewards, loss_mask)):
                idx += 1
                reward = b_reward.to(self.device)
                mean_rewards = reward.mean(dim=-1).unsqueeze(-1)
                std_rewards = reward.std(dim=-1).unsqueeze(-1)

                # even grop are too big for vram
                # so we split them into micro groups (its same as micro batching)
                g_inputs = b_inputs.reshape(b_inputs.shape[0] // self.micro_group_size, self.micro_group_size,
                                            *b_inputs.shape[1:]).cpu()
                g_old_policy_log_probs = b_old_policy_log_probs.reshape(b_inputs.shape[0] // self.micro_group_size,
                                                                        self.micro_group_size,
                                                                        *b_old_policy_log_probs.shape[1:]).cpu()
                g_reward = b_reward.reshape(b_inputs.shape[0] // self.micro_group_size, self.micro_group_size,
                                            *b_reward.shape[1:]).cpu()
                g_loss_mask = b_loss_mask.reshape(b_inputs.shape[0] // self.micro_group_size, self.micro_group_size,
                                                  *b_loss_mask.shape[1:]).cpu()
                group_losses = []

                for inputs, old_policy_log_probs, reward, loss_mask in zip(g_inputs, g_old_policy_log_probs, g_reward,
                                                                           g_loss_mask):
                    inputs = inputs.to(self.device)
                    old_policy_log_probs = old_policy_log_probs.to(self.device)
                    reward = reward.to(self.device)
                    loss_mask = loss_mask.to(self.device)

                    loss = self.compute_loss(
                        inputs,
                        old_policy_log_probs,
                        reward,
                        mean_rewards,
                        std_rewards,
                        loss_mask
                    )
                    group_losses.append(loss.item())
                    loss.backward()
                    torch.cuda.empty_cache()

                self.optimizer.step()
                self.optimizer.zero_grad()

                print(f"{idx:04d} loss: {sum(group_losses) / len(group_losses)} reward: {reward.mean()}")
                if self.log_wandb:
                    self.metrics["idx"].append(idx)
                    self.metrics["total_reward"].append(reward.mean().item())
                    self.metrics["loss"].append(sum(group_losses) / len(group_losses))

                torch.cuda.empty_cache()

            print(f"iter {idx}  >>> reward: {rewards.mean()}")
            print(f"Total time: {str(datetime.timedelta(seconds=int(time.perf_counter() - start_time)))}")
            self.log_metrics()

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
from rich import print


SYSTEM_PROMPT = "Respond in following format:<thinking>{step by step reasoning}</thinking><answer>{number}</answer>"







def reward_func_len(sample: dict, s: str, *args, **kwargs):
    return 4 - (len(s)/1000)

def response_format_reward(sample: dict, s: str, *args, **kwargs):
    # print(sample.keys())
    correct_template =0
    try:
        s = s.split("<|eot_id|><|start_header_id|>assistant<|end_header_id|>")[1]
    except:
        return -1
    if "<|eot_id|>" in s:
        s = s.split("<|eot_id|>")[0]
    try:
        print("-"*100)
        print(s)
        print("-"*100)
    except:
        ...
    total_reward = 0
    for tag in ["<thinking>", "</thinking>", "<answer>", "</answer>"]:
        if tag in s:
            total_reward+=0.15
            if s.count(tag)>1:
                total_reward -= s.count(tag)*0.01

    if s.count("<thinking>")==1:
        total_reward += .5
    else:
        total_reward -= .1

    if s.count("</thinking><answer>")==1:
        total_reward += 1
        correct_template += 1
    else:
        if s.count("<thinking>")==1:
            total_reward += .2
        else:
            total_reward -= .1
        if s.count("<answer>")==1:
            total_reward += .2
        else:
            total_reward -= .1

    if s.count("</answer>")==1 and s.split("</answer>")[1].strip() == "":
            total_reward += 1
    else:
        total_reward -= 0.1

    if s.count("<answer>")==1:
        total_reward += .2

        r = s.split("<answer>")[1].strip()
        if "</answer>" in r:
            total_reward += .2
            if r.count("</answer>")==1:
                total_reward += 2
                split = r.split("</answer>")
                r = split[0].strip()
                try:
                    r = float(r)
                    total_reward += 1
                    if r == float(sample["answer"]):
                        total_reward += 2
                        correct_template += 1
                except:
                    total_reward -= 0.1

                if len(split) > 1:
                    if split[1].strip() != "":
                        total_reward += 3
                        correct_template += 1
                    else:
                        total_reward -= len(split[1].strip())/1000
                else:
                    total_reward -= 0.2
            else:
                total_reward -= 0.1
        else:
            total_reward -=0.1
    if correct_template == 3:
        total_reward += 2
    return total_reward


def process_example(example: dict):
    return {
        "prompt": [
            {"role": "system",
             "content": "Respond in following format:<thinking>{step by step reasoning}</thinking><answer>{number}</answer>"},
            {"role": "user", "content": str(example["nums"])}
        ],
        "answer": example['target']
    }


# Load and split dataset properly
dataset = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split='train')
dataset = dataset.map(process_example, remove_columns=[
            col for col in dataset.column_names if col not in ["prompt", "answer"]
        ], num_proc=12)

group_size = 8
micro_group_size =2
lr = 5e-6
weight_decay = 0.1
reward_functions = [
    response_format_reward,
]

# model_name = "Qwen/Qwen2.5-0.5B-Instruct"
# small models are kind of dumb, they need a little push so using this fine-tuned model
# source: https://github.com/joey00072/nanoGRPO/blob/master/cold_start/cold_start_finetune.py
# you can totally use the base model, it will just take longer to converge
# model_name = "joey00072/Llama-3.2-1B-Instruct-cold-start-ft2"

# model_name = "unsloth/Llama-3.2-3B-Instruct"
model_name = "/home/data_science/project_files/santhosh/models/qwen3_1p7_base"
tokenizer_model = "Qwen/Qwen3-1.7B"

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)

# Load reference model (copy of the original model)
ref_model = AutoModelForCausalLM.from_pretrained(model_name)

# Convert to bfloat16
model = model.to(torch.bfloat16)
ref_model = ref_model.to(torch.bfloat16)

print(model)
trainer = GRPO(
    model,
    ref_model,
    tokenizer=tokenizer,
    group_size=group_size,
    micro_group_size=micro_group_size,
    dataset=dataset,
    reward_functions=reward_functions,
    log_wandb=True,
    lr=lr,
    weight_decay=weight_decay
)

trainer.train()