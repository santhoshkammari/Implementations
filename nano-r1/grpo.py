import torch
import torch.nn.functional as F
from torch import Tensor
import contextlib
import time
import datetime
from collections import defaultdict
import logging

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset, load_dataset
import torch
from rich import print

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GRPO:
    def __init__(
        self,
        model,
        ref_model,
        reward_functions,
        tokenizer=None,
        group_size=8,
        micro_group_size=2,
        batch_size=1,
        max_iterations=1000,
        dataset=None,
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
        self.reward_functions: list = reward_functions

        self.using_lora = False
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto"
        ).to(self.device).to(dtype)

        self.distributed = False

        self.metrics = defaultdict(list)

        self.model.to(self.device).to(dtype)
        self.ref_model.to(self.device).to(dtype)

    def get_per_token_logps(self, model, input_ids, ) -> Tensor:
        logits = model(input_ids=input_ids).logits
        logits = logits[:, :-1, :]  # Shape: [2, 660, 128256]
        input_ids = input_ids[:, 1:]
        logps = F.log_softmax(logits, dim=-1)
        return torch.gather(logps, -1, input_ids.unsqueeze(-1)).squeeze(-1)

    def compute_loss(self, inputs, old_policy_log_probs, reward, mean_rewards, std_rewards, loss_mask) -> Tensor:
        logger.info(f"Computing loss - inputs shape: {inputs.shape}, reward shape: {reward.shape}")
        logger.info(
            f"Reward stats - mean: {reward.mean().item():.4f}, std: {reward.std().item():.4f}, min: {reward.min().item():.4f}, max: {reward.max().item():.4f}")

        policy_log_probs = self.get_per_token_logps(self.model, inputs)
        logger.info(f"Policy log probs shape: {policy_log_probs.shape}, mean: {policy_log_probs.mean().item():.4f}")

        with (
            self.ref_model.disable_adapter()
            if self.using_lora
            else contextlib.nullcontext()
        ):
            ref_policy_log_probs = self.get_per_token_logps(self.ref_model, inputs)
        logger.info(
            f"Ref policy log probs shape: {ref_policy_log_probs.shape}, mean: {ref_policy_log_probs.mean().item():.4f}")

        # advantage calculation
        advantage: Tensor = (reward - mean_rewards) / (std_rewards + 1e-6)
        advantage = advantage.reshape(-1, 1)
        logger.info(
            f"Advantage shape: {advantage.shape}, mean: {advantage.mean().item():.4f}, std: {advantage.std().item():.4f}")
        logger.info(f"Mean rewards: {mean_rewards.mean().item():.4f}, Std rewards: {std_rewards.mean().item():.4f}")

        # kl divergence calculation
        log_ratios = ref_policy_log_probs - policy_log_probs
        kld = torch.exp(log_ratios) - log_ratios - 1
        logger.info(f"KL divergence shape: {kld.shape}, mean: {kld.mean().item():.4f}")

        policy_ratio = torch.exp(policy_log_probs - old_policy_log_probs.detach())
        logger.info(
            f"Policy ratio shape: {policy_ratio.shape}, mean: {policy_ratio.mean().item():.4f}, min: {policy_ratio.min().item():.4f}, max: {policy_ratio.max().item():.4f}")

        loss1 = policy_ratio * advantage
        loss2 = torch.clamp(policy_ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage
        loss = -torch.min(loss1, loss2)
        logger.info(f"Raw loss shape: {loss.shape}, mean: {loss.mean().item():.4f}")

        loss = (loss * loss_mask).sum(dim=-1) / (loss_mask.sum(dim=-1) + 1e-6)
        kld = (kld * loss_mask).sum(dim=-1) / (loss_mask.sum(dim=-1) + 1e-6)
        logger.info(f"Masked loss mean: {loss.mean().item():.4f}, KLD mean: {kld.mean().item():.4f}")
        logger.info(f"Loss mask sum: {loss_mask.sum().item()}, total elements: {loss_mask.numel()}")

        loss += kld * self.beta
        final_loss = loss.mean()
        logger.info(f"Final loss: {final_loss.item():.6f}, beta: {self.beta}")

        for _kd in kld:
            self.metrics["kld"].append(_kd.mean().item())
        return final_loss

    def sample_batch(self):
        if self.distributed:
            return self.distributed_sample_batch()

        inputs_texts = []
        samples = []
        for _ in range(self.batch_size):
            item = next(self.data_loader_iter)
            samples.append(item)
            prompt = item["prompt"]

            # Use simple format instead of chat template that's causing issues
            system_msg = prompt[0]["content"]
            user_msg = prompt[1]["content"]
            formatted = f"System: {system_msg}\n\nUser: {user_msg}\n\nAssistant:"
            inputs_texts.append(formatted)
            logger.info(f"Formatted prompt: {formatted}")

        encoded = self.tokenizer(inputs_texts, padding=True, return_tensors="pt")
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        # Also set pad_token_id if not set:
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        prompt_length = input_ids.shape[1]

        input_ids = torch.repeat_interleave(input_ids, self.group_size, dim=0)
        samples = [sample for _ in range(self.group_size) for sample in samples]

        start_time = time.time()
        max_new_tokens = 512
        outputs = self.model.generate(
            input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            max_new_tokens=max_new_tokens,
            temperature=1.2,  # Higher temperature for more variety
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.2,  # Prevent repetition
            pad_token_id=self.tokenizer.pad_token_id,
        )
        end_time = time.time()
        print(f"Time for generation: {end_time - start_time} seconds")

        decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)

        # Log first few responses to debug
        for i, output in enumerate(decoded_outputs[:2]):
            logger.info(f"Generated response {i}: {output}")

        rewards = self.compute_rewards(samples, decoded_outputs)

        loss_mask = torch.zeros(outputs.shape, dtype=torch.bool)

        gen_tokens = outputs[:, prompt_length:]
        valid_gen_mask = gen_tokens != self.tokenizer.pad_token_id
        loss_mask[:, prompt_length:] = valid_gen_mask

        return outputs, rewards.to(self.dtype).float(), loss_mask[:, 1:]

    def compute_rewards(self, samples, responces) -> list:
        logger.info(f"Computing rewards for {len(samples)} samples, {len(responces)} responses")
        rewards = [[[] for _ in range(self.batch_size)] for _ in range(len(self.reward_functions))]

        total_individual_rewards = []
        for idx, (sample, resp) in enumerate(zip(samples, responces)):
            reward = 0
            individual_rewards = []
            for func_idx, func in enumerate(self.reward_functions):
                func_reward = func(sample, resp)
                reward += func_reward
                individual_rewards.append(func_reward)
                logger.info(f"Sample {idx}: {func.__name__} reward: {func_reward:.4f}")
                rewards[func_idx][idx % self.batch_size].append(reward)
            total_individual_rewards.append(reward)
            logger.info(f"Sample {idx}: Total reward: {reward:.4f}")

        logger.info(
            f"Individual rewards stats - mean: {sum(total_individual_rewards) / len(total_individual_rewards):.4f}, min: {min(total_individual_rewards):.4f}, max: {max(total_individual_rewards):.4f}")

        rewards = torch.tensor(rewards, dtype=self.dtype).to(self.device)
        logger.info(f"Rewards tensor shape: {rewards.shape}, mean: {rewards.mean().item():.4f}")

        for func_idx, func in enumerate(self.reward_functions):
            rwds = rewards[func_idx].mean(dim=-1)
            for r in rwds:
                self.metrics[f"reward_{func.__name__}"].append(r.item())

        prompt_lenghts = [[] for _ in range(self.batch_size)]
        for idx, sample in enumerate(samples):
            prompt_lenghts[idx % self.batch_size].append(len(sample["prompt"]))

        for idx, pl in enumerate(prompt_lenghts):
            self.metrics[f"prompt_length"].append(sum(pl) / len(pl))

        final_rewards = rewards.sum(dim=0)
        logger.info(f"Final rewards shape: {final_rewards.shape}, mean: {final_rewards.mean().item():.4f}")
        return final_rewards

    def log_metrics(self):
        pass

    def train(self, epochs=1, max_iterations=1000, max_time_seconds=None):
        idx = 0
        start_time = time.perf_counter()
        logger.info(f"Starting training - max_iterations: {max_iterations}, max_time_seconds: {max_time_seconds}")

        while idx < max_iterations:
            if max_time_seconds and (time.perf_counter() - start_time) > max_time_seconds:
                logger.info(f"Training stopped due to time limit: {max_time_seconds} seconds")
                break

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

                avg_loss = sum(group_losses) / len(group_losses)
                logger.info(f"Before optimizer step - avg_loss: {avg_loss:.6f}")
                logger.info(f"Group losses: {[f'{loss:.6f}' for loss in group_losses]}")

                self.optimizer.step()
                self.optimizer.zero_grad()

                print(f"{idx:04d} loss: {avg_loss:.6f} reward: {reward.mean():.4f}")
                logger.info(f"Step {idx:04d} completed - loss: {avg_loss:.6f}, reward: {reward.mean().item():.4f}")

                self.metrics["idx"].append(idx)
                self.metrics["total_reward"].append(reward.mean().item())
                self.metrics["loss"].append(avg_loss)

                torch.cuda.empty_cache()

            print(f"iter {idx}  >>> reward: {rewards.mean()}")
            print(f"Total time: {str(datetime.timedelta(seconds=int(time.perf_counter() - start_time)))}")
            self.log_metrics()


def response_format_reward(sample: dict, s: str, *args, **kwargs):
    logger.info(f"Raw response (first 200 chars): {s[:200]}...")

    # Base reward to ensure minimum value and add some noise for variety
    import random
    total_reward = 0.1 + random.uniform(-0.05, 0.05)  # Add noise: 0.05 to 0.15

    # Extract assistant response - handle multiple possible formats
    original_s = s
    extracted = False

    # Try different extraction patterns
    extraction_patterns = [
        "Assistant:",
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
        "<|start_header_id|>assistant<|end_header_id|>",
        "<|im_start|>assistant"
    ]

    for pattern in extraction_patterns:
        if pattern in s:
            try:
                s = s.split(pattern)[1]
                extracted = True
                logger.info(f"Extracted using pattern '{pattern}': {s[:100]}...")
                break
            except:
                continue

    if not extracted:
        logger.info("No extraction pattern matched, using full response")
        s = original_s
        # Add variety even for non-extracted responses
        total_reward += random.uniform(-0.03, 0.03)

    # Clean up common end tokens
    for end_token in ["<|eot_id|>", "<|im_end|>", "<|endoftext|>"]:
        if end_token in s:
            s = s.split(end_token)[0]

    logger.info(f"Final cleaned response: {s}")

    # Basic criteria with partial rewards

    # 1. Length reward (longer responses get small bonus) + add variety
    length = len(s.strip())
    if length > 10:
        total_reward += 0.1 + random.uniform(0, 0.05)
    if length > 50:
        total_reward += 0.1 + random.uniform(0, 0.05)
    if length > 100:
        total_reward += 0.05 + random.uniform(0, 0.03)

    # Add length-based variety to prevent uniform rewards
    total_reward += min(length / 1000, 0.1)  # Up to 0.1 bonus for longer responses

    # 2. Contains any thinking-related words + variety
    thinking_words = ['think', 'reason', 'consider', 'analyze', 'calculate', 'step', 'first', 'then', 'so', 'therefore']
    found_words = 0
    for word in thinking_words:
        if word.lower() in s.lower():
            found_words += 1
            total_reward += 0.05 + random.uniform(0, 0.02)

    if found_words > 0:
        total_reward += 0.1 + random.uniform(0, 0.05)

    # 3. Contains numbers (since this is a math task) + variety
    import re
    numbers = re.findall(r'\d+', s)
    if numbers:
        total_reward += 0.2 + random.uniform(0, 0.1)
        # Bonus for multiple numbers
        if len(numbers) > 1:
            total_reward += 0.05 + random.uniform(0, 0.03)

    # 4. Tag-based rewards (progressive) + variety
    tag_rewards = {
        '<thinking>': 0.3,
        '</thinking>': 0.3,
        '<answer>': 0.4,
        '</answer>': 0.4
    }

    for tag, reward in tag_rewards.items():
        if tag in s:
            total_reward += reward + random.uniform(-0.02, 0.02)
            # Penalty for multiple occurrences
            if s.count(tag) > 1:
                total_reward -= (s.count(tag) - 1) * 0.05

    # 5. Structure rewards + variety
    if '<thinking>' in s and '</thinking>' in s:
        total_reward += 0.5 + random.uniform(-0.05, 0.05)
        # Check if thinking section has content
        thinking_match = re.search(r'<thinking>(.*?)</thinking>', s, re.DOTALL)
        if thinking_match and len(thinking_match.group(1).strip()) > 5:
            total_reward += 0.3 + random.uniform(0, 0.1)

    if '<answer>' in s and '</answer>' in s:
        total_reward += 0.5 + random.uniform(-0.05, 0.05)
        # Check if answer section has content
        answer_match = re.search(r'<answer>(.*?)</answer>', s, re.DOTALL)
        if answer_match:
            answer_content = answer_match.group(1).strip()
            if len(answer_content) > 0:
                total_reward += 0.3 + random.uniform(0, 0.05)

                # Try to parse as number and check correctness
                try:
                    predicted = float(answer_content)
                    actual = float(sample['answer'])
                    total_reward += 0.5 + random.uniform(0, 0.1)  # Bonus for parseable number

                    if predicted == actual:
                        total_reward += 1.0 + random.uniform(0, 0.2)  # Correct answer
                    elif abs(predicted - actual) / max(abs(actual), 1) < 0.1:  # Within 10%
                        total_reward += 0.5 + random.uniform(0, 0.1)  # Close answer
                except:
                    pass

    # 6. Perfect format bonus + variety
    if ('</thinking><answer>' in s and
        s.count('<thinking>') == 1 and
        s.count('</thinking>') == 1 and
        s.count('<answer>') == 1 and
        s.count('</answer>') == 1):
        total_reward += 1.0 + random.uniform(0, 0.2)

    # 7. No extra content after </answer> + variety
    if '</answer>' in s:
        parts = s.split('</answer>')
        if len(parts) > 1 and parts[1].strip() == '':
            total_reward += 0.2 + random.uniform(0, 0.05)

    logger.info(f"Final reward: {total_reward:.3f} for response length {len(s)}")
    return total_reward


# model_name = "Qwen/Qwen2.5-0.5B-Instruct"
# small models are kind of dumb, they need a little push so using this fine-tuned model
# source: https://github.com/joey00072/nanoGRPO/blob/master/cold_start/cold_start_finetune.py
# you can totally use the base model, it will just take longer to converge
# model_name = "joey00072/Llama-3.2-1B-Instruct-cold-start-ft2"

# model_name = "unsloth/Llama-3.2-3B-Instruct"
model_path = "/home/data_science/project_files/santhosh/models/qwen3_1p7_base"
tokenizer_model = "Qwen/Qwen3-1.7B"

model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

print(model)


def process_example(example: dict):
    return {
        "prompt": [
            {"role": "system",
             "content": "Respond in following format:<thinking>{step by step reasoning}</thinking><answer>{number}</answer>"},
            {"role": "user", "content": str(example["nums"])}
        ],
        "answer": example['target']
    }


dataset = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split='train')
dataset = dataset.map(process_example, num_proc=12)
ref_model = None

trainer = GRPO(
    model,
    ref_model,
    tokenizer=tokenizer,
    group_size=8,
    micro_group_size=2,
    lr=5e-6,
    weight_decay=0.1,
    dataset=dataset,
    reward_functions=[response_format_reward],
)

# Run training for 20 seconds for debugging
trainer.train(max_time_seconds=20)