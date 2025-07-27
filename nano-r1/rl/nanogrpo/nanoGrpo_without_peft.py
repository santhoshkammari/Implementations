import json

import torch
import torch.nn.functional as F
from torch import Tensor
import time
import wandb
import datetime
from collections import defaultdict
import logging
import numpy as np


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
        epsilon=0.1,
        debug_logging=True
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
        self.debug_logging = debug_logging
        
        # Setup debug logger
        if self.debug_logging:
            logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
            self.logger = logging.getLogger('GRPO')
            self.logger.info(f"Initializing GRPO with group_size={group_size}, micro_group_size={micro_group_size}, lr={lr}, beta={beta}, epsilon={epsilon}")
        
        self.model.to(self.device).to(dtype)
        self.ref_model.to(self.device).to(dtype)

    def get_per_token_logps(self, model, input_ids) -> Tensor:
        logits = model(input_ids=input_ids).logits
        logits = logits[:, :-1, :]  # Shape: [2, 660, 128256]
        input_ids = input_ids[:, 1:]
        logps = F.log_softmax(logits, dim=-1)
        per_token_logps = torch.gather(logps, -1, input_ids.unsqueeze(-1)).squeeze(-1)
        
        if self.debug_logging:
            self.logger.debug(f"get_per_token_logps: input_ids shape={input_ids.shape}, logits shape={logits.shape}")
            self.logger.debug(f"get_per_token_logps: per_token_logps shape={per_token_logps.shape}, mean={per_token_logps.mean().item():.6f}, std={per_token_logps.std().item():.6f}")
            self.logger.debug(f"get_per_token_logps: min={per_token_logps.min().item():.6f}, max={per_token_logps.max().item():.6f}")
        
        return per_token_logps

    def compute_loss(self, inputs, old_policy_log_probs, reward, mean_rewards, std_rewards, loss_mask) -> Tensor:
        if self.debug_logging:
            self.logger.debug(f"compute_loss: inputs shape={inputs.shape}")
            self.logger.debug(f"compute_loss: old_policy_log_probs shape={old_policy_log_probs.shape}, mean={old_policy_log_probs.mean().item():.6f}")
            self.logger.debug(f"compute_loss: reward shape={reward.shape}, mean={reward.mean().item():.6f}, std={reward.std().item():.6f}")
            self.logger.debug(f"compute_loss: mean_rewards={mean_rewards.mean().item():.6f}, std_rewards={std_rewards.mean().item():.6f}")
            self.logger.debug(f"compute_loss: loss_mask shape={loss_mask.shape}, sum={loss_mask.sum().item()}")
        
        policy_log_probs = self.get_per_token_logps(self.model, inputs)

        with torch.no_grad():
            ref_policy_log_probs = self.get_per_token_logps(self.ref_model, inputs)

        if self.debug_logging:
            self.logger.debug(f"compute_loss: policy_log_probs mean={policy_log_probs.mean().item():.6f}, std={policy_log_probs.std().item():.6f}")
            self.logger.debug(f"compute_loss: ref_policy_log_probs mean={ref_policy_log_probs.mean().item():.6f}, std={ref_policy_log_probs.std().item():.6f}")

        # advantage calculation
        advantage: Tensor = (reward - mean_rewards) / (std_rewards + 1e-6)
        advantage = advantage.reshape(-1, 1)
        
        if self.debug_logging:
            self.logger.debug(f"compute_loss: advantage shape={advantage.shape}, mean={advantage.mean().item():.6f}, std={advantage.std().item():.6f}")
            self.logger.debug(f"compute_loss: advantage min={advantage.min().item():.6f}, max={advantage.max().item():.6f}")

        # kl divergence calculation
        log_ratios = ref_policy_log_probs - policy_log_probs
        kld = torch.exp(log_ratios) - log_ratios - 1
        
        if self.debug_logging:
            self.logger.debug(f"compute_loss: log_ratios mean={log_ratios.mean().item():.6f}, std={log_ratios.std().item():.6f}")
            self.logger.debug(f"compute_loss: kld mean={kld.mean().item():.6f}, std={kld.std().item():.6f}")

        policy_ratio = torch.exp(policy_log_probs - old_policy_log_probs.detach())
        
        if self.debug_logging:
            self.logger.debug(f"compute_loss: policy_ratio mean={policy_ratio.mean().item():.6f}, std={policy_ratio.std().item():.6f}")
            self.logger.debug(f"compute_loss: policy_ratio min={policy_ratio.min().item():.6f}, max={policy_ratio.max().item():.6f}")

        loss1 = policy_ratio * advantage
        loss2 = torch.clamp(policy_ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage
        loss = -torch.min(loss1, loss2)
        
        if self.debug_logging:
            self.logger.debug(f"compute_loss: loss1 mean={loss1.mean().item():.6f}, loss2 mean={loss2.mean().item():.6f}")
            self.logger.debug(f"compute_loss: loss before masking mean={loss.mean().item():.6f}, std={loss.std().item():.6f}")
        
        loss = (loss * loss_mask).sum(dim=-1) / (loss_mask.sum(dim=-1) + 1e-6)
        kld = (kld * loss_mask).sum(dim=-1) / (loss_mask.sum(dim=-1) + 1e-6)
        
        if self.debug_logging:
            self.logger.debug(f"compute_loss: loss after masking mean={loss.mean().item():.6f}, std={loss.std().item():.6f}")
            self.logger.debug(f"compute_loss: kld after masking mean={kld.mean().item():.6f}, std={kld.std().item():.6f}")
        
        loss += kld * self.beta
        
        if self.debug_logging:
            self.logger.debug(f"compute_loss: final loss mean={loss.mean().item():.6f}, std={loss.std().item():.6f}")
            self.logger.debug(f"compute_loss: beta={self.beta}, kld contribution={kld.mean().item() * self.beta:.6f}")
        
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
            formatted = self.tokenizer.apply_chat_template(prompt, add_generation_prompt=True,
                                                           enable_thinking=True,
                                                           tokenize=False)
            inputs_texts.append(formatted)

        encoded = self.tokenizer(inputs_texts, padding=True, return_tensors="pt")
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        prompt_length = input_ids.shape[1]

        input_ids = torch.repeat_interleave(input_ids, self.group_size, dim=0)
        samples = [sample for _ in range(self.group_size) for sample in samples]

        start_time = time.time()
        max_new_tokens = 1024
        outputs = self.model.generate(
            input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
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
        if self.debug_logging:
            self.logger.debug(f"compute_rewards: processing {len(samples)} samples with {len(self.reward_functions)} reward functions")
        
        rewards = [[[] for _ in range(self.batch_size)] for _ in range(len(self.reward_functions))]

        for idx, (sample, resp) in enumerate(zip(samples, responces)):
            reward = 0
            individual_rewards = []
            for func_idx, func in enumerate(self.reward_functions):
                func_reward = func(sample, resp)
                individual_rewards.append(func_reward)
                reward += func_reward
                rewards[func_idx][idx % self.batch_size].append(reward)
            
            if self.debug_logging and idx < 3:  # Log first few samples
                self.logger.debug(f"compute_rewards: sample {idx} individual rewards={individual_rewards}, total={reward}")

        rewards = torch.tensor(rewards, dtype=self.dtype).to(self.device)
        
        if self.debug_logging:
            self.logger.debug(f"compute_rewards: rewards tensor shape={rewards.shape}")
            self.logger.debug(f"compute_rewards: rewards mean={rewards.mean().item():.6f}, std={rewards.std().item():.6f}")
            self.logger.debug(f"compute_rewards: rewards min={rewards.min().item():.6f}, max={rewards.max().item():.6f}")

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
        if self.debug_logging:
            self.logger.debug(f"compute_rewards: final rewards shape={final_rewards.shape}, mean={final_rewards.mean().item():.6f}")
        
        return final_rewards

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
        
        if self.debug_logging:
            self.logger.info(f"Starting training for {max_iterations} iterations")
            
        while idx < max_iterations:
            if self.debug_logging:
                self.logger.debug(f"\n=== Training iteration {idx+1} ===")

            x_batch_inputs, rewards, loss_mask = self.sample_batch()
            torch.cuda.empty_cache()

            if self.debug_logging:
                self.logger.debug(f"train: x_batch_inputs shape={x_batch_inputs.shape}")
                self.logger.debug(f"train: rewards shape={rewards.shape}, mean={rewards.mean().item():.6f}")
                self.logger.debug(f"train: loss_mask shape={loss_mask.shape}, active tokens={loss_mask.sum().item()}")

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

            for batch_idx, (b_inputs, b_old_policy_log_probs, b_reward, b_loss_mask) in enumerate(
                zip(batch_inputs, pi_old, rewards, loss_mask)):
                idx += 1
                reward = b_reward.to(self.device)
                mean_rewards = reward.mean(dim=-1).unsqueeze(-1)
                std_rewards = reward.std(dim=-1).unsqueeze(-1)
                
                if self.debug_logging:
                    self.logger.debug(f"train: batch {batch_idx}, reward stats - mean={mean_rewards.mean().item():.6f}, std={std_rewards.mean().item():.6f}")
                    self.logger.debug(f"train: b_inputs shape={b_inputs.shape}, b_reward shape={b_reward.shape}")

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

                for micro_idx, (inputs, old_policy_log_probs, reward, loss_mask) in enumerate(zip(g_inputs, g_old_policy_log_probs, g_reward,
                                                                           g_loss_mask)):
                    inputs = inputs.to(self.device)
                    old_policy_log_probs = old_policy_log_probs.to(self.device)
                    reward = reward.to(self.device)
                    loss_mask = loss_mask.to(self.device)
                    
                    if self.debug_logging:
                        self.logger.debug(f"train: micro-batch {micro_idx}, inputs shape={inputs.shape}")
                        self.logger.debug(f"train: micro-batch {micro_idx}, reward mean={reward.mean().item():.6f}")
                        self.logger.debug(f"train: micro-batch {micro_idx}, active tokens in loss_mask={loss_mask.sum().item()}")

                    loss = self.compute_loss(
                        inputs,
                        old_policy_log_probs,
                        reward,
                        mean_rewards,
                        std_rewards,
                        loss_mask
                    )
                    
                    if self.debug_logging:
                        self.logger.debug(f"train: micro-batch {micro_idx}, computed loss={loss.item():.6f}")
                        if torch.isnan(loss) or torch.isinf(loss):
                            self.logger.error(f"train: NaN or Inf loss detected! loss={loss.item()}")
                            self.logger.error(f"train: reward stats - mean={reward.mean().item()}, std={reward.std().item()}")
                            self.logger.error(f"train: loss_mask sum={loss_mask.sum().item()}")
                    
                    group_losses.append(loss.item())
                    loss.backward()
                    torch.cuda.empty_cache()

                self.optimizer.step()
                self.optimizer.zero_grad()
                
                avg_loss = sum(group_losses) / len(group_losses)
                avg_reward = reward.mean().item()
                
                if self.debug_logging:
                    self.logger.debug(f"train: optimizer step completed, avg_loss={avg_loss:.6f}, avg_reward={avg_reward:.6f}")
                    self.logger.debug(f"train: group_losses={group_losses}")
                    if avg_loss == 0.0:
                        self.logger.warning(f"train: ZERO LOSS DETECTED! This indicates a problem with loss computation")

                print(f"{idx:04d} loss: {avg_loss:.6f} reward: {avg_reward:.6f}")
                if self.log_wandb:
                    self.metrics["idx"].append(idx)
                    self.metrics["total_reward"].append(avg_reward)
                    self.metrics["loss"].append(avg_loss)

                torch.cuda.empty_cache()

            if self.debug_logging:
                self.logger.debug(f"train: completed iteration {idx}, total rewards mean={rewards.mean().item():.6f}")
            
            print(f"iter {idx}  >>> reward: {rewards.mean():.6f}")
            print(f"Total time: {str(datetime.timedelta(seconds=int(time.perf_counter() - start_time)))}")
            self.log_metrics()

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
from rich import print
import re
from typing import List

# Define EOS token for the model
EOS_TOKEN = "<|im_end|>"


SYSTEM_MESSAGE = (
    "You are a helpful assistant. You first think about the reasoning process in the mind "
    "and then provide the user with the answer."
)
PROMPT_TEMPLATE = (
    "Using the numbers {numbers}, create an equation that equals {target}. "
    "You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. "
    "Show your work in <think> </think> tags. And return the final equation and answer in "
    "<answer> </answer> tags, for example <answer>(1 + 2) / (3 * 5)</answer>."
)







def reward_func_len(sample: dict, s: str, *args, **kwargs):
    return 4 - (len(s)/1000)

def format_reward_func(completion: str) -> float:
    """
    Format: <think>...</think>\n<answer>...</answer>

    Also checks that the content within <answer>...</answer> conforms to a
    specified pattern (only digits, + - * / ( ) . and whitespace).

    Args:
        completion (str): Generated output

    Returns:
        float: Reward score
    """
    print(f"\n{'='*80}")
    print(f"FORMAT REWARD FUNCTION DEBUG:")
    print(f"Original completion length: {len(completion)}")
    print(f"Original completion (first 500 chars): {completion[:500]}")
    
    # Define the allowed pattern (only numbers, +, -, *, /, (, ), ., and whitespace)
    allowed_pattern = r"^[\d+\-*/().\s]+$"

    try:
        # Extract assistant response from Qwen3 format
        if "<|im_start|>assistant\n" in completion:
            completion = completion.split("<|im_start|>assistant\n")[1]
            print(f"After splitting on assistant tag: {completion[:200]}...")
        
        # Remove end tag if present
        if "<|im_end|>" in completion:
            completion = completion.split("<|im_end|>")[0]
            print(f"After removing end tag: {completion[:200]}...")
        
        # add synthetic <think> as its already part of the prompt and prefilled
        # for the assistant to more easily match the regex
        completion = "<think>" + completion
        print(f"After adding synthetic <think>: {completion[:200]}...")

        # Strip EOS token if present
        if completion.endswith(EOS_TOKEN):
            completion = completion[:-len(EOS_TOKEN)]
            print(f"After removing EOS token: {completion[:200]}...")

        # Check if the format is correct
        # Pattern means:
        # 1) <think>...contents not including other <think> tags...</think>
        # 2) \n
        # 3) <answer>...anything...</answer>
        regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"
        match = re.search(regex, completion, re.DOTALL)
        
        print(f"Regex pattern: {regex}")
        print(f"Match found: {match is not None}")
        
        if match is None or len(match.groups()) != 2:
            # Format is incorrect
            print(f"Format is incorrect, returning 0.0")
            return 0.0
        else:
            # Extract the content inside <answer>...</answer>
            think_content = match.group(1).strip()
            answer_content = match.group(2).strip()
            
            print(f"Think content: '{think_content[:100]}...'")
            print(f"Answer content: '{answer_content}'")

            # Check if answer content matches the allowed pattern
            if not re.match(allowed_pattern, answer_content):
                # If it doesn't match, reward is 0.5
                print(f"Answer content doesn't match allowed pattern, returning 0.5")
                return 0.5
            else:
                # If both format and pattern are correct, reward is 1
                print(f"Perfect format and pattern, returning 1.0")
                return 1.0
    except Exception as e:
        # Any error leads to 0 reward
        print(f"Exception occurred: {e}, returning 0.0")
        return 0.0
    finally:
        print(f"{'='*80}\n")


def equation_reward_func(completion: str, nums: List[int], target: int) -> float:
    """
    Evaluates completion based on mathematical correctness of the answer

    Args:
        completion (str): Generated output
        target (int): Expected answer
        nums (list): Available numbers to use in the equation

    Returns:
        float: Reward score
    """
    print(f"\n{'='*80}")
    print(f"EQUATION REWARD FUNCTION DEBUG:")
    print(f"Expected target: {target}")
    print(f"Available numbers: {nums}")
    print(f"Completion (first 300 chars): {completion[:300]}")
    
    try:
        # Extract assistant response from Qwen3 format first
        if "<|im_start|>assistant\n" in completion:
            completion = completion.split("<|im_start|>assistant\n")[1]
            
        if "<|im_end|>" in completion:
            completion = completion.split("<|im_end|>")[0]
        
        # Check if the format is correct
        match = re.search(r"<answer>(.*?)<\/answer>", completion)
        if match is None:
            print(f"No <answer> tags found, returning 0.0")
            return 0.0
        
        # Extract the "answer" part from the completion
        equation = match.group(1).strip()
        print(f"Extracted equation: '{equation}'")
        
        # Extract all numbers from the equation
        used_numbers = [int(n) for n in re.findall(r"\d+", equation)]
        print(f"Numbers used in equation: {used_numbers}")

        # Check if all numbers are used exactly once
        if sorted(used_numbers) != sorted(nums):
            print(f"Numbers don't match! Used: {sorted(used_numbers)}, Expected: {sorted(nums)}")
            return 0.0
        
        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
        allowed_pattern = r"^[\d+\-*/().\s]+$"
        if not re.match(allowed_pattern, equation):
            print(f"Equation contains invalid characters, returning 0.0")
            return 0.0

        # Evaluate the equation with restricted globals and locals
        result = eval(equation, {"__builtins__": None}, {})
        print(f"Equation result: {result}")
        
        # Check if the equation is correct and matches the ground truth
        if abs(float(result) - float(target)) < 1e-5:
            print(f"Equation is correct! Returning 1.0")
            return 1.0
        else:
            print(f"Equation result {result} doesn't match target {target}, returning 0.0")
            return 0.0
    except Exception as e:
        # If evaluation fails, reward is 0
        print(f"Exception during evaluation: {e}, returning 0.0")
        return 0.0
    finally:
        print(f"{'='*80}\n")


# Wrapper functions to match the expected signature
def format_reward_wrapper(sample: dict, s: str, *args, **kwargs):
    return format_reward_func(s)

def equation_reward_wrapper(sample: dict, s: str, *args, **kwargs):
    nums = sample.get('nums', [])
    target = sample.get('answer', 0)
    return equation_reward_func(s, nums, target)


def process_example(example: dict):
    numbers = example["nums"]
    target = example['target']
    user_prompt = PROMPT_TEMPLATE.format(numbers=numbers, target=target)
    
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": user_prompt}
        ],
        "answer": target,
        "nums": numbers  # Keep nums for equation validation
    }


# Load and split dataset properly
dataset = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split='train')
dataset = dataset.map(process_example, remove_columns=[
            col for col in dataset.column_names if col not in ["prompt", "answer", "nums"]
        ], num_proc=12)

group_size = 4
micro_group_size =2
lr = 5e-6
weight_decay = 0.1
reward_functions = [
    format_reward_wrapper,
    equation_reward_wrapper,
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
    weight_decay=weight_decay,
    debug_logging=True
)

trainer.train()