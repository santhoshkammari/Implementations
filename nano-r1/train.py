import re
import time
from functools import lru_cache
from typing import Dict, List, Tuple, Any
import torch

from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import SYSTEM_MESSAGE, PROMPT_TEMPLATE

MODEL_NAME = "HuggingFaceTB/SmolLM-135M"  # Use -Instruct directly
DATASET_NAME = "PREPROCESSED_DATA" #"Jiayi-Pan/Countdown-Tasks-3to4"
TEST_SIZE = 2
TRAIN_SIZE = 8
SEED = 42

device = "cpu"

@lru_cache(maxsize=1)  # Only need 1 cached model
def load_model():
    """Load and cache the model - only runs once"""
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",  # Let transformers pick best dtype
        low_cpu_mem_usage=True  # Saves RAM during loading
    ).to(device)
    return model

@lru_cache(maxsize=1)  # Cache tokenizer too
def load_tokenizer():
    """Load and cache tokenizer"""
    return AutoTokenizer.from_pretrained(MODEL_NAME+"-Instruct")


def preprocess_example(example):
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": PROMPT_TEMPLATE.format(numbers=example['nums'],target = example["target"])},
        {"role":"assistant","content":"<think>"}
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True
    )
    prompt = tokenizer.decode(input_ids, skip_special_tokens=False,clean_up_tokenization_spaces=False)
    return {"input_ids":input_ids,"prompt":prompt}

def format_reward_func(completion: str) -> float:
    """
    Format: <think>...</think>\n</answer>...</answer>

    Also checks that the content within <answer>...</answer> conforms to a
    specified pattern (only digits, + - * / ( ) . and whitespace).

    Args:
        completion (str): Generated output

    Returns:
        float: Reward score
    """
    # Define the allowed pattern (only numbers, +, -, *, /, (, ), ., and whitespace)
    allowed_pattern = r"^[\d+\-*/().\s]+$"

    try:
        # add synthetic <think> as its already part of the prompt and prefilled
        # for the assistant to more easily match the regex
        completion = "<think>" + completion

        # Strip EOS token if present
        if completion.endswith(EOS_TOKEN):
            completion = completion[:-len(EOS_TOKEN)]

        # Check if the format is correct
        # Pattern means:
        # 1) <think>...contents not including other <think> tags...</think>
        # 2) \n
        # 3) <answer>...anything...</answer>
        regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"
        match = re.search(regex, completion, re.DOTALL)

        if match is None or len(match.groups()) != 2:
            # Format is incorrect
            return 0.0
        else:
            # Extract the content inside <answer>...</answer>
            answer_content = match.group(2).strip()

            # Check if answer content matches the allowed pattern
            if not re.match(allowed_pattern, answer_content):
                # If it doesn't match, reward is 0.5
                return 0.5
            else:
                # If both format and pattern are correct, reward is 1
                return 1.0
    except Exception:
        # Any error leads to 0 reward
        return 0.0


def equation_reward_func(completion: str, nums: List[int], target: int) -> float:
    """
    Evaluates completion based on mathematical correctness of the answer

    Args:
        completion (str): Generated output
        target (str): Expected answer
        nums (list): Available numbers to use in the equation

    Returns:
        float: Reward score
    """
    try:
        # Check if the format is correct
        match = re.search(r"<answer>(.*?)<\/answer>", completion)
        if match is None:
            return 0.0
        # Extract the "answer" part from the completion
        equation = match.group(1).strip()
        # Extract all numbers from the equation
        used_numbers = [int(n) for n in re.findall(r"\d+", equation)]

        # Check if all numbers are used exactly once
        if sorted(used_numbers) != sorted(nums):
            return 0.0
        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
        allowed_pattern = r"^[\d+\-*/().\s]+$"
        if not re.match(allowed_pattern, equation):
            return 0.0

        # Evaluate the equation with restricted globals and locals
        result = eval(equation, {"__builtins__": None}, {})
        # Check if the equation is correct and matches the ground truth
        if abs(float(result) - float(target)) < 1e-5:
            return 1.0
        else:
            return 0.0
    except Exception:
        # If evaluation fails, reward is 0
        return 0.0


def compute_reward(completion: str, sample: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    nums = sample["nums"]
    target = sample["target"]

    format_reward = format_reward_func(completion)
    equation_reward = equation_reward_func(
        completion=completion, nums=nums, target=target
    )

    reward = format_reward + equation_reward

    metrics = {
        "format_reward": format_reward,
        "equation_reward": equation_reward,
    }

    return reward, metrics


def generate_completions(model,tokenizer,prompt: str, n: int = 1, max_tokens: int = 100, temperature: float = 0.8) -> List[str]:
    """
    Generate n completions for a given prompt

    Args:
        prompt (str): Input prompt
        n (int): Number of generations
        max_tokens (int): Max tokens to generate
        temperature (float): Sampling temperature

    Returns:
        List[str]: List of generated completions
    """
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_length = inputs.input_ids.shape[1]

    completions = []

    with torch.no_grad():
        for _ in range(n):
            # Generate with sampling
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

            # Extract only the new tokens (remove input prompt)
            generated_tokens = outputs[0][input_length:]
            completion = tokenizer.decode(generated_tokens, skip_special_tokens=True,
                                          clean_up_tokenization_spaces=False)
            completions.append(completion)

    return completions

# Usage
# st = time.perf_counter()
model = load_model()
tokenizer = load_tokenizer()
EOS_TOKEN_ID = tokenizer.eos_token_id #2
EOS_TOKEN = tokenizer.convert_ids_to_tokens(EOS_TOKEN_ID)  #<|im_end|>


# dataset = load_dataset(DATASET_NAME, split="train")
# dataset = dataset.map(preprocess_example, num_proc=8)
# dataset.save_to_disk('PREPROCESSED_DATA')
#
dataset = load_from_disk(DATASET_NAME)

train_test_spilt = dataset.train_test_split(test_size=TEST_SIZE,train_size=TRAIN_SIZE,seed=SEED)
train_dataset = train_test_spilt['train']
test_dataset = train_test_spilt['test']



prompt = "what is 2+3?"
n_generations = 5  # Generate 5 different completions

completions = generate_completions(
    model=model,
    tokenizer=tokenizer,
    prompt=prompt,
    n=n_generations,
    max_tokens=100,
    temperature=0.8
)

for i, completion in enumerate(completions):
    print(f"Generation {i+1}: {completion}")
    print("-" * 50)







