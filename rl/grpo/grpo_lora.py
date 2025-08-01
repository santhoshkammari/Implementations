import re

from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

MAX_COMPLETION_LENGTH = 1024


# prepare dataset
def prepare_prompts(example):
    SYSTEM_PROMPT = """\
    You are a helpful assistant. You first think about the reasoning process and then provide the user with the answer.

    Put your thinking process in <reasoning> tags.
    - As you're reasoning, say "Wait," and think more to check your work until you're confident.

    Put just the final answer in <answer> tags.

    Respond in the following format:
    <reasoning>
    ...
    </reasoning>
    <answer>
    ...
    </answer>
    """
    output = {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["question"]},
        ],
        "answer": None if "####" not in example['answer'] else example['answer'].split("####")[1].strip()
    }
    return output


dataset = load_dataset("openai/gsm8k", "main")
dataset = dataset.map(prepare_prompts)
print(dataset)


# Prepare rewards
def extract_completion_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def extract_completion_reasoning(text: str) -> str:
    reasoning = text.split("<reasoning>")[-1]
    reasoning = reasoning.split("</reasoning>")[0]
    return reasoning.strip()


def debug_reward(q, answer, responses, extracted_responses, rewards):
    print(
        "--------------",
        f"Q: {q}",
        "--------------",
        f"A: {answer[0]}",
        "--------------",
        f"Response:\n{responses[0]}",
        "--------------",
        f"Extracted: {extracted_responses}",
        "--------------",
        f"Reward: {rewards}",
        "--------------",
        sep="\n",
    )


def correctness_reward(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    q = prompts[0][-1]["content"]
    extracted_responses = [extract_completion_answer(r) for r in responses]

    def extract_float(text: str):
        text_no_commas = text.replace(",", "")
        match = re.search(r"([+-]?\d+(?:\.\d+)?)", text_no_commas)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
        return None

    rewards = []
    for extracted_text, gold_text in zip(extracted_responses, answer):
        num_extracted = extract_float(extracted_text)
        num_gold = extract_float(gold_text)

        if (num_extracted is not None) and (num_gold is not None):
            if abs(num_extracted - num_gold) < 1e-6:
                rewards.append(2.0)
            else:
                rewards.append(0.0)
        else:
            rewards.append(0.0)

    debug_reward(q, answer, responses, extracted_responses, rewards)

    return rewards


def strict_format_reward(completions, **kwargs) -> list[float]:
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n?$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def soft_format_reward(completions, **kwargs) -> list[float]:
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def xmlcount_reward(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]

    def count_xml(text) -> float:
        count = 0.0
        if text.count("<reasoning>\n") == 1:
            count += 0.125
        if text.count("\n</reasoning>\n") == 1:
            count += 0.125
        if text.count("\n<answer>\n") == 1:
            count += 0.125
            count -= len(text.split("\n</answer>\n")[-1]) * 0.001
        if text.count("\n</answer>") == 1:
            count += 0.125
            count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
        return count

    return [count_xml(c) for c in contents]


def length_reward(completions, **kwargs) -> list[float]:
    chars_per_token = 4
    responses = [completion[0]["content"] for completion in completions]
    reasoning = [extract_completion_reasoning(r) for r in responses]
    return [2 * (len(r)) / (MAX_COMPLETION_LENGTH * chars_per_token) for r in reasoning]


## Prepare model
from unsloth import FastLanguageModel, PatchFastRL, is_bfloat16_supported

MODEL_NAME = "Qwen/Qwen3-0.5B"
LORA_RANK = 64
GPU_MEMORY_UTILIZATION = 0.7
MAX_SEQ_LENGTH = 1024 + 256 + 8

PatchFastRL(algorithm="GRPO", FastLanguageModel=FastLanguageModel)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
    fast_inference=True,
    max_lora_rank=LORA_RANK,
    gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=LORA_RANK,
    use_gradient_checkpointing="unsloth",
    random_state=3407
)

# train

MAX_PROMPT_LENGTH = 256
MAX_COMPLETION_LENGTH = 1024
NUM_GENERATIONS = 12
MAX_STEPS = 1000

training_args = GRPOConfig(
    use_vllm=True,
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="adamw_8bit",
    logging_steps=1,
    bf16=is_bfloat16_supported(),
    fp16=not is_bfloat16_supported(),
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_generations=NUM_GENERATIONS,
    max_prompt_length=MAX_PROMPT_LENGTH,
    max_completion_length=MAX_COMPLETION_LENGTH,
    # num_train_epochs = 1, # set to 1 for full training run
    max_steps=MAX_STEPS,
    save_steps=500,
    max_grad_norm=0.1,
    report_to="wandb",
    output_dir="qwen3-grpo",
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        xmlcount_reward,
        soft_format_reward,
        strict_format_reward,
        correctness_reward,
        length_reward,
    ],  # type: ignore
    args=training_args,
    train_dataset=dataset,
)

trainer.train()


