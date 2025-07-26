import re
from typing import Dict, Any, Tuple, List

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


def debug_chat_template(tokenizer):
    messages = [{"role": "user", "content": "Hello"}]

    # add_generation_prompt
    print("=== add_generation_prompt ===")
    result_false = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
    result_true = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    print(f"False: {result_false}")
    print(f"True: {result_true}")

    # continue_final_message
    print("\n=== continue_final_message ===")
    messages_asst = [{"role": "assistant", "content": "Hello there"}]
    result_false = tokenizer.apply_chat_template(messages_asst, continue_final_message=False, tokenize=False)
    result_true = tokenizer.apply_chat_template(messages_asst, continue_final_message=True, tokenize=False)
    print(f"False: {result_false}")
    print(f"True: {result_true}")

    # tokenize
    print("\n=== tokenize ===")
    result_false = tokenizer.apply_chat_template(messages, tokenize=False)
    result_true = tokenizer.apply_chat_template(messages, tokenize=True)
    print(f"False: {result_false}")
    print(f"True: {result_true}")

    # padding
    print("\n=== padding ===")
    result_false = tokenizer.apply_chat_template(messages, padding=False, return_tensors="pt")
    result_true = tokenizer.apply_chat_template(messages, padding=True, return_tensors="pt")
    print(f"False: {result_false}")
    print(f"True: {result_true}")

    # truncation
    print("\n=== truncation ===")
    result_false = tokenizer.apply_chat_template(messages, truncation=False, max_length=50)
    result_true = tokenizer.apply_chat_template(messages, truncation=True, max_length=50)
    print(f"False: {result_false}")
    print(f"True: {result_true}")

    # max_length
    print("\n=== max_length ===")
    result_none = tokenizer.apply_chat_template(messages, max_length=None)
    result_50 = tokenizer.apply_chat_template(messages, max_length=50, truncation=True)
    print(f"None: {result_none}")
    print(f"50: {result_50}")

    # return_tensors
    print("\n=== return_tensors ===")
    result_none = tokenizer.apply_chat_template(messages)
    result_pt = tokenizer.apply_chat_template(messages, return_tensors="pt")
    print(f"None: {result_none}")
    print(f"pt: {result_pt}")


