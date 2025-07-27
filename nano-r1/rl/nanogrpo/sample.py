from transformers import AutoTokenizer

tokenizer_model = "Qwen/Qwen3-1.7B"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)

prompt = [
        {"role": "system", "content": 'systemmessage'},
        {"role": "user", "content": "usermessage"},
    # {"role":"assistant","content":"Lets Think step-by-step and solve problem, <think>"}
    ]
formatted = tokenizer.apply_chat_template(prompt, add_generation_prompt=True,
                                          # continue_final_message=True,
                                          enable_thinking=True,
                                                           tokenize=False)

print(formatted)