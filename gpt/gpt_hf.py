import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from transformers import GPT2LMHeadModel, GPT2Config, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

# 1. Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. Load or create tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# 3. Initialize a new GPT-2 model configuration (even smaller for quick testing)
config = GPT2Config(
    vocab_size=len(tokenizer),
    n_positions=64,
    n_ctx=64,
    n_embd=128,
    n_layer=2,
    n_head=2,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

# 4. Create a new model
model = GPT2LMHeadModel(config)
model.to(device)
print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")

# 5. Load and prepare dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# Take only 100 samples
train_dataset = dataset["train"].select(range(100))
eval_dataset = dataset["validation"].select(range(20))

print(f"Training on {len(train_dataset)} samples")
print(f"Sample text: {train_dataset[0]['text'][:100]}...")


def tokenize_function(examples):
    outputs = tokenizer(
        examples["text"],
        truncation=True,
        max_length=64,
        padding="max_length",
        return_tensors="pt"
    )
    return outputs


tokenized_train = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=train_dataset.column_names
)

tokenized_eval = eval_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=eval_dataset.column_names
)

# 6. Create dataloaders
tokenized_train.set_format("torch")
tokenized_eval.set_format("torch")

train_dataloader = DataLoader(
    tokenized_train,
    batch_size=4,
    shuffle=True
)
eval_dataloader = DataLoader(
    tokenized_eval,
    batch_size=4
)

# 7. Optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# 8. Training loop
num_epochs = 5  # More epochs for small dataset
print(f"Starting training for {num_epochs} epochs")

for epoch in range(num_epochs):
    # Training
    model.train()
    total_train_loss = 0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")

    for i, batch in enumerate(progress_bar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids
        )

        loss = outputs.loss
        total_train_loss += loss.item()

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        # Update progress bar
        progress_bar.set_postfix({"loss": loss.item()})

    avg_train_loss = total_train_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1} - Average training loss: {avg_train_loss:.4f}")

    # Evaluation
    model.eval()
    total_eval_loss = 0

    with torch.no_grad():
        for batch in eval_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )

            loss = outputs.loss
            total_eval_loss += loss.item()

    avg_eval_loss = total_eval_loss / len(eval_dataloader)
    print(f"Epoch {epoch + 1} - Average evaluation loss: {avg_eval_loss:.4f}")

    # Generate a sample after each epoch to see progress
    if epoch % 1 == 0:
        model.eval()
        prompt = "The quick brown fox"
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=30,
                num_return_sequences=1,
                temperature=0.8,
                do_sample=True
            )

        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"Sample generation: {generated_text}")

# Save the trained miniature model
model.save_pretrained("./mini-gpt2-from-scratch")
tokenizer.save_pretrained("./mini-gpt2-from-scratch")
print("Training complete and model saved!")