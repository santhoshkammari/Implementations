import os
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import tiktoken
import numpy as np
import gradio as gr
import matplotlib.pyplot as plt
from datetime import datetime
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from huggingface_hub import HfApi
from tqdm.auto import tqdm


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        q = self.q_proj(x)  # ( batch_size, seq_len, d_model)
        k = self.k_proj(x)  # ( batch_size, seq_len, d_model)
        v = self.v_proj(x)  # ( batch_size, seq_len, d_model)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        attn_scores = attn_scores.masked_fill(mask, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)  # ( batch_size, num_heads,seq_len,head_dim)

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.out_proj(out)  # stackign to learnable parameters

        return out


class FeedForward(nn.Module):
    def __init__(self, d_model, ff_dim, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.ff_dim = ff_dim
        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(d_model, ff_dim)
        self.fc2 = nn.Linear(ff_dim, d_model)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feedforward = FeedForward(d_model, ff_dim, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output = self.attention(self.ln1(x))
        x = x + self.dropout(attn_output)

        ff_output = self.feedforward(self.ln2(x))
        x = x + self.dropout(ff_output)

        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, max_seq_len, d_model, num_heads,
                 num_layers, dropout=0.1, ff_dim_multiplier=4):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        # Dropout after embeddings
        self.emb_dropout = nn.Dropout(dropout)

        # Stack of transformer blocks
        ff_dim = d_model * ff_dim_multiplier
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        # Final layer norm and output projection
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, x):
        batch_size, seq_len = x.size()

        # Create position indices
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)

        # Get embeddings
        token_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(positions)

        # Combine embeddings and apply dropout
        x = token_emb + pos_emb
        x = self.emb_dropout(x)

        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Apply final layer norm
        x = self.ln_final(x)

        # Project to vocabulary
        logits = self.lm_head(x)

        return logits


def generate(model, start_tokens, max_new_tokens, temperature=1.0):
    model.eval()

    # Start with the provided tokens
    tokens = start_tokens.clone()

    for _ in range(max_new_tokens):
        # Get the context (limited to max_seq_len)
        context = tokens[:, -model.max_seq_len:]

        # Forward pass
        with torch.no_grad():
            logits = model(context)

        # Focus on the last token's prediction
        next_token_logits = logits[:, -1, :] / temperature

        # Apply softmax to get probabilities
        probs = F.softmax(next_token_logits, dim=-1)

        # Sample from the distribution
        next_token = torch.multinomial(probs, num_samples=1)

        # Add to sequence
        tokens = torch.cat((tokens, next_token), dim=1)

    return tokens


# Tokenizer and dataset handling
class TextDataset(Dataset):
    def __init__(self, texts, max_length, tokenizer):
        self.texts = texts
        self.max_length = max_length
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        try:
            text = self.texts[idx]
            # Make sure text is not empty
            if not text or len(text.strip()) == 0:
                text = "empty text"

            tokens = self.tokenizer.encode(text, allowed_special={"<|endoftext|>"})

            # Ensure we have at least 2 tokens (for input and target)
            if len(tokens) < 2:
                tokens = tokens + [0] * (2 - len(tokens))

            # Truncate or pad as needed
            if len(tokens) > self.max_length:
                tokens = tokens[:self.max_length]
            else:
                # Add padding if needed
                tokens = tokens + [0] * (self.max_length - len(tokens))

            return torch.tensor(tokens)
        except Exception as e:
            print(f"Error processing item {idx}: {e}")
            # Return a safe fallback
            return torch.tensor([0] * self.max_length)
# Utility functions
def calculate_model_size(vocab_size, max_seq_len, d_model, num_heads, num_layers):
    # Token embedding
    params = vocab_size * d_model

    # Position embedding
    params += max_seq_len * d_model

    # Each transformer block
    for _ in range(num_layers):
        # Multi-head attention
        params += 4 * d_model * d_model  # Q, K, V, and output projections

        # Feed-forward network
        ff_dim = d_model * 4
        params += d_model * ff_dim + ff_dim * d_model

        # Layer norms
        params += 2 * d_model * 2  # Two layer norms, each with weight and bias

    # Final layer norm
    params += d_model * 2

    # Output projection
    params += d_model * vocab_size

    # Convert to millions
    return params / 1_000_000


def setup_hf_dataset(dataset_name, text_column, split, sample_size, hf_token):
    try:
        dataset = load_dataset(dataset_name, split=split, trust_remote_code=True, token=hf_token)

        if sample_size and sample_size < len(dataset):
            dataset = dataset.select(range(sample_size))

        texts = dataset[text_column]
        return texts, f"Successfully loaded {len(texts)} samples from {dataset_name}"
    except Exception as e:
        return None, f"Error loading dataset: {str(e)}"


def prepare_dataloader(texts, max_seq_len, batch_size, tokenizer):
    dataset = TextDataset(texts, max_seq_len, tokenizer)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=2
    )


def save_model(model, model_name):
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')

    save_path = os.path.join('saved_models', f"{model_name}.pt")
    torch.save(model.state_dict(), save_path)
    return save_path


def load_model(model_path, model_params):
    vocab_size = model_params['vocab_size']
    max_seq_len = model_params['max_seq_len']
    d_model = model_params['d_model']
    num_heads = model_params['num_heads']
    num_layers = model_params['num_layers']
    dropout = model_params['dropout']

    model = GPT(vocab_size, max_seq_len, d_model, num_heads, num_layers, dropout)
    model.load_state_dict(torch.load(model_path))
    return model


def plot_losses(losses):
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.grid(True)

    # Save the plot to a temporary file
    if not os.path.exists('temp'):
        os.makedirs('temp')
    plot_path = os.path.join('temp', 'loss_plot.png')
    plt.savefig(plot_path)
    plt.close()

    return plot_path


# Training function with UI updates
def train_model(
    model,
    data_loader,
    num_epochs,
    learning_rate,
    device,
    save_every,
    model_save_name,
    progress=None,
    log_callback=None,
    loss_plot_callback=None
):
    print(f"train_model called with device={device}, epochs={num_epochs}, batches={len(data_loader)}")

    # Move model to device
    print(f"Moving model to {device}")
    # Move model to device
    model = model.to(device)

    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # For tracking
    all_losses = []
    step = 0
    checkpoint_prefix = model_save_name

    # Training loop
    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch+1}/{num_epochs}")
        model.train()
        epoch_losses = []

        # Update progress
        if progress:
            progress(epoch / num_epochs, f"Starting epoch {epoch + 1}/{num_epochs}")

        # Use tqdm for batch progress
        for batch_idx, batch in enumerate(tqdm(data_loader, desc=f"Epoch {epoch + 1}")):
           try:
               # Get input tokens and move to device
               input_ids = batch.to(device)

               print(f"Batch {batch_idx} shape: {input_ids.shape}")
               # Make sure we have enough sequence length
               if input_ids.shape[1] <= 1:
                   print(f"Skipping batch {batch_idx}: sequence too short ({input_ids.shape[1]})")
                   continue
               # Forward pass
               logits = model(input_ids[:, :-1])  # all but last token
               print(f"Logits shape: {logits.shape}")

               # Calculate loss
               targets = input_ids[:, 1:]  # all but first token
               print(f"Targets shape: {targets.shape}")

               loss = F.cross_entropy(
                   logits.reshape(-1, model.vocab_size),
                   targets.reshape(-1)
               )

               # Backward and optimize
               optimizer.zero_grad()
               loss.backward()
               optimizer.step()

               # Record loss
               loss_value = loss.item()
               epoch_losses.append(loss_value)
               all_losses.append(loss_value)

               # Update step
               step += 1

               # Log
               if batch_idx % 10 == 0:
                   log_message = f"Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss_value:.4f}"
                   if log_callback:
                       log_callback(log_message)

                   # Update loss plot
                   if loss_plot_callback and step % 50 == 0:
                       plot_path = plot_losses(all_losses)
                       loss_plot_callback(plot_path)

               # Save checkpoint if requested
               if save_every > 0 and step % save_every == 0:
                   checkpoint_name = f"{checkpoint_prefix}_step{step}_epoch{epoch + 1}"
                   save_path = save_model(model, checkpoint_name)
                   if log_callback:
                       log_callback(f"Saved checkpoint to {save_path}")
           except Exception as e:
               print(f"Error in batch {batch_idx}: {e}")
               import traceback
               traceback.print_exc()
               continue

               # End of epoch
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        log_message = f"Epoch {epoch + 1} complete. Average loss: {avg_loss:.4f}"
        if log_callback:
            log_callback(log_message)

        # Update progress
        if progress:
            progress((epoch + 1) / num_epochs, log_message)

        # Save model at end of epoch
        epoch_save_name = f"{checkpoint_prefix}_epoch{epoch + 1}"
        save_path = save_model(model, epoch_save_name)
        if log_callback:
            log_callback(f"Saved model at end of epoch to {save_path}")

    # Save final model
    final_save_path = save_model(model, f"{checkpoint_prefix}_final")
    if log_callback:
        log_callback(f"Training complete. Final model saved to {final_save_path}")

    # Final loss plot
    if loss_plot_callback:
        plot_path = plot_losses(all_losses)
        loss_plot_callback(plot_path)

    return model, final_save_path, all_losses

# Text generation function
def generate_text(model, prompt, max_tokens, temperature, tokenizer, device):
    model.eval()
    model = model.to(device)

    # Encode the prompt
    prompt_tokens = tokenizer.encode(prompt, allowed_special={"<|endoftext|>"})
    prompt_tensor = torch.tensor([prompt_tokens], device=device)

    # Generate
    with torch.no_grad():
        output_tokens = generate(model, prompt_tensor, max_tokens, temperature)

    # Decode
    output_text = tokenizer.decode(output_tokens[0].cpu().tolist())

    return output_text


# Gradio UI
def create_ui():
    # Set up tokenizer
    tokenizer = tiktoken.get_encoding("cl100k_base")

    # Default HF token
    default_hf_token = "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"

    # State variables
    state = {
        "model": None,
        "model_params": None,
        "dataset_loaded": False,
        "texts": None,
        "training_in_progress": False,
        "logs": [],
        "available_models": []
    }

    # Update available models
    # Update available models
    def update_model_list():
        if os.path.exists('saved_models'):
            models = [f for f in os.listdir('saved_models') if f.endswith('.pt')]
            state["available_models"] = models
            return models  # Just return the list of models
        return []  # Return an empty list if there are no models

    # Calculate parameter count
    def update_param_count(vocab_size, max_seq_len, d_model, num_heads, num_layers):
        param_count = calculate_model_size(vocab_size, max_seq_len, d_model, num_heads, num_layers)
        return f"{param_count:.2f}M parameters"

    # Load dataset
    def hf_load_dataset(dataset_name, text_column, split, sample_size, hf_token, dataset_config=None):
        if not hf_token:
            hf_token = default_hf_token

        try:
            dataset = load_dataset(
                dataset_name,
                dataset_config,  # Add the config parameter here
                split=split,
                trust_remote_code=True,
                token=hf_token
            )

            if sample_size and sample_size < len(dataset):
                dataset = dataset.select(range(sample_size))

            texts = dataset[text_column]
            state["texts"] = texts  # Save texts in state
            state["dataset_loaded"] = True  # Set the flag to indicate dataset is loaded
            return texts, f"Successfully loaded {len(texts)} samples from {dataset_name}"
        except Exception as e:
            return None, f"Error loading dataset: {str(e)}"

    # Start training
    def start_training(
        vocab_size, max_seq_len, d_model, num_heads, num_layers,
        dropout, batch_size, learning_rate, num_epochs, save_every,
        model_name, device
    ):
        print(f"Start training called with: dataset_loaded={state['dataset_loaded']}")

        if not state["dataset_loaded"]:
            print("Dataset not loaded, returning early")
            return "Please load a dataset first", None

        if state["training_in_progress"]:
            print("Training already in progress, returning early")
            return "Training already in progress", None

        print(f"Creating model with vocab_size={vocab_size}, d_model={d_model}, layers={num_layers}")


        # Generate model name if not provided
        if not model_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"gpt_{timestamp}"

        # Create model
        model = GPT(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )

        # Save model parameters
        state["model_params"] = {
            "vocab_size": vocab_size,
            "max_seq_len": max_seq_len,
            "d_model": d_model,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "dropout": dropout
        }

        # Prepare data loader
        tokenizer = tiktoken.get_encoding("cl100k_base")
        data_loader = prepare_dataloader(
            state["texts"],
            max_seq_len,
            batch_size,
            tokenizer
        )

        # Clear logs
        state["logs"] = []

        def update_progress(value, message):
            print(f"Progress update: {value:.2f} - {message}")
            return value  # Just return the value instead of using .update()

        # Training function that updates Gradio components
        def training_thread():
            nonlocal  model
            print("Training thread started")
            state["training_in_progress"] = True

            # Train model
            try:
                print(f"Starting model training with {len(data_loader)} batches")
                trained_model, save_path, losses = train_model(
                    model=model,
                    data_loader=data_loader,
                    num_epochs=num_epochs,
                    learning_rate=learning_rate,
                    device=device,
                    save_every=save_every,
                    model_save_name=model_name,
                    progress=update_progress,
                    log_callback=lambda msg: state["logs"].append(msg),
                    loss_plot_callback=lambda path: loss_plot.update(value=path)
                )
                print(f"Training completed, saving model parameters to {model_name}_params.json")

                # Save model parameters
                params_path = os.path.join('saved_models', f"{model_name}_params.json")
                with open(params_path, 'w') as f:
                    json.dump(state["model_params"], f)

                # Update model list
                update_model_list()

                # Update state
                state["model"] = trained_model  # Use the trained_model returned from train_model


            except Exception as e:
                print(f'Exception as {e}')
                state["logs"].append(f"Error during training: {str(e)}")
            finally:
                state["training_in_progress"] = False

        # Start training in a separate thread
        import threading
        thread = threading.Thread(target=training_thread)
        thread.start()

        return "Training started", None

    # Update logs
    def update_logs():
        print(f"Updating logs, {len(state['logs'])} log entries")
        if state["logs"]:
            return "\n".join(state["logs"])
        return "No logs yet"

    # Load model for inference
    # Load model for inference
    def load_model_for_inference(model_name):
        if not model_name:
            return "No model selected"

        try:
            model_path = os.path.join('saved_models', model_name)
            params_path = os.path.join('saved_models', model_name.replace('.pt', '_params.json'))

            # Load parameters
            if os.path.exists(params_path):
                with open(params_path, 'r') as f:
                    params = json.load(f)
                    print(f"Loaded parameters from {params_path}: {params}")
            else:
                # Try to infer parameters from the model file
                print(f"No parameter file found for {model_name}, attempting to infer from state dict")
                state_dict = torch.load(model_path, map_location='cpu')

                # Infer d_model from embedding dimensions
                if 'token_embedding.weight' in state_dict:
                    d_model = state_dict['token_embedding.weight'].shape[1]
                else:
                    d_model = 512  # Default fallback

                # Count number of transformer blocks
                num_layers = 0
                while f'transformer_blocks.{num_layers}.ln1.weight' in state_dict:
                    num_layers += 1

                # Infer number of heads from attention dimensions
                if 'transformer_blocks.0.attention.q_proj.weight' in state_dict:
                    d_head = state_dict['transformer_blocks.0.attention.q_proj.weight'].shape[
                                 0] // 8  # Assuming 8 heads
                    num_heads = state_dict['transformer_blocks.0.attention.q_proj.weight'].shape[0] // d_head
                else:
                    num_heads = 8  # Default fallback

                params = {
                    "vocab_size": 100000,  # Default value
                    "max_seq_len": 1024,  # Default value
                    "d_model": d_model,
                    "num_heads": num_heads,
                    "num_layers": num_layers,
                    "dropout": 0.1  # Default value
                }
                print(f"Inferred parameters: {params}")

                # Save the inferred parameters for future use
                with open(params_path, 'w') as f:
                    json.dump(params, f)

            # Create model with correct architecture
            model = GPT(
                vocab_size=params["vocab_size"],
                max_seq_len=params["max_seq_len"],
                d_model=params["d_model"],
                num_heads=params["num_heads"],
                num_layers=params["num_layers"],
                dropout=params["dropout"]
            )

            # Load state dict
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            state["model"] = model
            state["model_params"] = params

            return f"Model {model_name} loaded successfully with parameters: d_model={params['d_model']}, layers={params['num_layers']}, heads={params['num_heads']}"

        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error loading model: {str(e)}"

    # Generate text
    def inference(prompt, max_tokens, temperature, model_name, device):
        if not model_name:
            return "Please select a model first"

        # Load model if not already loaded
        if state["model"] is None:
            load_model_for_inference(model_name)

        # Generate text
        tokenizer = tiktoken.get_encoding("cl100k_base")
        try:
            output = generate_text(
                model=state["model"],
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                tokenizer=tokenizer,
                device=device
            )
            return output
        except Exception as e:
            return f"Error generating text: {str(e)}"

    # Define the UI
    with gr.Blocks(title="GPT Model Trainer") as app:
        gr.Markdown("# GPT Model Trainer")

        with gr.Tabs():
            # Dataset tab
            with gr.TabItem("Dataset"):
                gr.Markdown("## Load Dataset from HuggingFace")

                with gr.Row():
                    dataset_name = gr.Textbox(label="Dataset Name", value="Salesforce/wikitext", info="HuggingFace dataset name")
                    dataset_config = gr.Textbox(label="Dataset Config", value="wikitext-2-raw-v1",
                                                info="Optional dataset configuration (e.g. '20231101.ab')")
                    text_column = gr.Textbox(label="Text Column", value="text", info="Column containing text data")

                with gr.Row():
                    split = gr.Textbox(label="Split", value="train", info="Dataset split (train, test, etc.)")
                    sample_size = gr.Number(label="Sample Size", value=10, info="Number of samples (None for all)")
                    hf_token = gr.Textbox(label="HuggingFace Token", value=default_hf_token, type="password")

                with gr.Row():
                    load_button = gr.Button("Load Dataset")
                    dataset_status = gr.Textbox(label="Status", interactive=False)

                # Update the click event to include the new parameter
                load_button.click(
                    hf_load_dataset,
                    inputs=[dataset_name, text_column, split, sample_size, hf_token, dataset_config],
                    outputs=dataset_status
                )

            # Training tab
            with gr.TabItem("Training"):
                gr.Markdown("## Model Architecture")

                with gr.Row():
                    vocab_size = gr.Slider(label="Vocabulary Size", minimum=1000, maximum=100000, value=100000,
                                           step=1000)
                    max_seq_len = gr.Slider(label="Max Sequence Length", minimum=128, maximum=2048, value=1024,
                                            step=128)

                with gr.Row():
                    d_model = gr.Slider(label="Model Dimension", minimum=128, maximum=1024, value=768, step=64)
                    num_heads = gr.Slider(label="Number of Attention Heads", minimum=1, maximum=16, value=12, step=1)
                    num_layers = gr.Slider(label="Number of Layers", minimum=1, maximum=24, value=12, step=1)

                with gr.Row():
                    dropout = gr.Slider(label="Dropout Rate", minimum=0.0, maximum=0.5, value=0.1, step=0.05)
                    param_count = gr.Textbox(label="Parameter Count", value="", interactive=False)

                gr.Markdown("## Training Parameters")

                with gr.Row():
                    batch_size = gr.Slider(label="Batch Size", minimum=1, maximum=128, value=1, step=1)
                    learning_rate = gr.Slider(label="Learning Rate", minimum=1e-5, maximum=1e-3, value=3e-4, step=1e-5)
                    num_epochs = gr.Slider(label="Number of Epochs", minimum=1, maximum=100, value=3, step=1)

                with gr.Row():
                    save_every = gr.Slider(label="Save Every N Steps", minimum=0, maximum=1000, value=500, step=50,
                                           info="0 to disable")
                    model_name = gr.Textbox(label="Model Save Name", value="", info="Leave blank for automatic naming")
                    device = gr.Radio(label="Device", choices=["cuda", "cpu"], value="cuda")

                # Update parameter count when architecture changes
                for param in [vocab_size, max_seq_len, d_model, num_heads, num_layers]:
                    param.change(
                        update_param_count,
                        inputs=[vocab_size, max_seq_len, d_model, num_heads, num_layers],
                        outputs=param_count
                    )

                with gr.Row():
                    start_button = gr.Button("Start Training")
                    training_status = gr.Textbox(label="Status", interactive=False)

                with gr.Row():
                    progress_bar = gr.Slider(label="Training Progress", minimum=0, maximum=1, value=0,
                                             interactive=False)

                with gr.Row():
                    gr.Markdown("## Training Logs")
                    logs = gr.Textbox(label="Logs", interactive=False, lines=10)
                    log_refresh = gr.Button("Refresh Logs")

                with gr.Row():
                    gr.Markdown("## Loss Plot")
                    loss_plot = gr.Image(label="Loss Plot", interactive=False)

                start_button.click(
                    start_training,
                    inputs=[
                        vocab_size, max_seq_len, d_model, num_heads, num_layers,
                        dropout, batch_size, learning_rate, num_epochs, save_every,
                        model_name, device
                    ],
                    outputs=[training_status, loss_plot]
                )

                log_refresh.click(update_logs, outputs=logs)

            # Inference tab
            with gr.TabItem("Inference"):
                gr.Markdown("## Generate Text")

                with gr.Row():
                    model_dropdown = gr.Dropdown(label="Select Model", choices=[], interactive=True)
                    refresh_models = gr.Button("Refresh Models")
                    load_model_button = gr.Button("Load Model")
                    model_load_status = gr.Textbox(label="Status", interactive=False)

                with gr.Row():
                    inference_device = gr.Radio(label="Device", choices=["cuda", "cpu"], value="cuda")

                with gr.Row():
                    prompt = gr.Textbox(label="Prompt", lines=3, placeholder="Enter your prompt here...")
                    max_tokens = gr.Slider(label="Max Tokens", minimum=1, maximum=1000, value=100, step=1)
                    temperature = gr.Slider(label="Temperature", minimum=0.1, maximum=2.0, value=1.0, step=0.1)

                with gr.Row():
                    generate_button = gr.Button("Generate")
                    output_text = gr.Textbox(label="Generated Text", lines=10, interactive=False)

                # Refresh model list
                refresh_models.click(update_model_list, outputs=model_dropdown)

                # Load model
                load_model_button.click(
                    load_model_for_inference,
                    inputs=[model_dropdown],
                    outputs=model_load_status
                )

                # Generate text
                generate_button.click(
                    inference,
                    inputs=[prompt, max_tokens, temperature, model_dropdown, inference_device],
                    outputs=output_text
                )

    # Initialize
        # Initialize
        def init_fn():
            models = update_model_list()
            return gr.update(choices=models)  # Use gr.update instead of gr.Dropdown.update

        app.load(
            fn=init_fn,
            inputs=None,
            outputs=model_dropdown
        )

    return app


if __name__ == "__main__":
    app = create_ui()
    app.launch(share=False,server_name="0.0.0.0")