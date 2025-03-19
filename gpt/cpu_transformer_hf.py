import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from typing import Optional, Tuple
import numpy as np
from torch.utils.checkpoint import checkpoint

# Add these imports
from datasets import load_dataset



class CPUEfficientCausalAttention(nn.Module):
    """
    Causal self-attention mechanism optimized for CPU execution.
    """

    def __init__(self, hidden_dim, num_heads, dropout=0.1, block_size=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout = dropout
        self.block_size = block_size

        assert self.head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"

        # Combined QKV projection for better CPU utilization
        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.dropout_layer = nn.Dropout(dropout)

        # Scaling factor for attention scores
        self.scale = self.head_dim ** -0.5

    def _split_heads(self, x, batch_size):
        """Split the hidden dimension into heads."""
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)  # [batch_size, num_heads, seq_len, head_dim]

    def _merge_heads(self, x, batch_size):
        """Merge the heads back into hidden dimension."""
        x = x.permute(0, 2, 1, 3)  # [batch_size, seq_len, num_heads, head_dim]
        return x.reshape(batch_size, -1, self.hidden_dim)

    def forward(self, x, attention_mask=None, key_padding_mask=None):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_dim]
            attention_mask: Attention mask of shape [seq_len, seq_len] or [batch_size, 1, seq_len, seq_len]
            key_padding_mask: Padding mask of shape [batch_size, seq_len]
        """
        batch_size, seq_len, _ = x.shape

        # Project queries, keys, and values at once for efficiency
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch_size, num_heads, seq_len, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: [batch_size, num_heads, seq_len, head_dim]

        # Apply scaling to queries
        q = q * self.scale

        # Use block-sparse attention with causal masking
        attn_output = self._block_causal_attention(q, k, v, attention_mask, key_padding_mask)

        # Combine attention heads
        attn_output = self._merge_heads(attn_output, batch_size)

        # Output projection
        return self.out_proj(attn_output)

    def _block_causal_attention(self, q, k, v, attention_mask, key_padding_mask):
        """
        Compute attention in blocks with causal masking.

        This implementation processes the sequence in blocks to optimize for CPU cache utilization.
        It ensures proper causal masking to prevent information flow from future tokens.
        """
        batch_size, num_heads, seq_len, head_dim = q.shape
        block_size = min(self.block_size, seq_len)

        # Initialize output tensor
        attn_output = torch.zeros_like(q)

        # Process query sequence in blocks for better CPU cache utilization
        for i in range(0, seq_len, block_size):
            i_end = min(i + block_size, seq_len)
            q_block = q[:, :, i:i_end]  # [batch_size, num_heads, block_size, head_dim]

            # Initialize attention tensor for this query block
            attn_scores = torch.zeros(batch_size, num_heads, i_end - i, seq_len,
                                      dtype=q.dtype, device=q.device)

            # Process keys up to current point (causal masking)
            for j in range(0, min(i_end, seq_len), block_size):
                j_end = min(j + block_size, seq_len)
                k_block = k[:, :, j:j_end]  # [batch_size, num_heads, block_size, head_dim]

                # Calculate attention scores for this block
                # [batch_size, num_heads, block_size_i, block_size_j]
                scores = torch.matmul(q_block, k_block.transpose(-2, -1))

                # Store scores in the appropriate location
                attn_scores[:, :, :, j:j_end] = scores

            # Apply causal masking - prevent attending to future tokens
            # Create causal mask (lower triangular)
            causal_mask = torch.ones(i_end, seq_len, dtype=torch.bool, device=q.device)
            causal_mask = torch.tril(causal_mask, diagonal=0)

            # Move the relevant portion of causal mask to current block
            block_causal_mask = causal_mask[i:i_end, :]

            # Apply causal mask
            causal_mask_expanded = block_causal_mask.unsqueeze(0).unsqueeze(0)
            attn_scores = attn_scores.masked_fill(~causal_mask_expanded, float('-inf'))

            # Apply additional attention mask if provided
            if attention_mask is not None:
                if attention_mask.dim() == 2:
                    mask = attention_mask[i:i_end, :].unsqueeze(0).unsqueeze(0)
                else:
                    mask = attention_mask[:, :, i:i_end, :]
                attn_scores = attn_scores.masked_fill(~mask, float('-inf'))

            # Apply padding mask if provided
            if key_padding_mask is not None:
                padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(1)
                attn_scores = attn_scores.masked_fill(~padding_mask, float('-inf'))

            # Apply softmax along key dimension
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.dropout_layer(attn_weights)

            # Apply attention weights to values
            # For each query position in this block, compute weighted sum of values up to seq_len
            block_output = torch.zeros(batch_size, num_heads, i_end - i, head_dim,
                                       dtype=v.dtype, device=v.device)

            for j in range(0, seq_len, block_size):
                j_end = min(j + block_size, seq_len)
                v_block = v[:, :, j:j_end]  # [batch_size, num_heads, block_size, head_dim]

                # Extract corresponding attention weights for this value block
                # [batch_size, num_heads, block_size_i, block_size_j]
                weights_block = attn_weights[:, :, :, j:j_end]

                # Update output with weighted values
                block_output += torch.matmul(weights_block, v_block)

            # Store the output for this query block
            attn_output[:, :, i:i_end] = block_output

        return attn_output


class CPUEfficientFeedForward(nn.Module):
    """
    CPU-optimized feed-forward network.
    """

    def __init__(self, hidden_dim, ff_dim, dropout=0.1, activation='gelu'):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, ff_dim, bias=False)
        self.fc2 = nn.Linear(ff_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Choose CPU-efficient activation function
        if activation == 'gelu':
            # Approximation of GELU for better CPU performance
            self.activation = lambda x: x * torch.sigmoid(1.702 * x)
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'silu' or activation == 'swish':
            self.activation = F.silu
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x):
        return self.fc2(self.dropout(self.activation(self.fc1(x))))


class CPUEfficientBlock(nn.Module):
    """
    Transformer block optimized for CPU execution.
    """

    def __init__(
        self,
        hidden_dim,
        num_heads,
        ff_dim,
        dropout=0.1,
        attention_dropout=0.1,
        ff_dropout=0.1,
        block_size=64,
        activation='gelu',
        use_checkpoint=False
    ):
        super().__init__()

        self.use_checkpoint = use_checkpoint

        # Pre-normalization layers (stabler training)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Causal self-attention
        self.attention = CPUEfficientCausalAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            block_size=block_size
        )

        # Feed-forward network
        self.ff = CPUEfficientFeedForward(
            hidden_dim=hidden_dim,
            ff_dim=ff_dim,
            dropout=ff_dropout,
            activation=activation
        )

        self.dropout = nn.Dropout(dropout)

    def _forward_impl(self, x, attention_mask=None, key_padding_mask=None):
        # Self-attention block with pre-normalization
        residual = x
        x = self.norm1(x)
        x = self.attention(x, attention_mask, key_padding_mask)
        x = self.dropout(x)
        x = residual + x

        # Feed-forward block with pre-normalization
        residual = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropout(x)
        x = residual + x

        return x

    def forward(self, x, attention_mask=None, key_padding_mask=None):
        if self.use_checkpoint and self.training:
            # Use gradient checkpointing to save memory during training
            return checkpoint(
                self._forward_impl,
                x, attention_mask, key_padding_mask
            )
        else:
            return self._forward_impl(x, attention_mask, key_padding_mask)


class CPUCausalGenerativeModel(nn.Module):
    """
    CPU-optimized causal generative model for text generation.
    Uses learned token embeddings only (no positional embeddings).
    """

    def __init__(
        self,
        vocab_size,
        hidden_dim=768,
        num_heads=12,
        num_layers=12,
        ff_dim=3072,
        dropout=0.1,
        attention_dropout=0.1,
        ff_dropout=0.1,
        block_size=64,
        activation='gelu',
        pad_idx=0,
        max_seq_len=1024,
        tie_weights=True,
        use_checkpoint=False
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.pad_idx = pad_idx
        self.max_seq_len = max_seq_len

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_idx)

        # Dropout
        self.embedding_dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            CPUEfficientBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout,
                attention_dropout=attention_dropout,
                ff_dropout=ff_dropout,
                block_size=block_size,
                activation=activation,
                use_checkpoint=use_checkpoint
            )
            for _ in range(num_layers)
        ])

        # Final layer normalization
        self.norm = nn.LayerNorm(hidden_dim)

        # Output projection
        if tie_weights:
            self.output_proj = nn.Linear(hidden_dim, vocab_size, bias=False)
            self.output_proj.weight = self.token_embedding.weight
        else:
            self.output_proj = nn.Linear(hidden_dim, vocab_size)

        # Initialize parameters
        self._init_parameters()

        # Track metrics for CPU efficiency
        self.perf_stats = {
            'forward_time': [],
            'backward_time': [],
            'generate_time': [],
            'tokens_per_sec': []
        }

    def _init_parameters(self):
        """Initialize model parameters for stable training."""
        for name, p in self.named_parameters():
            if p.dim() > 1:
                if 'output_proj.weight' in name and hasattr(self, 'token_embedding') and \
                    getattr(self, 'token_embedding').weight is self.output_proj.weight:
                    # Skip initialization for tied weights
                    continue

                nn.init.xavier_normal_(p, gain=0.02)
            elif 'norm' in name and 'weight' in name:
                # Initialize LayerNorm weights to 1
                nn.init.ones_(p)

    def create_causal_mask(self, seq_len):
        """Create causal attention mask for decoder."""
        return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))

    def create_padding_mask(self, x):
        """Create padding mask for input sequence."""
        return x != self.pad_idx

    def forward(self, x, targets=None):
        """
        Forward pass with optional loss calculation.

        Args:
            x: Input token ids [batch_size, seq_len]
            targets: Target token ids for loss calculation [batch_size, seq_len]

        Returns:
            logits: Predicted token logits [batch_size, seq_len, vocab_size]
            loss: Optional cross entropy loss if targets provided
        """
        start_time = time.time()

        batch_size, seq_len = x.shape
        assert seq_len <= self.max_seq_len, f"Input sequence length {seq_len} exceeds maximum {self.max_seq_len}"

        # Create attention mask (causal)
        attention_mask = self.create_causal_mask(seq_len).to(x.device)

        # Create padding mask if needed
        key_padding_mask = self.create_padding_mask(x) if self.pad_idx is not None else None

        # Token embeddings (no positional embeddings)
        h = self.token_embedding(x)
        h = self.embedding_dropout(h)

        # Apply transformer blocks
        for block in self.blocks:
            h = block(h, attention_mask, key_padding_mask)

        # Apply final normalization
        h = self.norm(h)

        # Predict next tokens
        logits = self.output_proj(h)

        forward_time = time.time() - start_time
        self.perf_stats['forward_time'].append(forward_time)

        # Calculate loss if targets provided
        loss = None
        if targets is not None:
            backward_start = time.time()

            # Reshape for efficient loss calculation
            flat_logits = logits.view(-1, self.vocab_size)
            flat_targets = targets.view(-1)

            # Cross entropy loss
            loss = F.cross_entropy(
                flat_logits,
                flat_targets,
                ignore_index=self.pad_idx if self.pad_idx is not None else -100
            )

            backward_time = time.time() - backward_start
            self.perf_stats['backward_time'].append(backward_time)

        return logits, loss

    def generate(self,
                 input_ids,
                 max_new_tokens=100,
                 temperature=1.0,
                 top_k=None,
                 top_p=None,
                 repetition_penalty=1.0,
                 do_sample=True,
                 use_cache=True):
        """
        Generate text from input prompt.

        Args:
            input_ids: Input token ids [batch_size, seq_len]
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Temperature for sampling
            top_k: Sample from top k most probable tokens
            top_p: Sample from top tokens with cumulative probability >= top_p
            repetition_penalty: Penalty for repeating tokens
            do_sample: If False, use greedy decoding
            use_cache: If True, use KV caching (not implemented in this CPU version)

        Returns:
            all_tokens: Generated token ids [batch_size, seq_len + max_new_tokens]
        """
        start_time = time.time()

        self.eval()  # Set to evaluation mode

        with torch.no_grad():
            batch_size = input_ids.shape[0]
            device = input_ids.device

            # Clone input to avoid modifying original
            all_tokens = input_ids.clone()
            cur_len = all_tokens.shape[1]

            # Check if we're already at max length
            max_len = min(cur_len + max_new_tokens, self.max_seq_len)
            if cur_len >= max_len:
                return all_tokens

            # Generate tokens one by one
            for _ in range(max_new_tokens):
                # Forward pass to get logits for the next token
                logits, _ = self.forward(all_tokens)

                # Only use the logits for the last token
                next_token_logits = logits[:, -1, :].float()

                # Apply temperature
                if temperature > 0:
                    next_token_logits = next_token_logits / temperature

                # Apply repetition penalty: reduce probability of tokens already generated
                if repetition_penalty != 1.0:
                    for i in range(batch_size):
                        for token_id in all_tokens[i].unique():
                            if token_id < next_token_logits.shape[-1]:
                                # If token has been used, apply penalty by dividing/multiplying logit
                                if next_token_logits[i, token_id] > 0:
                                    next_token_logits[i, token_id] /= repetition_penalty
                                else:
                                    next_token_logits[i, token_id] *= repetition_penalty

                # Set -inf to pad tokens
                if self.pad_idx is not None:
                    next_token_logits[:, self.pad_idx] = float('-inf')

                # Get probabilities
                probs = F.softmax(next_token_logits, dim=-1)

                # Apply top-k filtering
                if top_k is not None and top_k > 0:
                    values, _ = torch.topk(probs, top_k)
                    min_values = values[:, -1].unsqueeze(-1)
                    probs = torch.where(probs < min_values,
                                        torch.zeros_like(probs),
                                        probs)
                    # Renormalize probabilities
                    probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)

                # Apply top-p (nucleus) filtering
                if top_p is not None and top_p > 0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p

                    # Shift the indices to the right to keep the first token above threshold
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = 0

                    # Create scatter mask
                    indices_to_remove = torch.zeros_like(probs, dtype=torch.bool).scatter_(
                        dim=-1,
                        index=sorted_indices,
                        src=sorted_indices_to_remove
                    )

                    probs = torch.where(indices_to_remove, torch.zeros_like(probs), probs)
                    # Renormalize probabilities
                    probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)

                # Sample or greedy decode
                if do_sample:
                    # Multinomial sampling from the probability distribution
                    next_tokens = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy selection of the token with maximum probability
                    next_tokens = torch.argmax(probs, dim=-1, keepdim=True)

                # Append generated tokens
                all_tokens = torch.cat([all_tokens, next_tokens], dim=1)

                # Check if we've reached max length
                if all_tokens.shape[1] >= max_len:
                    break

        generate_time = time.time() - start_time
        tokens_per_sec = max_new_tokens / generate_time

        self.perf_stats['generate_time'].append(generate_time)
        self.perf_stats['tokens_per_sec'].append(tokens_per_sec)

        return all_tokens

    def get_performance_stats(self):
        """Get average performance statistics."""
        stats = {}
        for key, values in self.perf_stats.items():
            if values:
                stats[f'avg_{key}'] = sum(values) / len(values)

        return stats

    def clear_performance_stats(self):
        """Reset all performance tracking."""
        for key in self.perf_stats:
            self.perf_stats[key] = []


class CPUOptimizedAdamW(torch.optim.Optimizer):
    """
    CPU-optimized implementation of AdamW.
    Uses fused operations and in-place updates where possible.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Get parameters
                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1

                # Weight decay (L2 penalty)
                if group['weight_decay'] > 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])

                # Decay the first and second moment running averages
                # Using in-place operations for CPU efficiency
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Compute denominator
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                # Update parameters
                step_size = group['lr'] / bias_correction1
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


def prepare_huggingface_dataset(tokenizer, dataset_name="wikitext", dataset_config="wikitext-103-v1",
                                max_seq_len=512, batch_size=8):
    """
    Prepare dataset from Hugging Face for training.

    Args:
        tokenizer: Tokenizer to use (TiktokenWrapper)
        dataset_name: Name of the dataset on Hugging Face
        dataset_config: Configuration/subset of the dataset
        max_seq_len: Maximum sequence length
        batch_size: Batch size

    Returns:
        train_dataloader, eval_dataloader
    """
    from torch.utils.data import Dataset, DataLoader

    class TextDataset(Dataset):
        def __init__(self, texts, tokenizer, max_length):
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.examples = []

            # Process all texts
            for text in texts:
                if not text.strip():  # Skip empty texts
                    continue

                # Tokenize text
                tokens = tokenizer.encode(text)

                # Create examples with sliding window
                for i in range(0, max(1, len(tokens) - max_length), max_length // 2):
                    end = min(i + max_length, len(tokens))
                    if end - i < 4:  # Skip very short sequences
                        continue
                    self.examples.append(tokens[i:end])

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, idx):
            item = torch.tensor(self.examples[idx], dtype=torch.long)
            return item, item  # input_ids, labels are the same for causal LM

    # Load the dataset from Hugging Face
    print(f"Loading dataset {dataset_name}/{dataset_config} from Hugging Face...")
    raw_datasets = load_dataset(dataset_name, dataset_config)

    # Extract text data
    train_texts = [example['text'] for example in raw_datasets['train']]
    if 'validation' in raw_datasets:
        eval_texts = [example['text'] for example in raw_datasets['validation']]
    elif 'test' in raw_datasets:
        eval_texts = [example['text'] for example in raw_datasets['test']]
    else:
        # If no validation/test split exists, use a portion of train data
        split_idx = int(0.95 * len(train_texts))
        eval_texts = train_texts[split_idx:]
        train_texts = train_texts[:split_idx]

    print(f"Loaded {len(train_texts)} training examples and {len(eval_texts)} evaluation examples")

    # Create datasets
    train_dataset = TextDataset(train_texts, tokenizer, max_seq_len)
    eval_dataset = TextDataset(eval_texts, tokenizer, max_seq_len)

    print(f"Created {len(train_dataset)} training sequences and {len(eval_dataset)} evaluation sequences")

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn
    )

    return train_dataloader, eval_dataloader


from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch):
    """Custom collate function to handle variable length sequences."""
    # Separate inputs and targets
    inputs, targets = zip(*batch)

    # Pad sequences to the same length
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=0)

    return inputs_padded, targets_padded

# Training function optimized for CPU
def train_on_cpu(
    model,
    dataloader,
    optimizer,
    num_epochs=3,
    gradient_accumulation_steps=1,
    clip_grad_norm=1.0,
    scheduler=None,
    eval_dataloader=None,
    eval_steps=10,
    log_steps=1,
    checkpoint_path=None,
    checkpoint_steps=1000
):
    """
    Train model efficiently on CPU.

    Args:
        model: The CPUCausalGenerativeModel
        dataloader: Training dataloader
        optimizer: Optimizer (preferably CPUOptimizedAdamW)
        num_epochs: Number of training epochs
        gradient_accumulation_steps: Number of steps to accumulate gradients
        clip_grad_norm: Maximum gradient norm for clipping
        scheduler: Learning rate scheduler (optional)
        eval_dataloader: Evaluation dataloader (optional)
        eval_steps: Steps between evaluations
        log_steps: Steps between logging
        checkpoint_path: Path to save model checkpoints
        checkpoint_steps: Steps between saving checkpoints

    Returns:
        List of training losses
    """
    model.train()

    # Track training progress
    losses = []
    global_step = 0
    total_start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        epoch_loss = 0

        for step, batch in enumerate(dataloader):
            # Unpack batch - adapt this to your dataloader's format
            if isinstance(batch, dict):
                input_ids = batch['input_ids']
                labels = batch.get('labels', input_ids)
            else:
                # Assume batch is a tuple (input_ids, labels)
                input_ids, labels = batch

            # Forward pass
            _, loss = model(input_ids, labels)
            loss_value = loss.item()

            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps
            loss.backward()

            # Track loss
            epoch_loss += loss_value
            losses.append(loss_value)

            # Gradient accumulation
            if (step + 1) % gradient_accumulation_steps == 0 or step == len(dataloader) - 1:
                # Gradient clipping
                if clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

                # Optimizer step
                optimizer.step()
                optimizer.zero_grad()

                if scheduler is not None:
                    scheduler.step()

                global_step += 1

                # Logging
                if global_step % log_steps == 0:
                    perf_stats = model.get_performance_stats()
                    tokens_per_sec = perf_stats.get('avg_tokens_per_sec', 0)

                    # Modified to include more prominent loss information
                    print(f"[LOSS] Epoch {epoch + 1}/{num_epochs}, Step {global_step}, "
                          f"Loss: {loss_value:.4f}, Tokens/sec: {tokens_per_sec:.2f}")
                    model.clear_performance_stats()

                # Evaluation
                if eval_dataloader is not None and global_step % eval_steps == 0:
                    eval_loss, perplexity = evaluate_on_cpu(model, eval_dataloader)
                    print(f"[EVAL] Step {global_step}, Loss: {eval_loss:.4f}, Perplexity: {perplexity:.2f}")
                    # Return to training mode
                    model.train()

                # Save checkpoint
                if checkpoint_path and global_step % checkpoint_steps == 0:
                    save_path = f"{checkpoint_path}_step_{global_step}.pt"
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'global_step': global_step,
                        'epoch': epoch,
                    }, save_path)
                    print(f"Checkpoint saved to {save_path}")

        # End of epoch stats
        epoch_time = time.time() - epoch_start_time
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s, Avg Loss: {avg_epoch_loss:.4f}")

    total_time = time.time() - total_start_time
    print(f"Training completed in {total_time:.2f}s")

    # Final save
    if checkpoint_path:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step,
            'epoch': num_epochs,
        }, f"{checkpoint_path}_final.pt")

    return losses


def evaluate_on_cpu(model, dataloader):
    """
    Evaluate model on CPU.

    Args:
        model: The CPUCausalGenerativeModel
        dataloader: Evaluation dataloader

    Returns:
        Average loss and perplexity
    """
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            # Unpack batch - adapt this to your dataloader's format
            if isinstance(batch, dict):
                input_ids = batch['input_ids']
                labels = batch.get('labels', input_ids)
            else:
                # Assume batch is a tuple (input_ids, labels)
                input_ids, labels = batch

            # Count non-padding tokens
            non_pad_mask = labels != model.pad_idx
            num_tokens = non_pad_mask.sum().item()

            # Forward pass
            _, loss = model(input_ids, labels)
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    # Calculate average loss and perplexity
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = math.exp(avg_loss)

    return avg_loss, perplexity


# Example usage with Wikipedia dataset
def prepare_wiki_dataset(tokenizer, wiki_path, max_seq_len=512, batch_size=8):
    """
    Prepare Wikipedia dataset for training.

    Args:
        tokenizer: Tokenizer to use
        wiki_path: Path to Wikipedia dataset (text files)
        max_seq_len: Maximum sequence length
        batch_size: Batch size

    Returns:
        train_dataloader, eval_dataloader
    """
    import os
    from torch.utils.data import Dataset, DataLoader, random_split

    class WikiTextDataset(Dataset):
        def __init__(self, file_paths, tokenizer, max_length):
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.examples = []

            # Process all files
            for file_path in file_paths:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()

                # Tokenize text
                tokens = tokenizer.encode(text)

                # Create examples with sliding window
                for i in range(0, len(tokens) - max_length, max_length // 2):
                    end = min(i + max_length, len(tokens))
                    self.examples.append(tokens[i:end])

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, idx):
            item = torch.tensor(self.examples[idx], dtype=torch.long)
            return item, item  # input_ids, labels are the same for causal LM

    # Get list of wiki text files
    if os.path.isdir(wiki_path):
        files = [os.path.join(wiki_path, f) for f in os.listdir(wiki_path)
                 if f.endswith('.txt')]
    else:
        files = [wiki_path]

    # Create dataset
    dataset = WikiTextDataset(files, tokenizer, max_seq_len)

    # Split into train and eval
    train_size = int(0.95 * len(dataset))
    eval_size = len(dataset) - train_size

    train_dataset, eval_dataset = random_split(
        dataset, [train_size, eval_size]
    )

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    collate_fn=collate_fn

    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    collate_fn=collate_fn

    )

    return train_dataloader, eval_dataloader


import tiktoken


class TiktokenWrapper:
    """Wrapper for tiktoken to make it compatible with the existing code."""

    def __init__(self, encoding_name="cl100k_base"):
        self.encoding = tiktoken.get_encoding(encoding_name)
        self.vocab_size = self.encoding.n_vocab
        self.pad_token_id = 0  # We need to define special tokens
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3

    def encode(self, text):
        """Convert text to token IDs."""
        ids = [self.bos_token_id]  # Start with <bos>
        ids.extend(self.encoding.encode(text))
        ids.append(self.eos_token_id)  # End with <eos>
        return ids

    def decode(self, ids):
        """Convert token IDs back to text."""
        # Filter out special tokens
        filtered_ids = [id for id in ids if id not in
                        [self.pad_token_id, self.bos_token_id, self.eos_token_id]]

        # Use tiktoken's decode method
        return self.encoding.decode(filtered_ids)

# Function to train model on Wikipedia data
def train_on_wikipedia(wiki_path, output_path, epochs=3):
    """
    Train the CPU-optimized causal model on Wikipedia data.

    Args:
        wiki_path: Path to Wikipedia text files
        output_path: Path to save model checkpoint
        epochs: Number of training epochs
    """
    print("Starting Wikipedia training...")

    # 1. Prepare tokenizer
    print("Building tokenizer...")
    tokenizer = TiktokenWrapper(encoding_name='o200k_base')

    # Build vocabulary from wiki files
    wiki_texts = []
    import os
    if os.path.isdir(wiki_path):
        for file in os.listdir(wiki_path):
            if file.endswith('.txt'):
                with open(os.path.join(wiki_path, file), 'r', encoding='utf-8') as f:
                    wiki_texts.append(f.read())
    else:
        with open(wiki_path, 'r', encoding='utf-8') as f:
            wiki_texts.append(f.read())

    tokenizer.build_vocab(wiki_texts)
    print(f"Vocabulary built with {len(tokenizer.word2idx)} tokens")

    # 2. Prepare datasets
    print("Preparing datasets...")
    train_dataloader, eval_dataloader = prepare_wiki_dataset(
        tokenizer=tokenizer,
        wiki_path=wiki_path,
        max_seq_len=512,
        batch_size=8
    )
    print(f"Created training dataloader with {len(train_dataloader)} batches")
    print(f"Created evaluation dataloader with {len(eval_dataloader)} batches")

    # 3. Initialize model
    print("Initializing model...")
    model = CPUCausalGenerativeModel(
        vocab_size=len(tokenizer.word2idx),
        hidden_dim=768,  # Medium-sized model for CPU training
        num_heads=12,
        num_layers=12,
        ff_dim=3072,
        dropout=0.1,
        attention_dropout=0.1,
        ff_dropout=0.1,
        block_size=64,  # Optimized for CPU cache size
        activation='gelu',
        pad_idx=tokenizer.pad_token_id,
        max_seq_len=512,
        tie_weights=True,  # Reduce memory usage
        use_checkpoint=True  # Save memory during training
    )

    # 4. Set up optimizer with CPU-specific optimizations
    print("Setting up optimizer...")
    optimizer = CPUOptimizedAdamW(
        model.parameters(),
        lr=5e-5,
        weight_decay=0.01
    )

    # 5. Set up learning rate scheduler
    from torch.optim.lr_scheduler import LinearLR
    scheduler = LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.1,
        total_iters=len(train_dataloader) * epochs
    )

    # 6. Train model
    print("Starting training...")
    train_on_cpu(
        model=model,
        dataloader=train_dataloader,
        optimizer=optimizer,
        num_epochs=epochs,
        gradient_accumulation_steps=4,  # Larger batches for CPU efficiency
        clip_grad_norm=1.0,
        scheduler=scheduler,
        eval_dataloader=eval_dataloader,
        eval_steps=500,
        log_steps=100,
        checkpoint_path=output_path,
        checkpoint_steps=1000
    )

    print("Training complete!")

    # 7. Final evaluation
    print("Performing final evaluation...")
    eval_loss, perplexity = evaluate_on_cpu(model, eval_dataloader)
    print(f"Final evaluation - Loss: {eval_loss:.4f}, Perplexity: {perplexity:.2f}")

    # 8. Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'tokenizer': tokenizer,
        'vocab': tokenizer.word2idx,
    }, f"{output_path}_final_complete.pt")

    print(f"Model saved to {output_path}_final_complete.pt")

    return model, tokenizer

def train_model(dataset_name="wikitext", dataset_config="wikitext-103-v1", output_path="model", epochs=3):
    """
    Train the CPU-optimized causal model on a Hugging Face dataset.

    Args:
        dataset_name: Name of the dataset on Hugging Face
        dataset_config: Configuration/subset of the dataset
        output_path: Path to save model checkpoint
        epochs: Number of training epochs
    """
    print(f"Starting training on {dataset_name}/{dataset_config}...")

    # 1. Initialize tokenizer
    print("Initializing tiktoken...")
    tokenizer = TiktokenWrapper(encoding_name="cl100k_base")
    print(f"Tokenizer initialized with vocabulary size of {tokenizer.vocab_size}")

    # 2. Prepare datasets
    print("Preparing datasets...")
    train_dataloader, eval_dataloader = prepare_huggingface_dataset(
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        max_seq_len=512,
        batch_size=8
    )
    print(f"Created training dataloader with {len(train_dataloader)} batches")
    print(f"Created evaluation dataloader with {len(eval_dataloader)} batches")

    # 3. Initialize model
    print("Initializing model...")
    model = CPUCausalGenerativeModel(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=768,  # Medium-sized model for CPU training
        num_heads=12,
        num_layers=12,
        ff_dim=3072,
        dropout=0.1,
        attention_dropout=0.1,
        ff_dropout=0.1,
        block_size=64,  # Optimized for CPU cache size
        activation='gelu',
        pad_idx=tokenizer.pad_token_id,
        max_seq_len=512,
        tie_weights=True,  # Reduce memory usage
        use_checkpoint=True  # Save memory during training
    )

    # 4. Set up optimizer with CPU-specific optimizations
    print("Setting up optimizer...")
    optimizer = CPUOptimizedAdamW(
        model.parameters(),
        lr=5e-5,
        weight_decay=0.01
    )

    # 5. Set up learning rate scheduler
    from torch.optim.lr_scheduler import LinearLR
    scheduler = LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.1,
        total_iters=len(train_dataloader) * epochs
    )

    # 6. Train model
    print("Starting training...")
    train_on_cpu(
        model=model,
        dataloader=train_dataloader,
        optimizer=optimizer,
        num_epochs=epochs,
        gradient_accumulation_steps=4,  # Larger batches for CPU efficiency
        clip_grad_norm=1.0,
        scheduler=scheduler,
        eval_dataloader=eval_dataloader,
        eval_steps=500,
        log_steps=100,
        checkpoint_path=output_path,
        checkpoint_steps=1000
    )

    print("Training complete!")

    # 7. Final evaluation
    print("Performing final evaluation...")
    eval_loss, perplexity = evaluate_on_cpu(model, eval_dataloader)
    print(f"Final evaluation - Loss: {eval_loss:.4f}, Perplexity: {perplexity:.2f}")

    # 8. Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'tokenizer_config': {
            'encoding_name': 'cl100k_base',
            'pad_token_id': tokenizer.pad_token_id,
            'bos_token_id': tokenizer.bos_token_id,
            'eos_token_id': tokenizer.eos_token_id,
        }
    }, f"{output_path}_final_complete.pt")

    print(f"Model saved to {output_path}_final_complete.pt")

    return model, tokenizer


# Example of text generation with the trained model
def generate_text(model, tokenizer, prompt, max_length=100):
    """
    Generate text using the trained model.

    Args:
        model: Trained CPUCausalGenerativeModel
        tokenizer: Tokenizer used during training
        prompt: Text prompt to start generation
        max_length: Maximum number of tokens to generate

    Returns:
        Generated text
    """
    model.eval()

    # Tokenize prompt
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)

    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_length,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True
        )

    # Decode output
    generated_text = tokenizer.decode(output_ids[0].tolist())

    return generated_text


# Example usage
if __name__ == "__main__":
    # Example of training with a Hugging Face dataset
    model, tokenizer = train_model(
        dataset_name="wikitext",
        dataset_config="wikitext-2-v1",  # You can also try "wikitext-2-v1" for a smaller dataset
        output_path="cpu_model",
        epochs=3
    )

    prompt = "The history of artificial intelligence"
    generated_text = generate_text(model, tokenizer, prompt, max_length=200)
    print(f"Generated: {generated_text}")
    exit()

    # Example of model creation and forward pass
    # model = CPUCausalGenerativeModel(
    #     vocab_size=50000,
    #     hidden_dim=768,
    #     num_heads=12,
    #     num_layers=12,
    #     ff_dim=3072
    # )
    #
    # # Print model size
    # param_count = sum(p.numel() for p in model.parameters())
    # print(f"Model created with {param_count:,} parameters")
    #
    # # Example forward pass
    # batch_size = 4
    # seq_len = 128
    # input_ids = torch.randint(0, 50000, (batch_size, seq_len))
    #
    # start_time = time.time()
    # logits, _ = model(input_ids)
    # forward_time = time.time() - start_time
    #
    # print(f"Forward pass completed in {forward_time:.4f} seconds")
    # print(f"Output shape: {logits.shape}")
    #
    # # Example of text generation (with random inputs)
    # start_time = time.time()
    # generated = model.generate(
    #     input_ids[:, :10],  # Use first 10 tokens as prompt
    #     max_new_tokens=20
    # )
    # generate_time = time.time() - start_time
    #
    # print(f"Generated {generated.shape[1] - 10} tokens in {generate_time:.4f} seconds")

    # Example of training with Wikipedia
    # Uncomment to run actual training:
    model, tokenizer = train_on_wikipedia(
        wiki_path="/path/to/wiki/data",
        output_path="cpu_model",
        epochs=3
    )

    prompt = "The history of artificial intelligence"
    generated_text = generate_text(model, tokenizer, prompt, max_length=200)
    print(f"Generated: {generated_text}")