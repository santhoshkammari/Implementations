import torch
import torch.nn as nn
import torch.nn.functional as F
import math



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

        out = torch.matmul(attn_weights, v) # ( batch_size, num_heads,seq_len,head_dim)

        out = out.transpose(1,2).contiguous().view(batch_size,seq_len,self.d_model)
        out = self.out_proj(out) #stackign to learnable parameters

        return out

class FeedForward(nn.Module):
    def __init__(self,d_model,ff_dim,dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.ff_dim = ff_dim
        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(d_model,ff_dim)
        self.fc2 = nn.Linear(ff_dim,d_model)
        self.activation = nn.GELU()

    def forward(self,x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self,d_model,num_heads,ff_dim,dropout = 0.1):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.attention = MultiHeadAttention(d_model,num_heads,dropout)
        self.feedforward = FeedForward(d_model, ff_dim, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
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
        print(f'{positions.shape=}')
        print(f'{token_emb.shape=}')
        print(f'{pos_emb.shape=}')
        print(f'{x.shape=}')
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


def train_gpt(model, data_loader, num_epochs, lr=3e-4, device='cuda'):
    # Move model to device
    model = model.to(device)

    # Set up optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in data_loader:
            # Get input tokens
            input_ids = batch.to(device)

            # Forward pass
            logits = model(input_ids[:, :-1])  # all but last token

            # Calculate loss (predict next token)
            targets = input_ids[:, 1:]  # all but first token
            loss = F.cross_entropy(
                logits.reshape(-1, model.vocab_size),
                targets.reshape(-1)
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

    return model

if __name__ == '__main__':
    model = GPT(1000, 10, 256, 4, 4)
    x = torch.randint(0,100,(2,3))
    # res = generate(model,x,5)
    # res = model(x)

