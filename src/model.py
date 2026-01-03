import torch
import torch.nn as nn

class SelfAttentionHead(nn.Module):
    def __init__(self, C=128, head_size=32):
        super().__init__()
        self.key = nn.Linear(C, head_size, bias=False)
        self.query = nn.Linear(C, head_size, bias=False)
        self.value = nn.Linear(C, head_size, bias=False)

    def forward(self, x):
        B, T, C_in = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        wei = q @ k.transpose(-2, -1) / (k.shape[-1] ** 0.5)
        mask = torch.tril(torch.ones(T, T, device=x.device))
        wei = wei.masked_fill(mask == 0, float('-inf'))
        prob = torch.softmax(wei, dim=-1)
        out = prob @ v
        return out

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, C=128, num_heads=4, head_size=32):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttentionHead(C, head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(C, C)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)

class AttentionBlock(nn.Module):
    def __init__(self, C=128, num_heads=4):
        super().__init__()
        self.attn = MultiHeadSelfAttention(C, num_heads)
        self.ln = nn.LayerNorm(C)

    def forward(self, x):
        x = x + self.attn(x)
        x = self.ln(x)
        return x

class FeedForwardBlock(nn.Module):
    def __init__(self, C=128, hidden_mult=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(C, hidden_mult*C),
            nn.ReLU(),
            nn.Linear(hidden_mult*C, C)
        )
        self.ln = nn.LayerNorm(C)

    def forward(self, x):
        x = x + self.net(x)
        x = self.ln(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, C=128, num_heads=4, ff_hidden_mult=2):
        super().__init__()
        self.attn_block = AttentionBlock(C, num_heads)
        self.ffn_block = FeedForwardBlock(C, ff_hidden_mult)

    def forward(self, x):
        x = self.attn_block(x)
        x = self.ffn_block(x)
        return x

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, C=128, num_heads=4, n_layers=4, block_size=128, ff_hidden_mult=4):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, C)
        self.position_embedding = nn.Embedding(block_size, C)
        self.blocks = nn.Sequential(
            *[DecoderBlock(C, num_heads, ff_hidden_mult) for _ in range(n_layers)]
        )
        self.ln_final = nn.LayerNorm(C)
        self.head = nn.Linear(C, vocab_size)
        self.block_size = block_size

    def forward(self, idx):
        B, T = idx.shape
        token_emb = self.token_embedding(idx)
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        pos_emb = self.position_embedding(pos)
        x = token_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_final(x)
        logits = self.head(x)
        return logits
