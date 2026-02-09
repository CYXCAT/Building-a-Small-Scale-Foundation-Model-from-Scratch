import math
import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, embed_dim):
        super().__init__()
        self.embed = nn.Embedding(max_len, embed_dim)

    def forward(self, x):
        B, L, _ = x.shape
        positions = torch.arange(L, device=x.device)
        return x + self.embed(positions).unsqueeze(0)


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        B, L, C = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, L, C)
        return self.proj(out)


class FFN(nn.Module):
    def __init__(self, embed_dim, hidden_mult=4, act="gelu"):
        super().__init__()
        hidden = embed_dim * hidden_mult
        self.fc1 = nn.Linear(embed_dim, hidden)
        self.fc2 = nn.Linear(hidden, embed_dim)
        self.act = nn.GELU() if act == "gelu" else nn.ReLU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, ffn_mult=4, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, n_heads, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = FFN(embed_dim, ffn_mult)

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ffn(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, seq_len, embed_dim=128, n_heads=4, n_layers=2, ffn_mult=4, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=None)
        self.pos = PositionalEmbedding(seq_len, embed_dim)
        self.layers = nn.ModuleList(
            [TransformerBlock(embed_dim, n_heads, ffn_mult, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, input_ids, mask=None):
        x = self.embed(input_ids)
        x = self.pos(x)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.ln_f(x)
        return self.head(x)
