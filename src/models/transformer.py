"""Transformer encoder for pose sequences. Multi-head self-attention is implemented FROM SCRATCH
(no nn.MultiheadAttention) - this is the 'component written by ourselves' of requirement #5."""
import math, torch, torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d, heads, dropout=0.0):
        super().__init__()
        assert d % heads == 0
        self.h, self.dk = heads, d // heads
        self.qkv, self.out, self.drop = nn.Linear(d, 3 * d), nn.Linear(d, d), nn.Dropout(dropout)

    def forward(self, x):                                       # (B,T,D)
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)   # (B,h,T,dk)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)    # scaled dot-product
        att = self.drop(att.softmax(-1))
        return self.out((att @ v).transpose(1, 2).reshape(B, T, D))


class Block(nn.Module):
    def __init__(self, d, heads, mlp_ratio, dropout, attn=True):
        super().__init__()
        self.use_attn = attn
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = MultiHeadSelfAttention(d, heads, dropout) if attn else None
        self.mlp = nn.Sequential(nn.Linear(d, d * mlp_ratio), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(d * mlp_ratio, d), nn.Dropout(dropout))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        if self.use_attn:                                       # pre-LN residual blocks
            x = x + self.drop(self.attn(self.ln1(x)))
        return x + self.mlp(self.ln2(x))


class SignTransformer(nn.Module):
    def __init__(self, J, C, num_classes, T=64, d_model=128, heads=4, layers=3, mlp_ratio=2,
                 dropout=0.2, attn=True):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(J * C, d_model), nn.LayerNorm(d_model))
        self.pos = nn.Parameter(torch.zeros(1, T, d_model)); nn.init.trunc_normal_(self.pos, std=0.02)
        self.mask_token = nn.Parameter(torch.zeros(d_model))     # used only by SSL pre-training
        self.blocks = nn.ModuleList([Block(d_model, heads, mlp_ratio, dropout, attn) for _ in range(layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, num_classes))

    def encode(self, x, frame_mask=None):                       # x (B,T,J,C); frame_mask (B,T) bool
        h = self.embed(x.flatten(2))
        if frame_mask is not None:
            h = torch.where(frame_mask[..., None], self.mask_token.expand_as(h), h)
        h = h + self.pos
        for b in self.blocks:
            h = b(h)
        return self.norm(h)

    def forward(self, x):
        return self.head(self.encode(x).mean(1))                # mean pooling over time
