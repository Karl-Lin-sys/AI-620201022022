"""
ProteinAI - Diffusion Model for Protein Sequence Generation
============================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional
import math

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
AA_TO_IDX = {aa: idx for idx, aa in enumerate(AMINO_ACIDS)}
IDX_TO_AA = {idx: aa for idx, aa in enumerate(AMINO_ACIDS)}
NUM_AA = len(AMINO_ACIDS)


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class DiffusionBlock(nn.Module):
    def __init__(self, ch: int, time_dim: int):
        super().__init__()
        self.conv1 = nn.Conv1d(ch, ch, 3, padding=1)
        self.conv2 = nn.Conv1d(ch, ch, 3, padding=1)
        self.norm = nn.GroupNorm(8, ch)
        self.time_mlp = nn.Linear(time_dim, ch)
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm(self.conv1(x)))
        h = h + self.time_mlp(t_emb)[:, :, None]
        return F.silu(self.conv2(h)) + x


class ProteinDiffusionNet(nn.Module):
    def __init__(self, hidden: int = 128, time_dim: int = 64):
        super().__init__()
        self.time_embed = nn.Sequential(
            TimeEmbedding(time_dim // 2),
            nn.Linear(time_dim // 2, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        self.proj_in = nn.Conv1d(NUM_AA, hidden, 1)
        self.blocks = nn.ModuleList([DiffusionBlock(hidden, time_dim) for _ in range(4)])
        self.proj_out = nn.Conv1d(hidden, NUM_AA, 1)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)
        h = self.proj_in(x)
        for block in self.blocks:
            h = block(h, t_emb)
        return self.proj_out(h)


class ProteinDiffusion:
    def __init__(self, steps: int = 100, device: str = 'cpu'):
        self.device = torch.device(device)
        self.steps = steps
        self.betas = torch.linspace(1e-4, 0.02, steps).to(self.device)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        self.model = ProteinDiffusionNet().to(self.device)
    
    def seq_to_onehot(self, seqs: List[str], max_len: int) -> torch.Tensor:
        batch = torch.zeros(len(seqs), NUM_AA, max_len, device=self.device)
        for i, s in enumerate(seqs):
            for j, aa in enumerate(s[:max_len]):
                if aa in AA_TO_IDX:
                    batch[i, AA_TO_IDX[aa], j] = 1.0
        return batch
    
    def onehot_to_seq(self, x: torch.Tensor) -> List[str]:
        idx = x.argmax(dim=1)
        return [''.join([IDX_TO_AA[i.item()] for i in row]) for row in idx]
    
    @torch.no_grad()
    def sample(self, n: int = 1, length: int = 30) -> List[str]:
        self.model.eval()
        x = torch.randn(n, NUM_AA, length, device=self.device)
        for t in range(self.steps - 1, -1, -1):
            t_batch = torch.full((n,), t, device=self.device, dtype=torch.long)
            noise_pred = self.model(x, t_batch)
            alpha = self.alphas[t]
            alpha_bar = self.alpha_bar[t]
            x = (x - (1 - alpha) / (1 - alpha_bar).sqrt() * noise_pred) / alpha.sqrt()
            if t > 0:
                x += self.betas[t].sqrt() * torch.randn_like(x)
        return self.onehot_to_seq(x)


def demo():
    print("=" * 50)
    print("ProteinAI - Diffusion Sequence Generator")
    print("=" * 50)
    gen = ProteinDiffusion()
    seqs = gen.sample(3, 20)
    print("\nGenerated Sequences:")
    for i, s in enumerate(seqs, 1):
        print(f"  {i}: {s}")
    print("=" * 50)
    return gen

if __name__ == "__main__":
    demo()
