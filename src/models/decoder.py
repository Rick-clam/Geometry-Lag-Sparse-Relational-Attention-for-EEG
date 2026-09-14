from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class WaveformDecoder(nn.Module):
    def __init__(self, dim: int = 1024, samples: int = 200):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, samples)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(self.norm(x))))

