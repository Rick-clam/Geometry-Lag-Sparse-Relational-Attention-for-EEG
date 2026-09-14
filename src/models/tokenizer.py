from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ConvPatchBranch(nn.Module):
    def __init__(self, kernel_size: int, output_dim: int = 96, channels: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(channels, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.net(x).squeeze(-1))


class MultiScalePatchTokenizer(nn.Module):
    """Map one-second EEG patches to 512-D temporal-spectral tokens."""

    def __init__(self):
        super().__init__()
        self.branches = nn.ModuleList([ConvPatchBranch(k) for k in (3, 7, 15, 31)])
        self.fft_proj = nn.Linear(101, 128)
        self.norm = nn.LayerNorm(512)
        self.mask_embedding = nn.Parameter(torch.zeros(512))
        nn.init.normal_(self.mask_embedding, std=0.02)

    def forward(self, patches: torch.Tensor, masked: torch.Tensor) -> torch.Tensor:
        shape = patches.shape[:-1]
        flat = patches.reshape(-1, 1, patches.shape[-1])
        temporal = [branch(flat) for branch in self.branches]
        spectrum = torch.fft.rfft(flat.squeeze(1).float(), dim=-1, norm="ortho")
        log_power = torch.log1p(spectrum.abs().square())
        if log_power.shape[-1] != 101:
            log_power = F.adaptive_avg_pool1d(log_power.unsqueeze(1), 101).squeeze(1)
        spectral = self.fft_proj(log_power.to(dtype=flat.dtype))
        tokens = self.norm(torch.cat([*temporal, spectral], dim=-1)).reshape(*shape, 512)
        mask_token = self.mask_embedding.to(tokens.dtype).view(1, 1, 1, -1)
        return torch.where(masked.unsqueeze(-1), mask_token, tokens)

