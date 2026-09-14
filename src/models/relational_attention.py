from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class NeighborTemplate:
    indices_t: torch.Tensor
    valid_t: torch.Tensor
    relation_t: torch.Tensor
    indices_s: torch.Tensor
    valid_s: torch.Tensor
    relation_s: torch.Tensor
    indices_st: torch.Tensor
    valid_st: torch.Tensor
    relation_st: torch.Tensor

    def to(self, device: torch.device | str) -> "NeighborTemplate":
        return NeighborTemplate(**{
            name: getattr(self, name).to(device=device)
            for name in self.__dataclass_fields__
        })


class SparseRelationAttention(nn.Module):
    """Multi-head attention over a precomputed typed sparse neighborhood."""

    def __init__(self, input_dim: int, output_dim: int, heads: int, relation_dim: int = 2):
        super().__init__()
        if output_dim % heads:
            raise ValueError("output_dim must be divisible by heads")
        self.heads = heads
        self.head_dim = output_dim // heads
        self.output_dim = output_dim
        self.q = nn.Linear(input_dim, output_dim, bias=False)
        self.k = nn.Linear(input_dim, output_dim, bias=False)
        self.v = nn.Linear(input_dim, output_dim, bias=False)
        self.bias = nn.Sequential(nn.Linear(relation_dim, 32), nn.GELU(), nn.Linear(32, heads))
        self.out = nn.Linear(output_dim, output_dim)

    def forward(
        self,
        x: torch.Tensor,
        indices: torch.Tensor,
        valid: torch.Tensor,
        relation: torch.Tensor,
    ) -> torch.Tensor:
        batch, length, _ = x.shape
        neighbors_per_token = indices.shape[-1]
        batch_index = torch.arange(batch, device=x.device).view(batch, 1, 1)
        neighbors = x[batch_index, indices.clamp(min=0)]
        q = self.q(x).view(batch, length, self.heads, self.head_dim)
        k = self.k(neighbors).view(batch, length, neighbors_per_token, self.heads, self.head_dim)
        v = self.v(neighbors).view(batch, length, neighbors_per_token, self.heads, self.head_dim)
        logits = torch.einsum("blhd,blkhd->blkh", q, k) / math.sqrt(self.head_dim)
        logits = logits + self.bias(relation.float()).to(logits.dtype)
        logits = logits.masked_fill(~valid.unsqueeze(-1), -1e4)
        weights = torch.softmax(logits, dim=2) * valid.unsqueeze(-1).to(logits.dtype)
        weights = weights / weights.sum(dim=2, keepdim=True).clamp_min(1e-6)
        message = torch.einsum("blkh,blkhd->blhd", weights, v)
        return self.out(message.reshape(batch, length, self.output_dim))

