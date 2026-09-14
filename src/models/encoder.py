from __future__ import annotations

from functools import partial

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .relational_attention import NeighborTemplate, SparseRelationAttention


class DropPath(nn.Module):
    def __init__(self, probability: float = 0.0):
        super().__init__()
        self.probability = probability

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.probability == 0.0:
            return x
        keep = 1.0 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x * random.floor() / keep


class RelationBlock(nn.Module):
    """Parallel T/S/ST messages fused into one shared residual state."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        branch_dims: tuple[int, int, int],
        branch_heads: tuple[int, int, int],
        drop_path: float,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.attn_t = SparseRelationAttention(input_dim, branch_dims[0], branch_heads[0])
        self.attn_s = SparseRelationAttention(input_dim, branch_dims[1], branch_heads[1])
        self.attn_st = SparseRelationAttention(input_dim, branch_dims[2], branch_heads[2])
        total = sum(branch_dims)
        self.expanding = input_dim != output_dim
        self.fuse = nn.Linear(input_dim + total if self.expanding else total, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path)
        self.ffn_norm = nn.LayerNorm(output_dim)
        self.ffn = nn.Sequential(
            nn.Linear(output_dim, output_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 4, output_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, template: NeighborTemplate, valid: torch.Tensor) -> torch.Tensor:
        z = self.norm(x)
        messages = torch.cat((
            self.attn_t(z, template.indices_t, template.valid_t, template.relation_t),
            self.attn_s(z, template.indices_s, template.valid_s, template.relation_s),
            self.attn_st(z, template.indices_st, template.valid_st, template.relation_st),
        ), dim=-1)
        if self.expanding:
            x = self.fuse(torch.cat((x, messages), dim=-1))
        else:
            x = x + self.drop_path(self.dropout(self.fuse(messages)))
        x = x + self.drop_path(self.ffn(self.ffn_norm(x)))
        return x * valid.unsqueeze(-1).to(x.dtype)


class GeometryLagEncoder(nn.Module):
    def __init__(self, activation_checkpointing: bool = True):
        super().__init__()
        rates = [0.1 * index / 18 for index in range(19)]
        self.stage1 = nn.ModuleList([
            RelationBlock(512 if i == 0 else 768, 768, (256, 256, 256), (4, 4, 4), rates[i])
            for i in range(4)
        ])
        self.stage2 = nn.ModuleList([
            RelationBlock(768 if i == 0 else 1024, 1024, (256, 256, 512), (4, 4, 8), rates[i + 4])
            for i in range(15)
        ])
        self.activation_checkpointing = activation_checkpointing

    def _run(self, block, x, template, valid):
        if self.activation_checkpointing and self.training:
            fn = partial(block, template=template, valid=valid)
            return checkpoint(fn, x, use_reentrant=False)
        return block(x, template, valid)

    def forward(self, tokens: torch.Tensor, template: NeighborTemplate, valid: torch.Tensor) -> torch.Tensor:
        x = tokens
        for block in self.stage1:
            x = self._run(block, x, template, valid)
        for block in self.stage2:
            x = self._run(block, x, template, valid)
        return x

