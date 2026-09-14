from __future__ import annotations

import torch
from torch import nn

from .decoder import WaveformDecoder
from .encoder import GeometryLagEncoder
from .relational_attention import NeighborTemplate
from .tokenizer import MultiScalePatchTokenizer


class GeometryLagEEG(nn.Module):
    """Complete 19-block model for masked EEG reconstruction."""

    def __init__(self, activation_checkpointing: bool = True):
        super().__init__()
        self.tokenizer = MultiScalePatchTokenizer()
        self.encoder = GeometryLagEncoder(activation_checkpointing=activation_checkpointing)
        self.decoder = WaveformDecoder(1024, 200)

    def encode(
        self,
        patches: torch.Tensor,
        masked: torch.Tensor,
        token_valid_grid: torch.Tensor,
        template: NeighborTemplate,
    ) -> torch.Tensor:
        batch, channels, times, _ = patches.shape
        valid = token_valid_grid.reshape(batch, channels * times)
        tokens = self.tokenizer(patches, masked).reshape(batch, channels * times, 512)
        encoded = self.encoder(tokens, template, valid)
        return encoded.reshape(batch, channels, times, 1024)

    def forward(self, patches, masked, token_valid_grid, template):
        encoded = self.encode(patches, masked, token_valid_grid, template)
        return self.decoder(encoded)


def parameter_report(model: nn.Module) -> dict[str, int]:
    return {
        "tokenizer": sum(p.numel() for p in model.tokenizer.parameters()),
        "encoder": sum(p.numel() for p in model.encoder.parameters()),
        "decoder": sum(p.numel() for p in model.decoder.parameters()),
        "total": sum(p.numel() for p in model.parameters()),
    }

