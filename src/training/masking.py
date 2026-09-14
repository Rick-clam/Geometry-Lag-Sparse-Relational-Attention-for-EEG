from __future__ import annotations

import math

import torch


MASK_KINDS = ("random_token", "temporal_block", "whole_channel", "spatiotemporal_block")
MASK_PROBABILITIES = (0.4, 0.2, 0.2, 0.2)


def _exact_half(valid: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    result = torch.zeros_like(valid)
    for b in range(valid.shape[0]):
        indices = torch.nonzero(valid[b].flatten(), as_tuple=False).flatten()
        count = len(indices) // 2
        chosen = indices[torch.randperm(len(indices), generator=generator)[:count]]
        result[b].view(-1)[chosen] = True
    return result


def make_mask(kind: str, valid: torch.Tensor, coordinates: torch.Tensor, seed: int) -> torch.Tensor:
    """Mask exactly half of valid tokens using one of four paper strategies."""
    generator = torch.Generator().manual_seed(seed)
    batch, _, times = valid.shape
    if kind == "random_token":
        return _exact_half(valid, generator)
    result = torch.zeros_like(valid)
    for b in range(batch):
        valid_channels = torch.nonzero(valid[b].any(dim=1), as_tuple=False).flatten()
        if kind == "temporal_block":
            width = max(1, times // 2)
            for channel in valid_channels.tolist():
                start = int(torch.randint(0, times - width + 1, (1,), generator=generator))
                result[b, channel, start:start + width] = True
        elif kind == "whole_channel":
            count = len(valid_channels) // 2
            order = torch.randperm(len(valid_channels), generator=generator)[:count]
            result[b, valid_channels[order], :] = True
        elif kind == "spatiotemporal_block":
            coord = coordinates[b, valid_channels]
            usable = valid_channels[torch.isfinite(coord).all(dim=1)]
            spatial_count = max(1, math.ceil(len(valid_channels) / math.sqrt(2.0)))
            if len(usable):
                anchor = usable[int(torch.randint(0, len(usable), (1,), generator=generator))]
                distances = torch.linalg.vector_norm(coordinates[b, usable] - coordinates[b, anchor], dim=1)
                selected = usable[torch.argsort(distances)[:spatial_count]]
            else:
                selected = valid_channels[:spatial_count]
            temporal_count = math.ceil(times / math.sqrt(2.0))
            start = int(torch.randint(0, times - temporal_count + 1, (1,), generator=generator))
            result[b, selected, start:start + temporal_count] = True
        else:
            raise ValueError(f"unknown mask kind: {kind}")
    for b in range(batch):
        target = int(valid[b].sum()) // 2
        current = torch.nonzero((result[b] & valid[b]).flatten(), as_tuple=False).flatten()
        if len(current) > target:
            keep = current[torch.randperm(len(current), generator=generator)[:target]]
            result[b].zero_()
            result[b].view(-1)[keep] = True
        elif len(current) < target:
            candidates = torch.nonzero((valid[b] & ~result[b]).flatten(), as_tuple=False).flatten()
            add = candidates[torch.randperm(len(candidates), generator=generator)[:target - len(current)]]
            result[b].view(-1)[add] = True
    return result & valid

