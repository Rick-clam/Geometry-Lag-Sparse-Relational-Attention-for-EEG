from __future__ import annotations

import torch

from src.models.geometry_lag import build_neighbor_template, channel_coordinates
from src.models.model import GeometryLagEEG, parameter_report


def main() -> None:
    channels = ["F3", "F4", "C3", "C4", "O1", "O2"]
    patches = torch.randn(1, len(channels), 30, 200)
    valid = torch.ones(1, len(channels), 30, dtype=torch.bool)
    masked = torch.zeros_like(valid)
    coordinates = torch.from_numpy(channel_coordinates(channels))[None]
    template = build_neighbor_template(valid, coordinates)
    model = GeometryLagEEG(activation_checkpointing=False).eval()
    with torch.no_grad():
        output = model(patches, masked, valid, template)
    print(parameter_report(model))
    print("input:", tuple(patches.shape), "output:", tuple(output.shape))


if __name__ == "__main__":
    main()

