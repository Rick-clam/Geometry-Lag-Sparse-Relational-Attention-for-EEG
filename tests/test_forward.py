import torch

from src.models.encoder import RelationBlock
from src.models.geometry_lag import build_neighbor_template, channel_coordinates
from src.models.tokenizer import MultiScalePatchTokenizer


def test_tokenizer_and_relation_block_forward():
    channels, times = 4, 10
    patches = torch.randn(1, channels, times, 200)
    masked = torch.zeros(1, channels, times, dtype=torch.bool)
    valid_grid = torch.ones_like(masked)
    coordinates = torch.from_numpy(channel_coordinates(["F3", "F4", "C3", "C4"]))[None]
    template = build_neighbor_template(valid_grid, coordinates)
    tokens = MultiScalePatchTokenizer()(patches, masked).reshape(1, channels * times, 512)
    block = RelationBlock(512, 768, (256, 256, 256), (4, 4, 4), drop_path=0.0)
    output = block(tokens, template, valid_grid.reshape(1, channels * times))
    assert output.shape == (1, channels * times, 768)
    assert torch.isfinite(output).all()

