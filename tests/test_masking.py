import torch

from src.training.masking import MASK_KINDS, make_mask


def test_every_mask_covers_exactly_half():
    valid = torch.ones(1, 6, 30, dtype=torch.bool)
    coordinates = torch.randn(1, 6, 3)
    for index, kind in enumerate(MASK_KINDS):
        mask = make_mask(kind, valid, coordinates, seed=100 + index)
        assert int(mask.sum()) == int(valid.sum()) // 2
        assert not (mask & ~valid).any()

