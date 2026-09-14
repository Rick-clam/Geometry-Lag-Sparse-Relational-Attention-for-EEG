import torch

from src.models.geometry_lag import build_neighbor_template, channel_coordinates


def test_geometry_lag_shapes():
    coordinates = torch.from_numpy(channel_coordinates(["F3", "F4", "C3", "C4"]))[None]
    valid = torch.ones(1, 4, 30, dtype=torch.bool)
    template = build_neighbor_template(valid, coordinates)
    assert template.indices_t.shape == (1, 120, 8)
    assert template.indices_s.shape == (1, 120, 8)
    assert template.indices_st.shape == (1, 120, 16)
    assert template.valid_t.any()
    assert template.valid_s.any()
    assert template.valid_st.any()

