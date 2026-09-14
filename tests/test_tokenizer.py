import torch

from src.models.tokenizer import MultiScalePatchTokenizer


def test_tokenizer_shape():
    tokenizer = MultiScalePatchTokenizer()
    patches = torch.randn(2, 3, 4, 200)
    masked = torch.zeros(2, 3, 4, dtype=torch.bool)
    output = tokenizer(patches, masked)
    assert output.shape == (2, 3, 4, 512)
    assert torch.isfinite(output).all()

