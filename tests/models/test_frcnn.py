import pytest
import torch

from detection.models.frcnn import FasterRCNN


@pytest.fixture
def batch(size):
    return torch.rand(4, 3, size, size)


@pytest.mark.parametrize("size", [
    32,
    32 * 2,
    32 * 10,
])
def test_module(batch, size):
    model = FasterRCNN()
    s1, s2, s3 = model(batch)

    out_size = size // 32

    assert s1.shape == (4, 85, out_size, out_size, 3)
    assert s2.shape == (4, 85, out_size * 2, out_size * 2, 3)
    assert s3.shape == (4, 85, out_size * 4, out_size * 4, 3)
