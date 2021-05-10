import pytest
import torch

from detection.models.v4 import DownSample1


@pytest.fixture
def batch(size):
    return torch.rand(4, 3, size, size)


@pytest.mark.parametrize("size", [
    32,
    32 * 2,
    32 * 10,
])
def test_backbone(batch, size):
    down1 = DownSample1()
    d1 = down1(batch)
    assert d1.shape == (4, 64, size / 2, size / 2)
