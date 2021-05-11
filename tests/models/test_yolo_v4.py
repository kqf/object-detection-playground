import pytest
import torch

from detection.models.v4 import DownSample1, DownSample2


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

    down2 = DownSample2()
    d2 = down2(d1)
    assert d2.shape == (4, 128, size / 4, size / 4)
