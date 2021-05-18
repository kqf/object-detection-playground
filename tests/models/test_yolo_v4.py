import pytest
import torch

from detection.models.v4 import DownSample1, DownSample2, DownSample3
from detection.models.v4 import DownSample4, DownSample5


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

    down3 = DownSample3()
    d3 = down3(d2)
    assert d3.shape == (4, 256, size / 8, size / 8)

    down4 = DownSample4()
    d4 = down4(d3)
    assert d4.shape == (4, 512, size / 16, size / 16)

    down5 = DownSample5()
    d5 = down5(d4)
    assert d5.shape == (4, 1024, size / 32, size / 32)
