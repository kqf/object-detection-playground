import pytest
import torch

from detection.models.darknet import Darknet
from detection.models.legacy import build_model


@pytest.fixture
def batch(size):
    return torch.rand(4, 3, size, size)


@pytest.mark.parametrize("size", [
    32,
    32 * 2,
    32 * 10,
])
def test_module(batch, size):
    model = Darknet()
    l1, l2, l3 = model(batch)

    out_size = size // 32
    assert l1.shape == (4, 1024, out_size, out_size)
    assert l2.shape == (4, 512, out_size * 2, out_size * 2)
    assert l3.shape == (4, 256, out_size * 4, out_size * 4)


@pytest.mark.parametrize("size", [
    32,
    32 * 2,
    32 * 10,
])
def test_legacy(batch, size):
    model = torch.nn.Sequential(*build_model(3, 40))
    l1 = model(batch)
    out_size = size // 32
    assert l1.shape == (4, 1024, out_size, out_size)
