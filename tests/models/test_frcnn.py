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
    output = model(batch)
    assert output.shape == (4, 3, size, size)
