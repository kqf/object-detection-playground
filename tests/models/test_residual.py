import pytest
import torch
from detection.models.darknet import SkipConcat


@pytest.fixture
def batch():
    return torch.rand(32, 3, 224, 224)


def test_skips(batch):
    layer = SkipConcat()
    x = layer(batch)
    assert layer.cache() == x
