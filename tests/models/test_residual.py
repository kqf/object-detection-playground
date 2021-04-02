import pytest
import torch
from detection.models.legacy import SkipConcat


@pytest.fixture
def batch():
    return torch.rand(32, 3, 224, 224)


def test_skips(batch):
    layer = SkipConcat()
    x = layer(batch)
    assert torch.allclose(layer.cache(), x)
