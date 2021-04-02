import pytest
import torch
from detection.models.legacy import SkipConcat


@pytest.fixture
def batch():
    return torch.rand(32, 3, 224, 224)


def test_skips(batch):
    layer = SkipConcat()
    x = layer(batch)

    # Just identity
    assert torch.allclose(batch, x)

    # Second pass differs from the previous one
    assert torch.allclose(layer(x), torch.cat([x, x], axis=1))
