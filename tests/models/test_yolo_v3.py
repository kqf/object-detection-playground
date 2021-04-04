import pytest
import torch

from detection.models.v3 import YOLO
from detection.models.legacy import YOLO as LOLO


@pytest.fixture
def batch(size):
    return torch.rand(4, 3, size, size)


@pytest.mark.skip("Fix the structure of this model")
def test_module(batch):
    model = YOLO()
    s1, s2, s3 = model(batch)

    assert s1.shape == (4, 3, 8, 8, 85)
    assert s2.shape == (4, 3, 16, 16, 85)
    assert s3.shape == (4, 3, 32, 32, 85)


@pytest.mark.parametrize("size", [
    32,
    32 * 2,
    32 * 10,
    460,
])
def test_legacy_module(batch, size):
    model = LOLO()
    s1, s2, s3 = model(batch)

    out_size = size // 32

    assert s1.shape == (4, 85, out_size, out_size, 3)
    assert s2.shape == (4, 85, out_size * 2, out_size * 2, 3)
    assert s3.shape == (4, 85, out_size * 4, out_size * 4, 3)
