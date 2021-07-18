import pytest
import torch

from detection.models.v3 import YOLO


@pytest.fixture
def batch(size):
    return torch.rand(4, 3, size, size)


@pytest.mark.parametrize("size", [
    32,
    32 * 2,
    32 * 10,
])
def test_module(batch, size):
    model = YOLO()
    s1, s2, s3 = model(batch)

    out_size = size // 32

    # [batch_size, scales, n_cells, n_cells, labels]
    assert s1.shape == (4, 3, out_size * 1, out_size * 1, 85)
    assert s2.shape == (4, 3, out_size * 2, out_size * 2, 85)
    assert s3.shape == (4, 3, out_size * 4, out_size * 4, 85)
