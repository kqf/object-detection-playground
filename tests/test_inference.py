import torch
import pytest

from detection.model import merge_scales


@pytest.fixture
def predictions(bsize):
    scales = [
        torch.zeros((bsize, 3, 13, 13, 6)),
        torch.zeros((bsize, 3, 26, 26, 6)),
        torch.zeros((bsize, 3, 52, 52, 6)),
    ]
    return scales


@pytest.mark.parametrize("bsize", [16])
def test_merge_scales(predictions, bsize):
    merged = merge_scales(predictions)
    assert len(merged) == bsize
    assert all([x.shape[-1] == 6 for x in merged])
