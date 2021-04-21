import torch
import pytest

from detection.inference import merge_scales, nms
from detection.plot import plot


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
    nms(merged[0])


@pytest.fixture
def merged_batch(predictions_size=6):
    preds = torch.zeros(predictions_size, 6)
    preds[:, 0] = torch.linspace(0, 0.99, predictions_size)
    preds[:, 1] = torch.linspace(0.4, 0.6, predictions_size)
    preds[:, 2] = torch.linspace(0.4, 0.6, predictions_size)
    preds[:, 3] = 0.4
    preds[:, 4] = 0.4
    return preds


@pytest.mark.skip("Implement me")
def test_nms(merged_batch):
    img = torch.ones(3, 460, 460)
    plot((img, [x[1:5] for x in merged_batch]), convert_bbox=True)
