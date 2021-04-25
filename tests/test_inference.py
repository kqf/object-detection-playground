import torch
import pytest

from detection.inference import infer, nms
from detection.plot import plot
from detection.datasets.v3 import DEFAULT_ANCHORS


@pytest.fixture
def batch(bsize, scale=13):
    scales = []

    for n in [scale, scale * 2, scale * 4]:
        x = torch.zeros([bsize, 85, n, n, 3])
        x[:, 0] = torch.linspace(0, 0.9, n * n * 3).reshape(1, n, n, 3)
        x[:, 1] = torch.linspace(0.4, 0.6, n * n * 3).reshape(1, n, n, 3)
        x[:, 2] = torch.linspace(0.4, 0.6, n * n * 3).reshape(1, n, n, 3)
        x[:, 3] = 0.2
        x[:, 4] = 0.2
        scales.append(x)

    return scales


@pytest.mark.parametrize("bsize", [16])
def test_inference(batch, bsize):
    predictions = infer(batch, DEFAULT_ANCHORS)
    assert len(predictions) == bsize
    assert all([x.shape[-1] == 5 for x in predictions])

    # Check if nms works
    for sample in predictions:
        nms(sample)


@pytest.fixture
def merged_batch(predictions_size=6):
    preds = torch.zeros(predictions_size, 6)
    preds[:, 0] = torch.linspace(0, 0.99, predictions_size)
    preds[:, 1] = torch.linspace(0.4, 0.6, predictions_size)
    preds[:, 2] = torch.linspace(0.4, 0.6, predictions_size)
    preds[:, 3] = 0.2
    preds[:, 4] = 0.2
    return preds


@pytest.mark.skip
def test_nms(merged_batch):
    img = torch.ones(3, 460, 460)
    plot((img, [x[1:5] for x in merged_batch]), convert_bbox=True)
