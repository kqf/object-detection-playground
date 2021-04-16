import pytest
import torch

from detection.plot import plot


@pytest.fixture
def batch(predictions_size=6):
    preds = torch.zeros(predictions_size, 6)
    preds[:, 0] = torch.linspace(0, 0.99, predictions_size)
    preds[:, 1] = torch.linspace(0.4, 0.6, predictions_size)
    preds[:, 2] = torch.linspace(0.4, 0.6, predictions_size)
    preds[:, 3] = 0.4
    preds[:, 4] = 0.4
    return preds


def test_nms(batch):
    img = torch.ones(3, 460, 460)
    plot((img, [x[1:5] for x in batch]), convert_bbox=True)
