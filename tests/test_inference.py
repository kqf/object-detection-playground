import torch
import pytest

from detection.inference import infer, nms, to_global, merge_scales
from detection.plot import plot
from detection.datasets.v3 import DEFAULT_ANCHORS


@pytest.fixture
def batch(bsize, scale=13):
    scales = []

    for n in [scale, scale * 2, scale * 4]:
        x = torch.zeros([bsize, 85, n, n, 3])
        x[:, 0] = torch.zeros(1, n, n, 3) + 0.8
        x[:, 1] = torch.zeros(1, n, n, 3) + 0.5
        x[:, 2] = torch.zeros(1, n, n, 3) + 0.5
        x[:, 3] = 0.01 * n
        x[:, 4] = 0.01 * n

        # Conver to the global scale
        x_nonlin = to_global(x[1:].permute(0, 2, 3, 4, 1), n)
        x[1:] = x_nonlin.permute(0, 4, 1, 2, 3).detach().clone()

        # Append the global predictions
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


@pytest.mark.parametrize("bsize", [4])
def test_nms(batch, bsize=10):
    merged_batch = merge_scales([x.permute(0, 2, 3, 4, 1) for x in batch])
    img = torch.ones(3, 460, 460)
    plot((img, [x[1:5] for x in merged_batch[0]]), convert_bbox=True)
