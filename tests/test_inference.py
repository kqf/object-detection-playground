import torch
import pytest

from detection.inference import infer, nms, merge_scales
from detection.plot import plot
from detection.datasets.v3 import DEFAULT_ANCHORS


@pytest.fixture
def batch(bsize, scale=13):
    scales = []

    for i, n in enumerate([scale, scale * 2, scale * 4]):
        x = torch.zeros([bsize, 85, n, n, 3])
        x_cells = torch.arange(n).reshape(1, 1, n, 1, 1)
        y_cells = torch.arange(n).reshape(1, 1, 1, n, 1)

        x[:, 0] = torch.zeros(1, n, n, 3) + 0.9
        x[:, 1] = (torch.zeros(1, n, n, 3) + 0.5 + x_cells) / n
        x[:, 2] = (torch.zeros(1, n, n, 3) + 0.5 + y_cells) / n
        x[:, 3] = 0.01
        x[:, 4] = 0.01
        x[:, 5] = 1

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


@pytest.mark.parametrize("bsize", [4])
def test_nms(batch, bsize=10):
    merged_batch = merge_scales([x.permute(0, 2, 3, 4, 1) for x in batch])
    img = torch.ones(3, 460, 460)
    plot((img, [x[1:5] for x in merged_batch[0]]), convert_bbox=True)
