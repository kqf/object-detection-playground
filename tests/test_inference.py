import torch
import pytest

from detection.inference import infer, nms, merge_scales
from detection.plot import plot
from detection.datasets.v3 import DEFAULT_ANCHORS


@pytest.fixture
def expected_batch(bsize, scale=13):
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


@pytest.fixture
def batch(expected_batch, scale=13):
    scales = []

    for i, (x, anchors) in enumerate(zip(expected_batch, DEFAULT_ANCHORS)):
        n = x.shape[2]

        x_cells = torch.arange(n).reshape(1, 1, n, 1, 1)
        y_cells = torch.arange(n).reshape(1, 1, 1, n, 1)

        # Convert to local
        x[:, 1] = x[:, 1] * n - x_cells
        x[:, 2] = x[:, 2] * n - y_cells
        x[:, 3:5] /= n

        # Apply inverse "nonlinearity" for predictions
        x[:, 0:3] = torch.logit(x[:, 0:3])
        x[:, 3:5] = torch.log(x[:, 3:5] / anchors.T.reshape(1, 2, 1, 1, 3))

        # Append the global predictions
        scales.append(x)

    return scales


@pytest.fixture
def expected(expected_batch):
    merged_batch = merge_scales([x.permute(0, 2, 3, 4, 1)[..., :5]
                                 for x in expected_batch])
    return merged_batch


@pytest.mark.skip("Fix the label scores")
@pytest.mark.parametrize("bsize", [16])
def test_inference(expected, batch, bsize):
    predictions = infer(batch, DEFAULT_ANCHORS)
    assert len(predictions) == bsize
    # assert all([x.shape[-1] == 5 for x in predictions])

    for pred, nominal in zip(predictions, expected):
        assert pred.shape == nominal.shape
        torch.testing.assert_allclose(pred, nominal)

        # Check if nms works
    for sample in predictions:
        nms(sample)


@pytest.mark.skip()
@pytest.mark.parametrize("bsize", [4])
def test_nms(expected_batch, bsize=10):
    merged_batch = merge_scales([x.permute(0, 2, 3, 4, 1)
                                 for x in expected_batch])
    img = torch.ones(3, 460, 460)
    plot((img, [x[1:5] for x in merged_batch[0]]), convert_bbox=True)
