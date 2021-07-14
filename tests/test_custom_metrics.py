import pytest

import torch
from torch import tensor as tt
from detection.metrics import bbox_iou, mAP
# from detection.plot import check_boxes


@pytest.mark.parametrize("pred, gt, answer", [
    (tt([0.5, 0.5, 1.0, 1.0]), tt([0.5, 0.5, 1.0, 1.0]), 1.0),
    (tt([1.0, 0.5, 1.0, 1.0]), tt([0.5, 0.5, 1.0, 1.0]), 0.3333333),
    (tt([1.0, 1.0, 1.0, 1.0]), tt([0.5, 0.5, 1.0, 1.0]), 0.25 / (2 - 0.25)),
    (tt([1.0, 1.0, 0.5, 0.5]), tt([0.5, 0.5, 0.5, 0.5]), 0),
])
def test_iou(gt, pred, answer):
    print(bbox_iou(pred, gt).item(), answer)
    # check_boxes([pred, gt])
    torch.testing.assert_allclose(bbox_iou(pred, gt).item(), answer)


@pytest.fixture
def preds(n_samples):
    x = torch.zeros(n_samples, 5)
    x[:, 0] = torch.linspace(0.4, 0.5, n_samples)
    x[:, 1] = torch.linspace(0.4, 0.5, n_samples)
    x[:, 2] = 0.2
    x[:, 3] = 0.2
    x[:, 4] = torch.linspace(0.0, 1, n_samples)
    return x


def test_map(preds):
    x = mAP(preds, preds)
    assert torch.allclose(x, torch.tensor(1.))
