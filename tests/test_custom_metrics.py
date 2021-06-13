import pytest

import torch
from torch import tensor as tt
from detection.metrics import bbox_iou


@pytest.mark.parametrize("pred, gt, answer", [
    (tt([0.5, 0.5, 1.0, 1.0]), tt([0.5, 0.5, 1.0, 1.0]), 1.0),
    # TODO: Check me
    # (tt([1, 0.5, 1.0, 1.0]), tt([0.5, 0.5, 1.0, 1.0]), 0.5),
])
def test_iou(gt, pred, answer):
    print(bbox_iou(pred, gt).item(), answer)
    torch.testing.assert_allclose(bbox_iou(pred, gt).item(), answer)
