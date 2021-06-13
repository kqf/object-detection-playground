import pytest

import torch
from torch import tensor as tt
from detection.metrics import bbox_iou
from detection.plot import check_boxes


@pytest.mark.parametrize("pred, gt, answer", [
    # (tt([0.5, 0.5, 1.0, 1.0]), tt([0.5, 0.5, 1.0, 1.0]), 1.0),
    # TODO: Check me
    (tt([0.75, 0.5, 0.5, 0.5]), tt([0.5, 0.5, 0.5, 0.5]), 0.5),
])
def test_iou(gt, pred, answer):
    print(bbox_iou(pred, gt).item(), answer)
    check_boxes([pred, gt])
    torch.testing.assert_allclose(bbox_iou(pred, gt).item(), answer)
