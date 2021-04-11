import torch
from detection.datasets.v3 import DEFAULT_ANCHORS
from detection.losses.v3 import CombinedLoss


def test_loss():
    target = torch.ones([1, 3, 13, 13, 6])
    predictions = torch.ones([1, 85, 13, 13, 3])

    criterion = CombinedLoss(DEFAULT_ANCHORS)
    loss = criterion._forward(predictions, target, DEFAULT_ANCHORS[0])
    print(loss)
