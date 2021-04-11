import torch
from detection.datasets.v3 import DEFAULT_ANCHORS
from detection.losses.v3 import CombinedLoss


def test_loss():
    target = torch.zeros([1, 3, 13, 13, 6])
    target[..., 0] = 1
    # x, y
    target[..., 1] = 0.5
    target[..., 2] = 0.5
    target[..., 3:5] = DEFAULT_ANCHORS[0][0]

    predictions = torch.zeros([1, 85, 13, 13, 3])
    predictions = predictions.transpose(1, -1)
    predictions[..., 0] = 0.9999

    # predictions go through the sigmoid function
    predictions[..., 1] = torch.logit(0.5)
    predictions[..., 2] = torch.logit(0.5)
    predictions[..., 3] = 0
    predictions[..., 4] = 0

    criterion = CombinedLoss(DEFAULT_ANCHORS)
    loss = criterion._forward(predictions, target, DEFAULT_ANCHORS[0])
    print(loss)
