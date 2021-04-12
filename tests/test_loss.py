import torch
from detection.datasets.v3 import DEFAULT_ANCHORS
from detection.losses.v3 import CombinedLoss


def test_loss():
    target = torch.zeros([1, 3, 13, 13, 6])
    target[..., 0] = 1
    # x, y
    target[..., 1] = 0.5
    target[..., 2] = 0.5

    anchors_scale1 = DEFAULT_ANCHORS[0]
    target[..., 3:5] = anchors_scale1[None, :, None, None, :]
    # NB: class label is zero, this corresponds to the first class
    #     the prediction index for the first class is 5 = 1 (is obj) + 4 coords
    target[..., 5] = 0

    predictions = torch.zeros([1, 85, 13, 13, 3])
    predictions = predictions.transpose(1, -1)
    predictions[..., 0] = 9

    # predictions go through the sigmoid function
    predictions[..., 1] = torch.logit(torch.tensor(0.5))
    predictions[..., 2] = torch.logit(torch.tensor(0.5))

    # It should be scaled through the exp
    predictions[..., 3] = 0
    predictions[..., 4] = 0

    # Set the proper label
    predictions[..., 5] = 1
    # Transpose back
    predictions = predictions.transpose(1, -1)

    criterion = CombinedLoss(DEFAULT_ANCHORS)
    loss = criterion._forward(predictions, target, DEFAULT_ANCHORS[0])
    torch.testing.assert_allclose(loss, 3.403, rtol=1e-3)
