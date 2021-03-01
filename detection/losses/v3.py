import torch
import torch.nn as nn

from utils import iou


class YoloLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        self.lambda_class = 1
        self.lambda_noobj = 1
        self.lambda_obj = 1
        self.lambda_box = 1

    def forward(self, predictions, target, anchors):
        # Check where obj and noobj (we ignore if target == -1)
        obj = target[..., 0] == 1  # in paper this is Iobj_i
        noobj = target[..., 0] == 0  # in paper this is Inoobj_i

        no_detection = self.bce(
            (predictions[..., 0:1][noobj]), (target[..., 0:1][noobj]),
        )

        anchors = anchors.reshape(1, 3, 1, 1, 2)
        box_preds = torch.cat([
            self.sigmoid(predictions[..., 1:3]),
            torch.exp(predictions[..., 3:5]) * anchors
        ], dim=-1)

        ious = iou(box_preds[obj], target[..., 1:5][obj]).detach()
        detection = self.bce(
            (predictions[..., 0:1][obj]), (ious * target[..., 0:1][obj]))

        # x,y coordinates
        predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3])

        # width, height coordinate
        target[..., 3:5] = torch.log((1e-16 + target[..., 3:5] / anchors))
        box = self.mse(predictions[..., 1:5][obj], target[..., 1:5][obj])

        classification = self.entropy(
            (predictions[..., 5:][obj]), (target[..., 5][obj].long()),
        )

        return (
            self.box * box
            + self.obj * detection
            + self.noobj * no_detection
            + self.classification * classification
        )  # noqa
