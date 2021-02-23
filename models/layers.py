
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class FasterRCNN(torch.nn.Module):
    def __init__(self, n_classes=15, pretrained=False):
        super().__init__()
        self.backbone = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=pretrained)

        n_classes = 15
        in_features = (
            self.backbone
            .roi_heads
            .box_predictor
            .cls_score.in_features
        )
        self.backbone.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, n_classes)

    def forward(self, x, targets=None):
        if not self.train:
            return self.backbone(x)

        if self.train and targets is None:
            return x

        losses = self.backbone(x, targets)
        return sum(loss for loss in losses.values())
