
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor



class YOLO(torch.nn.Module):
    def __init__(self, n_classes=15, pretrained=False):
        super().__init__()
        self.model = torch.hub.load(
            'ultralytics/yolov5', 'yolov5s',
            pretrained=pretrained, autoshape=False)

    def forward(self, x):
        return self.model(x)


class Loss(torch.nn.Module):
    def forward(self, y_pred, y_true):
        small, medium, large = y_pred
        return large.sum()
