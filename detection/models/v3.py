import torch
import torch.nn as nn

from detection.models.darknet import conv, block, build_darknet


class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pred = nn.Sequential(
            block(2 * in_channels),
            conv(2 * in_channels, in_channels, kernel_size=1),
            conv(in_channels, 2 * in_channels, kernel_size=3, padding=1),
            torch.nn.Conv2d(
                2 * in_channels, (num_classes + 5) * 3, kernel_size=1
            ),
        )
        self.num_classes = num_classes

    def forward(self, x):
        return (
            self.pred(x)
            .reshape(
                x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3]
            )
            .permute(0, 1, 3, 4, 2)
        )


class YOLOv3(nn.Module):
    def __init__(self, in_channels=3, num_classes=80):
        super().__init__()
        self.backbone = build_darknet(in_channels=in_channels)
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.layers = self._create_conv_layers()

        self.scale1 = torch.nn.Sequential(
            conv(1024, 512, 1, 1),
            conv(512, 1024, 3, 1),
            ScalePrediction(1024 // 2, num_classes),
        )
        self.scale2 = torch.nn.Sequential(
            conv(1024 // 2, 256, 1, 1),
            torch.nn.Upsample(scale_factor=2),
            conv(256, 256, 1, 1),
            conv(256, 512, 3, 1),
            ScalePrediction(512 // 2, num_classes),
        )
        self.scale2 = torch.nn.Sequential(
            conv(256, 128, 1, 1),
            torch.nn.Upsample(scale_factor=2),
            conv(128, 128, 1, 1),
            conv(128, 256, 3, 1),
            ScalePrediction(256 // 2, num_classes),

        )

    def forward(self, x):
        x = self.backbone(x)

        scale1 = self.scale1(x)
        scale2 = self.scale2(scale1)
        scale3 = self.scale3(scale2)

        return scale1, scale2, scale3
