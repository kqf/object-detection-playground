import torch
import torch.nn as nn

from detection.models.darknet import conv, block, Darknet


def scaling(in_channels):
    model = nn.Sequential(
        conv(in_channels, in_channels // 2, kernel_size=1, stride=1),
        conv(in_channels // 2, in_channels, kernel_size=3, stride=1,
             padding=0),
        block(in_channels),
        conv(in_channels, in_channels // 2, kernel_size=1, padding=0),
    )
    return model


class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.xscale = scaling(in_channels)
        self.pred = nn.Sequential(
            conv(in_channels, in_channels // 2, kernel_size=3, padding=1),
            torch.nn.Conv2d(in_channels, (num_classes + 5) * 3, kernel_size=1),
        )
        self.num_classes = num_classes

    def forward(self, x):
        xscale = self.xscale(x)
        out = self.pred(xscale)
        return xscale, (
            out.reshape(
                x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3]
            )
            .permute(0, 1, 3, 4, 2)
        )


class YOLO(nn.Module):
    def __init__(self, in_channels=3, num_classes=80):
        super().__init__()
        self.backbone = Darknet(in_channels=in_channels)
        self.num_classes = num_classes
        self.in_channels = in_channels

        self.scale1 = torch.nn.Sequential(
            ScalePrediction(1024, num_classes),
        )

        self.upsample2 = torch.nn.Sequential(
            conv(1024 // 2, 256, kernel_size=1, stride=1, padding=0),
            torch.nn.Upsample(scale_factor=2),
        )
        self.scale2 = torch.nn.Sequential(
            conv(256 * 3, 256, kernel_size=1, stride=1, padding=0),
            conv(256, 512, kernel_size=3, stride=1, padding=1),
            ScalePrediction(512, num_classes),
        )

        self.upsample3 = torch.nn.Sequential(
            conv(256, 128, kernel_size=1, stride=1, padding=0),
            torch.nn.Upsample(scale_factor=2),
        )
        self.scale3 = torch.nn.Sequential(
            conv(128 * 3, 128, kernel_size=1, stride=1, padding=0),
            conv(128, 256, kernel_size=3, stride=1, padding=1),
            ScalePrediction(256, num_classes),

        )

    def forward(self, x):
        l1, l2, l3 = self.backbone(x)

        xscale, scale1 = self.scale1(l1)
        x = self.upsample2(xscale)
        xscale, scale2 = self.scale2(torch.cat([x, l2], dim=1))

        x = self.upsample3(xscale)
        _, scale3 = self.scale3(torch.cat([x, l3], dim=1))

        return scale1, scale2, scale3
