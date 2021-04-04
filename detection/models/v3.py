import torch
import torch.nn as nn

from detection.models.darknet import conv, block, Darknet


def cblock(in_channels, out_channels):
    model = nn.Sequential(
        conv(in_channels, out_channels, kernel_size=1, stride=1),
        conv(out_channels, in_channels, kernel_size=3, stride=1, padding=0),
        block(in_channels),
        conv(in_channels, out_channels, kernel_size=1, padding=0),
    )
    return model


class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.n_preds = num_classes + 5
        self.n_scales = 3
        self.pred = nn.Sequential(
            conv(in_channels // 2, in_channels, kernel_size=3, padding=1),
            torch.nn.Conv2d(in_channels, (num_classes + 5) * 3, kernel_size=1),
        )
        self.num_classes = num_classes

    def forward(self, x):
        b, _, h, w = x.shape

        out = self.pred(x)
        return (
            out.reshape(b, self.n_preds, self.n_scales, h, w)
            .permute(0, 1, 3, 4, 2)
        )


class YOLO(nn.Module):
    def __init__(self, in_channels=3, num_classes=80):
        super().__init__()
        self.backbone = Darknet(in_channels=in_channels)
        self.num_classes = num_classes
        self.in_channels = in_channels

        self.conv1 = torch.nn.Sequential(
            cblock(1024, 1024 // 2),
        )

        self.upsample2 = torch.nn.Sequential(
            conv(1024 // 2, 256, kernel_size=1, stride=1, padding=1),
            torch.nn.Upsample(scale_factor=2),
        )
        self.conv2 = torch.nn.Sequential(
            conv(256 * 3, 256, kernel_size=1, stride=1, padding=1),
            conv(256, 512, kernel_size=3, stride=1, padding=1),
            cblock(512, 512 // 2),
        )

        self.upsample3 = torch.nn.Sequential(
            conv(256, 128, kernel_size=1, stride=1, padding=1),
            torch.nn.Upsample(scale_factor=2),
        )
        self.conv3 = torch.nn.Sequential(
            conv(128 * 3, 128, kernel_size=1, stride=1, padding=1),
            conv(128, 256, kernel_size=3, stride=1, padding=1),
            cblock(256, 256 // 2),
        )

        self.scale1 = ScalePrediction(1024, num_classes)
        self.scale2 = ScalePrediction(512, num_classes)
        self.scale3 = ScalePrediction(256, num_classes)

    def forward(self, x):
        l1, l2, l3 = self.backbone(x)

        x1 = self.conv1(l1)

        xu = self.upsample2(x1)
        x2 = self.conv2(torch.cat([xu, l2], dim=1))

        xu = self.upsample3(x2)
        x3 = self.conv3(torch.cat([xu, l3], dim=1))

        return self.scale1(x1), self.scale2(x2), self.scale3(x3)
