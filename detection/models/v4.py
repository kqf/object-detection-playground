import torch

from detection.models.darknet import Residual


class Conv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 activation=torch.nn.ReLU(), bn=True, bias=False):
        super().__init__()
        pad = (kernel_size - 1) // 2

        layers = [
            torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                            stride, pad, bias=bias),
            torch.nn.BatchNorm2d(out_channels) if bn else torch.nn.Identity(),
            activation,
        ]
        self.conv = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


def res_block(self, ch, nblocks=1, activation=torch.nn.ReLU):
    layers = []
    for _ in range(nblocks):
        layers.append(Residual(Conv(ch, ch, 1, 1, activation=activation)))
        layers.append(Residual(Conv(ch, ch, 3, 1, activation=activation)))
    return torch.nn.Sequential(*layers)


class DownSample(torch.nn.Module):
    pass


class Neck(torch.nn.Module):
    def __init__(self, inference):
        super().__init__()
        pass


class Head(torch.nn.Module):
    def __init__(self, inference):
        super().__init__()
        pass


class YOLO(torch.nn.Module):
    """
    Heavily inspired by:

    https://github.com/Tianxiaomo/pytorch-YOLOv4/blob/master/models.py
    """

    def __init__(self, n_classes=80, inference=False):
        super().__init__()

        # backbone
        self.down1 = DownSample()
        self.down2 = DownSample()
        self.down3 = DownSample()
        self.down4 = DownSample()
        self.down5 = DownSample()
        # neck
        self.neek = Neck(inference)

        ochannels = (4 + 1 + n_classes) * 3
        self.head = Head(ochannels, n_classes, inference)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        scale1, scale2, scale3 = self.neek(d5, d4, d3)
        output = self.head(scale1, scale2, scale3)
        return output
