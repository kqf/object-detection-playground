import torch

from detection.models.darknet import Residual


class Mish(torch.nn.Module):
    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))


def conv(in_channels, out_channels, kernel_size, stride, activation,
         bn=True, bias=False):
    pad = (kernel_size - 1) // 2

    layers = [
        torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                        stride, pad, bias=bias),
        torch.nn.BatchNorm2d(out_channels) if bn else torch.nn.Identity(),
        activation(),
    ]
    return torch.nn.Sequential(*layers)


def resblock(ch, nblocks=1, activation=torch.nn.ReLU):
    layers = []
    for _ in range(nblocks):
        layers.append(Residual(conv(ch, ch, 1, 1, activation=activation)))
        layers.append(Residual(conv(ch, ch, 3, 1, activation=activation)))
    return torch.nn.Sequential(*layers)


class DownSample(torch.nn.Module):
    pass


class Upsample(torch.nn.Module):
    def forward(self, x, tsize, inference=False):
        assert (x.data.dim() == 4)
        # _, _, tH, tW = tsize

        size = (tsize[2], tsize[3])
        return torch.nn.functional.interpolate(x, size=size, mode='nearest')


class Neck(torch.nn.Module):
    def __init__(self, activation=torch.nn.LeakyReLU):
        super().__init__()

        self.conv1 = conv(1024, 512, 1, 1, activation)
        self.conv2 = conv(512, 1024, 3, 1, activation)
        self.conv3 = conv(1024, 512, 1, 1, activation)

        # SPP
        self.maxpool1 = torch.nn.MaxPool2d(5, stride=1, padding=5 // 2)
        self.maxpool2 = torch.nn.MaxPool2d(9, stride=1, padding=9 // 2)
        self.maxpool3 = torch.nn.MaxPool2d(13, stride=1, padding=13 // 2)

        # R -1 -3 -5 -6
        # SPP
        self.conv4 = conv(2048, 512, 1, 1, activation)
        self.conv5 = conv(512, 1024, 3, 1, activation)
        self.conv6 = conv(1024, 512, 1, 1, activation)
        self.conv7 = conv(512, 256, 1, 1, activation)
        # UP
        self.upsample1 = Upsample()
        # R 85
        self.conv8 = conv(512, 256, 1, 1, activation)
        # R -1 -3
        self.conv9 = conv(512, 256, 1, 1, activation)
        self.conv10 = conv(256, 512, 3, 1, activation)
        self.conv11 = conv(512, 256, 1, 1, activation)
        self.conv12 = conv(256, 512, 3, 1, activation)
        self.conv13 = conv(512, 256, 1, 1, activation)
        self.conv14 = conv(256, 128, 1, 1, activation)
        # UP
        self.upsample2 = Upsample()
        # R 54
        self.conv15 = conv(256, 128, 1, 1, activation)
        # R -1 -3
        self.conv16 = conv(256, 128, 1, 1, activation)
        self.conv17 = conv(128, 256, 3, 1, activation)
        self.conv18 = conv(256, 128, 1, 1, activation)
        self.conv19 = conv(128, 256, 3, 1, activation)
        self.conv20 = conv(256, 128, 1, 1, activation)

    def forward(self, input, downsample4, downsample3):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        # SPP
        m1 = self.maxpool1(x3)
        m2 = self.maxpool2(x3)
        m3 = self.maxpool3(x3)
        spp = torch.cat([m3, m2, m1, x3], dim=1)
        # SPP end
        x4 = self.conv4(spp)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        # UP
        up = self.upsample1(x7, downsample4.size())
        # R 85
        x8 = self.conv8(downsample4)
        # R -1 -3
        x8 = torch.cat([x8, up], dim=1)

        x9 = self.conv9(x8)
        x10 = self.conv10(x9)
        x11 = self.conv11(x10)
        x12 = self.conv12(x11)
        x13 = self.conv13(x12)
        x14 = self.conv14(x13)

        # UP
        up = self.upsample2(x14, downsample3.size())
        # R 54
        x15 = self.conv15(downsample3)
        # R -1 -3
        x15 = torch.cat([x15, up], dim=1)

        x16 = self.conv16(x15)
        x17 = self.conv17(x16)
        x18 = self.conv18(x17)
        x19 = self.conv19(x18)
        x20 = self.conv20(x19)
        return x20, x13, x6


class DownSample1(torch.nn.Module):
    def __init__(self, activation=Mish):
        super().__init__()
        self.conv1 = conv(3, 32, 3, 1, activation)

        self.conv2 = conv(32, 64, 3, 2, activation)
        self.conv3 = conv(64, 64, 1, 1, activation)
        # [route]
        # layers = -2
        self.conv4 = conv(64, 64, 1, 1, activation)

        self.conv5 = conv(64, 32, 1, 1, activation)
        self.conv6 = conv(32, 64, 3, 1, activation)
        # [shortcut]
        # from=-3
        # activation = linear

        self.conv7 = conv(64, 64, 1, 1, activation)
        # [route]
        # layers = -1, -7
        self.conv8 = conv(128, 64, 1, 1, activation)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        # route -2
        x4 = self.conv4(x2)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        # shortcut -3
        x6 = x6 + x4

        x7 = self.conv7(x6)
        # [route]
        # layers = -1, -7
        x7 = torch.cat([x7, x3], dim=1)
        x8 = self.conv8(x7)
        return x8


class DownSample2(torch.nn.Module):
    def __init__(self, activation=Mish):
        super().__init__()
        self.conv1 = conv(64, 128, 3, 2, activation)
        self.conv2 = conv(128, 64, 1, 1, activation)
        # r -2
        self.conv3 = conv(128, 64, 1, 1, activation)

        self.resblock = resblock(ch=64, nblocks=2)

        # s -3
        self.conv4 = conv(64, 64, 1, 1, activation)
        # r -1 -10
        self.conv5 = conv(128, 128, 1, 1, activation)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)

        r = self.resblock(x3)
        x4 = self.conv4(r)

        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.conv5(x4)
        return x5


class DownSample3(torch.nn.Module):
    def __init__(self, activation=Mish):
        super().__init__()
        self.conv1 = conv(128, 256, 3, 2, activation)
        self.conv2 = conv(256, 128, 1, 1, activation)
        self.conv3 = conv(256, 128, 1, 1, activation)

        self.resblock = resblock(ch=128, nblocks=8)
        self.conv4 = conv(128, 128, 1, 1, activation)
        self.conv5 = conv(256, 256, 1, 1, activation)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)

        r = self.resblock(x3)
        x4 = self.conv4(r)

        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.conv5(x4)
        return x5


class DownSample4(torch.nn.Module):
    def __init__(self, activation=Mish):
        super().__init__()
        self.conv1 = conv(256, 512, 3, 2, activation)
        self.conv2 = conv(512, 256, 1, 1, activation)
        self.conv3 = conv(512, 256, 1, 1, activation)

        self.resblock = resblock(ch=256, nblocks=8)
        self.conv4 = conv(256, 256, 1, 1, activation)
        self.conv5 = conv(512, 512, 1, 1, activation)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)

        r = self.resblock(x3)
        x4 = self.conv4(r)

        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.conv5(x4)
        return x5


class DownSample5(torch.nn.Module):
    def __init__(self, activation=Mish):
        super().__init__()
        self.conv1 = conv(512, 1024, 3, 2, activation)
        self.conv2 = conv(1024, 512, 1, 1, activation)
        self.conv3 = conv(1024, 512, 1, 1, activation)

        self.resblock = resblock(ch=512, nblocks=4)
        self.conv4 = conv(512, 512, 1, 1, activation)
        self.conv5 = conv(1024, 1024, 1, 1, activation)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)

        r = self.resblock(x3)
        x4 = self.conv4(r)

        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.conv5(x4)
        return x5


class Head(torch.nn.Module):
    def __init__(self, output_ch, n_classes):
        super().__init__()
        leaky = torch.nn.LeakyReLU
        lin = torch.nn.Identity
        self.conv1 = conv(128, 256, 3, 1, leaky)
        self.conv2 = conv(256, output_ch, 1, 1, lin, bn=False, bias=True)

        # First scale

        # R -4
        self.conv3 = conv(128, 256, 3, 2, leaky)

        # R -1 -16
        self.conv4 = conv(512, 256, 1, 1, leaky)
        self.conv5 = conv(256, 512, 3, 1, leaky)
        self.conv6 = conv(512, 256, 1, 1, leaky)
        self.conv7 = conv(256, 512, 3, 1, leaky)
        self.conv8 = conv(512, 256, 1, 1, leaky)
        self.conv9 = conv(256, 512, 3, 1, leaky)
        self.conv10 = conv(512, output_ch, 1, 1, lin, bn=False, bias=True)

        # Second scale

        # R -4
        self.conv11 = conv(256, 512, 3, 2, leaky)

        # R -1 -37
        self.conv12 = conv(1024, 512, 1, 1, leaky)
        self.conv13 = conv(512, 1024, 3, 1, leaky)
        self.conv14 = conv(1024, 512, 1, 1, leaky)
        self.conv15 = conv(512, 1024, 3, 1, leaky)
        self.conv16 = conv(1024, 512, 1, 1, leaky)
        self.conv17 = conv(512, 1024, 3, 1, leaky)
        self.conv18 = conv(1024, output_ch, 1, 1, lin, bn=False, bias=True)
        # The third scale

    def forward(self, scale1, scale2, scale3):
        x1 = self.conv1(scale1)
        x2 = self.conv2(x1)

        x3 = self.conv3(scale1)
        # R -1 -16
        x3 = torch.cat([x3, scale2], dim=1)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        x8 = self.conv8(x7)
        x9 = self.conv9(x8)
        x10 = self.conv10(x9)

        # R -4
        x11 = self.conv11(x8)
        # R -1 -37
        x11 = torch.cat([x11, scale3], dim=1)

        x12 = self.conv12(x11)
        x13 = self.conv13(x12)
        x14 = self.conv14(x13)
        x15 = self.conv15(x14)
        x16 = self.conv16(x15)
        x17 = self.conv17(x16)
        x18 = self.conv18(x17)

        return x2, x10, x18


class YOLO(torch.nn.Module):
    """
    Heavily inspired by:

    https://github.com/Tianxiaomo/pytorch-YOLOv4/blob/master/models.py
    """

    def __init__(self, n_classes=80):
        super().__init__()

        # backbone
        self.down1 = DownSample1()
        self.down2 = DownSample2()
        self.down3 = DownSample3()
        self.down4 = DownSample4()
        self.down5 = DownSample5()
        # neck
        self.neek = Neck()

        ochannels = (4 + 1 + n_classes) * 3
        self.head = Head(ochannels, n_classes)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        scale1, scale2, scale3 = self.neek(d5, d4, d3)
        output = self.head(scale1, scale2, scale3)
        return output
