import torch
import torch.nn as nn

config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],
]


class CNN(nn.Module):
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              bias=not bn_act, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)


class Residual(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()
        self.layers = nn.ModuleList()
        for repeat in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    CNN(channels, channels // 2, kernel_size=1),
                    CNN(channels // 2, channels, kernel_size=3, padding=1),
                )
            ]

        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            if self.use_residual:
                x = x + layer(x)
            else:
                x = layer(x)

        return x


class ScalePrediction(torch.nn.Module):
    pass


def build_model(in_channels, num_classes):
    layers = nn.ModuleList()
    for module in config:
        if isinstance(module, tuple):
            out_channels, kernel_size, stride = module
            layers.append(
                CNN(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=1 if kernel_size == 3 else 0,
                )
            )
            in_channels = out_channels

        elif isinstance(module, list):
            num_repeats = module[1]
            layers.append(Residual(in_channels, num_repeats=num_repeats,))

        elif isinstance(module, str):
            if module == "S":
                layers += [
                    Residual(
                        in_channels, use_residual=False, num_repeats=1),
                    CNN(in_channels, in_channels // 2, kernel_size=1),
                    ScalePrediction(in_channels // 2, num_classes=num_classes),
                ]
                in_channels = in_channels // 2

            elif module == "U":
                layers.append(nn.Upsample(scale_factor=2),)
                in_channels = in_channels * 3

    return layers
