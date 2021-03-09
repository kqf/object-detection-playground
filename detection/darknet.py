import torch

config = [
    (32, 3, 1),
    (64, 3, 2),
    [1],
    (128, 3, 2),
    [2],
    (256, 3, 2),
    [8],
    (512, 3, 2),
    [8],
    (1024, 3, 2),
    [4],
]


class CNNBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels,
                                    bias=not bn_act, **kwargs)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.leaky = torch.nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            return self.leaky(self.bn(self.conv(x)))
        return self.conv(x)


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for repeat in range(num_repeats):
            self.layers += [
                torch.nn.Sequential(
                    CNNBlock(channels, channels // 2, kernel_size=1),
                    CNNBlock(channels // 2, channels,
                             kernel_size=3, padding=1),
                )
            ]
        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x) * self.use_residual
        return x


def build_darknet(in_channels=3):
    layers = []

    for module in config:
        if isinstance(module, tuple):
            out_channels, kernel_size, stride = module
            layers.append(
                CNNBlock(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=1 if kernel_size == 3 else 0,
                )
            )
            in_channels = out_channels

        if isinstance(module, list):
            layers.append(ResidualBlock(in_channels, num_repeats=module[0],))

    return torch.nn.Sequential(*layers)
