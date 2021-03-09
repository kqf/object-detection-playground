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


def conv(in_channels, out_channels, **kwargs):
    block = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, out_channels, bias=False, **kwargs),
        torch.nn.BatchNorm2d(out_channels),
        torch.nn.LeakyReLU(0.1),
    )
    return block


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for repeat in range(num_repeats):
            self.layers += [
                torch.nn.Sequential(
                    conv(channels, channels // 2, kernel_size=1),
                    conv(channels // 2, channels, kernel_size=3, padding=1),
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
                conv(
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
