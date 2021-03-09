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


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels, num_repeats=1):
        super().__init__()
        self.layer = torch.nn.Sequential(
            conv(channels, channels // 2, kernel_size=1),
            conv(channels // 2, channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.layer(x)


def conv(in_channels, out_channels, **kwargs):
    block = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, out_channels, bias=False, **kwargs),
        torch.nn.BatchNorm2d(out_channels),
        torch.nn.LeakyReLU(0.1),
    )
    return block


def residual(channels, num_repeats):
    block = torch.nn.Sequential(*[
        ResidualBlock(channels) for _ in range(num_repeats)
    ])
    return block


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
            layers.append(residual(in_channels, num_repeats=module[0],))

    return torch.nn.Sequential(*layers)
