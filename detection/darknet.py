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
            conv(channels, channels // 2, kernel_size=1, padding=0),
            conv(channels // 2, channels, kernel_size=3),
        )

    def forward(self, x):
        return x + self.layer(x)


def conv(in_channels, out_channels, padding=1, **kwargs):
    block = torch.nn.Sequential(
        torch.nn.Conv2d(
            in_channels, out_channels,
            bias=False, padding=padding, **kwargs
        ),
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

    model = torch.nn.Sequential(
        conv(3, 32, kernel_size=3, stride=1),
        conv(32, 64, kernel_size=3, stride=2),
        residual(64, num_repeats=1),
        conv(64, 128, kernel_size=3, stride=2),
        residual(128, num_repeats=2),
        conv(128, 256, kernel_size=3, stride=2),
        residual(256, num_repeats=8),
        conv(256, 512, kernel_size=3, stride=2),
        residual(512, num_repeats=8),
        conv(512, 1024, kernel_size=3, stride=2),
        residual(1024, num_repeats=4),
    )
    return model
