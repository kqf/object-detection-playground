import torch


class Residual(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer

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


def block(channels):
    layer = torch.nn.Sequential(
        conv(channels, channels // 2, kernel_size=1, padding=0),
        conv(channels // 2, channels, kernel_size=3),
    )
    return layer


def residual(channels, num_repeats):
    resblock = torch.nn.Sequential(*[
        Residual(block(channels)) for _ in range(num_repeats)
    ])
    return resblock


def build_darknet(in_channels=3):
    model = torch.nn.Sequential(
        conv(in_channels, 32, kernel_size=3, stride=1),
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