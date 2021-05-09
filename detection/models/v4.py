import torch


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
