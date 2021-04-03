import pytest
import torch

from detection.models.darknet import Darknet
from detection.models.legacy import build_model


@pytest.fixture
def batch():
    return torch.rand(64, 3, 256, 256)


def test_module(batch):
    model = Darknet()
    l1, l2, l3 = model(batch)

    assert l1.shape == (64, 1024, 8, 8)
    assert l2.shape == (64, 512, 16, 16)
    assert l3.shape == (64, 256, 32, 32)


def test_legacy(batch):
    model = torch.nn.Sequential(*build_model(3, 40))
    l1 = model(batch)
    l2 = model[10].pop()
    l3 = model[7].pop()

    assert l1.shape == (64, 1024, 8, 8)
    assert l2.shape == (64, 512, 16, 16)
    assert l3.shape == (64, 256, 32, 32)
