import pytest
import torch

from detection.models.darknet import Darknet


@pytest.fixture
def batch():
    return torch.rand(64, 3, 256, 256)


def test_module(batch):
    model = Darknet()
    assert model(batch).shape == (64, 1024, 8, 8)
