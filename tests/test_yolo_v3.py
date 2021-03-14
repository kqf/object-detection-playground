import pytest
import torch

from detection.models.v3 import YOLO


@pytest.fixture
def batch():
    return torch.rand(64, 3, 256, 256)


def test_module(batch):
    model = YOLO()
    l1, l2, l3 = model(batch)

    assert l1.shape == (64, 1024, 8, 8)
    assert l2.shape == (64, 512, 16, 16)
    assert l3.shape == (64, 256, 32, 32)
