import pytest
import torch

from detection.models.v3 import YOLO


@pytest.fixture
def batch():
    return torch.rand(16, 3, 256, 256)


def test_module(batch):
    model = YOLO()
    s1, s2, s3 = model(batch)

    assert s1.shape == (64, 1024, 8, 8)
    assert s2.shape == (64, 512, 16, 16)
    assert s3.shape == (64, 256, 32, 32)
