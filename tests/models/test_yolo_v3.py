import pytest
import torch

from detection.models.v3 import YOLO


@pytest.fixture
def batch():
    return torch.rand(16, 3, 256, 256)


def test_module(batch):
    model = YOLO()
    s1, s2, s3 = model(batch)

    assert s1.shape == (16, 3, 8, 8, 85)
    assert s2.shape == (16, 3, 16, 16, 85)
    assert s3.shape == (16, 3, 32, 32, 85)