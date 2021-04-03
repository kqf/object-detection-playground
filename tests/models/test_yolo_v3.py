import pytest
import torch

from detection.models.v3 import YOLO
from detection.models.legacy import YOLO as LOLO


@pytest.fixture
def batch():
    return torch.rand(4, 3, 416, 416)


@pytest.mark.skip("Fix the structure of this model")
def test_module(batch):
    model = YOLO()
    s1, s2, s3 = model(batch)

    assert s1.shape == (4, 3, 8, 8, 85)
    assert s2.shape == (4, 3, 16, 16, 85)
    assert s3.shape == (4, 3, 32, 32, 85)


def test_legacy_module(batch):
    model = LOLO()
    s1, s2, s3 = model(batch)

    assert s1.shape == (4, 85, 13, 13, 3)
    assert s2.shape == (4, 85, 26, 26, 3)
    assert s3.shape == (4, 85, 52, 52, 3)
