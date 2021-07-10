import torch
import pytest
import numpy as np
# from detection.ionference import nms
from detection.plot import plot


@pytest.fixture
def candidates(n_candidates=13 + 26 + 52):
    x = np.zeros((n_candidates, 6))
    x[:, 0] = np.linspace(0, 0.6, n_candidates)
    x[n_candidates // 2, 0] = 1
    x[:, 1] = np.linspace(0.2, 0.8, n_candidates)
    x[:, 2] = np.linspace(0.8, 0.2, n_candidates)
    x[:, 3] = 0.2
    x[:, 4] = 0.2
    return x


def test_nms(candidates):
    img = torch.ones(3, 460, 460)
    plot((img, candidates[:, 1:], []), convert_bbox=True)
