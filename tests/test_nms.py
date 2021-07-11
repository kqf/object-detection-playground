import torch
import pytest
import numpy as np
from detection.inference import nms


@pytest.fixture
def candidates(n_candidates=13 + 26 + 52):
    x = np.zeros((n_candidates, 6))
    x[:, 0] = np.linspace(0.4, 0.6, n_candidates)
    x[n_candidates // 2, 0] = 1
    x[:, 1] = np.linspace(0.4, 0.5, n_candidates)
    x[:, 2] = np.linspace(0.4, 0.5, n_candidates)
    x[:, 3] = 0.2
    x[:, 4] = 0.2
    x[-1, 5] = 1
    return torch.tensor(x)


def test_nms(candidates):
    img = torch.ones(3, 460, 460)
    sup = nms(candidates)
    print(candidates[sup, 1:])
