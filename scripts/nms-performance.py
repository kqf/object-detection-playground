import torch
import timeit
import numpy as np

from detection.inference import nms


def preds(n_candidates=13 * 3 + 26 * 3 + 52 * 3):
    x = np.zeros((n_candidates, 6))
    x[:, 0] = np.linspace(0.4, 0.6, n_candidates)
    x[n_candidates // 2, 0] = 1
    x[:, 1] = np.linspace(0.4, 0.5, n_candidates)
    x[:, 2] = np.linspace(0.4, 0.5, n_candidates)
    x[:, 3] = 0.2
    x[:, 4] = 0.2
    # This one should be detected (high objectness, another label)
    x[-1, 5] = 1
    # This one should be supressed (objectness < 0.5)
    x[0, 5] = 2
    return torch.tensor(x)


def main():
    candidates = preds()
    n_calls = 100
    print(timeit.timeit(lambda: nms(candidates), number=n_calls) / n_calls)


if __name__ == '__main__':
    main()
