import torch
import random

import pytest
import tempfile
import numpy as np
import pandas as pd

from pathlib import Path

from detection.mc import generate_to_directory


def pytest_addoption(parser):
    parser.addoption(
        "--max-epochs",
        action="store",
        default=2,
        type=int,
        help="Number of epochs to run the tests",
    )
    parser.addoption(
        "--num-images-per-batch",
        action="store",
        default=4,
        type=int,
        help="Number of epochs to run the tests",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "onlylocal: mark test to run only as they require the data ",
    )


@pytest.fixture
def fixed_seed(seed=137):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


@pytest.fixture
def size():
    return 256


@pytest.fixture
def n_samples(request):
    return request.config.getoption("--num-images-per-batch")


@pytest.fixture
def annotations(n_samples, fixed_seed, width=2000):
    """
                               image_id          class_name  class_id rad_id   x_min   y_min   x_max   y_max
    0  50a418190bc3fb1ef1633bf9678929b3          No finding        14    R11     NaN     NaN     NaN     NaN
    1  21a10246a5ec7af151081d0cd6d65dc9          No finding        14     R7     NaN     NaN     NaN     NaN
    2  9a5094b2563a1ef3ff50dc5c7ff71345        Cardiomegaly         3    R10   691.0  1375.0  1653.0  1831.0
    3  051132a778e61a86eb147c7c6f564dfe  Aortic enlargement         0    R10  1264.0   743.0  1611.0  1019.0
    4  063319de25ce7edb9b1c6b8881290140          No finding        14    R10     NaN     NaN     NaN     NaN
    """  # noqa

    data = {
        "image_id": [1, 2, 3, 4, 5],
        "class_name": ["No", "No", "Cardiomegaly", "Aortic", "No"],
        "class_id": [14, 14, 3, 0, 14],
        "x_min": [np.nan, np.nan, 691.0, 1264.0, np.nan],
        "y_min": [np.nan, np.nan, 1375.0, 743.0, np.nan],
        "x_max": [np.nan, np.nan, 1653.0, 1611.0, np.nan],
        "y_max": [np.nan, np.nan, 1831.0, 1019.0, np.nan],
    }

    df = pd.DataFrame(data)
    df.loc[df["class_id"] == 14, 'x_min'] = 691.0
    df.loc[df["class_id"] == 14, 'x_max'] = 1653.0
    df.loc[df["class_id"] == 14, 'y_min'] = 1375.0
    df.loc[df["class_id"] == 14, 'y_max'] = 1831.0

    shift = 1 + df.index / len(df)
    shift = 1
    df.loc[:, 'x_min'] = 200.0 * shift
    df.loc[:, 'x_max'] = 200.0 * shift + width * 0.28
    df.loc[:, 'y_min'] = 400.0 * shift
    df.loc[:, 'y_max'] = 400.0 * shift + width * 0.22
    df.loc[:, "class_id"] = 1

    df["h"] = width
    df["w"] = width

    x1, y1, x2, y2 = df[['x_min', 'y_min', 'x_max', 'y_max']].values.T
    df['x_center'] = (x1 + x2) / 2 / df["w"]
    df['y_center'] = (y1 + y2) / 2 / df["h"]
    df['width'] = (x2 - x1) / df["w"]
    df['height'] = (y2 - y1) / df["h"]

    # TODO: remove this part when debugging is over
    df = df.sample(n=n_samples, replace=True).reset_index(drop=True)
    df["image_id"] = df.index
    return df


@pytest.fixture
def fake_dataset(annotations, size=256):
    with tempfile.TemporaryDirectory() as dirname:
        path = Path(dirname)
        generate_to_directory(annotations, path)
        yield path
