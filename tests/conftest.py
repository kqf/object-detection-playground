import cv2
import pytest
import tempfile
import numpy as np
import pandas as pd

from pathlib import Path

from detection.mc import make_blob, blob2image


@pytest.fixture
def size():
    return 256


@pytest.fixture
def annotations():
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
    return df


@pytest.fixture
def fake_dataset(annotations, size=256):
    with tempfile.TemporaryDirectory() as dirname:
        path = Path(dirname)
        for row in annotations.to_dict(orient="records"):
            image_id = row["image_id"]
            img = blob2image(make_blob(**row))
            ifile = f"{image_id}.png"
            cv2.imwrite(str(path / ifile), img)

        annotations.to_csv(path / "train.csv", index=False)
        yield path
