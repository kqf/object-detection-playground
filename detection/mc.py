import cv2
import pandas as pd
import numpy as np

from pathlib import Path


def make_blob(
    x_min=50, y_min=50,
    x_max=90, y_max=90,
    h=2000, w=2000,
    **kwargs
):
    Y, X = np.ogrid[:h, :w]

    w = (x_max - x_min)
    h = (y_max - y_min)

    cx = x_min + w / 2.
    cy = y_min + h / 2.

    xx = (X[..., None] - cx)
    yy = (Y[..., None] - cy)
    dists = np.sqrt((xx / w) ** 2 + (yy / h) ** 2)

    mask = dists <= 1. / 2.
    return mask.sum(axis=-1).astype(np.uint8)


def blob2image(blob, channels=3, epsilon=0.1):
    h, w = blob.shape

    extended = blob[..., None]

    # Add a small term to add noise to the empty regions
    noise = np.random.poisson(extended + epsilon, size=(h, w, channels))

    # Convet to image scale
    return (extended + noise * 255).astype(np.uint8)


def annotations(n_points=1000, h=2000, w=2000):
    x = np.random.uniform(0, w, (n_points, 2))
    y = np.random.uniform(0, h, (n_points, 2))
    df = pd.DataFrame({"image_id": np.arange(n_points)})
    df["image_id"] = [1, 2, 3, 4, 5],

    df["x_min"] = x.min(axis=1)
    df["y_min"] = y.min(axis=1)

    df["x_max"] = x.max(axis=0)
    df["y_max"] = x.max(axis=0)
    labels = (df["x_max"] - df["x_min"]) > (df["y_max"] - df["y_min"])
    df["class_id"] = labels.astype(int)
    df["class_name"] = labels.astype(str)
    return df


def generate_to_directory(annotations, dirname):
    path = Path(dirname)
    for row in annotations.to_dict(orient="records"):
        image_id = row["image_id"]
        img = blob2image(make_blob(**row))
        ifile = f"{image_id}.png"
        cv2.imwrite(str(path / ifile), img)
    annotations.to_csv(path / "train.csv", index=False)
