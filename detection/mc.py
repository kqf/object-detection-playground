import numpy as np


def make_blob(
    h=2000, w=2000,
    x_min=50, y_min=50,
    x_max=90, y_max=90,
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

    mask = dists <= 1
    return mask.sum(axis=-1).astype(np.uint8)


def blob2image(blob, channels=3, epsilon=0.1):
    h, w = blob.shape

    extended = blob[..., None]

    # Add a small term to add noise to the empty regions
    noise = np.random.poisson(extended + epsilon, size=(h, w, channels))

    # Convet to image scale
    return (extended + noise * 255).astype(np.uint8)
