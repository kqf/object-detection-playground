import pytest
import pandas as pd
from detection.datasets.v3 import DetectionDatasetV3
from detection.augmentations import transform
from detection.plot import plot


@pytest.mark.parametrize("transforms", [
    transform(train=False),
    transform(train=True),
])
def test_dataset(fake_dataset, transforms):
    df = pd.read_csv(fake_dataset / "train.csv")
    dataset = DetectionDatasetV3(df, fake_dataset, transforms=transforms)

    for image, (s1, s2, s3) in dataset:
        assert len(image.shape) == 3, "There are only 3 dimensions"
        assert image.shape[0] == 3, f"There are only 3 channels {image.shape}"

        assert s1.shape == (3, 13, 13, 6)
        assert s2.shape == (3, 26, 26, 6)
        assert s3.shape == (3, 52, 52, 6)


def test_augmentations(fake_dataset):
    df = pd.read_csv(fake_dataset / "train.csv")
    dataset = DetectionDatasetV3(
        df, fake_dataset,
        transforms=transform(train=True),
        no_anchors=True,
    )

    for image, anchors in dataset:
        plot([image, anchors])
