# import torch

from detection.data import read_data
from detection.datasets.v3 import DetectionDatasetV3
from detection.augmentations import transform
# from detection.plot import plot


def test_targets(fake_dataset, fixed_seed):
    df = read_data(fake_dataset / "train.csv")
    train = DetectionDatasetV3(df, fake_dataset, transforms=transform())

    for i in range(len(train)):
        image, example, targets = train.example(i)
        # scale1, scale2, scale3 = target
