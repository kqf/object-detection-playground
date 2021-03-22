import pytest

from detection.data import read_data
from detection.datasets.v3 import DetectionDatasetV3
from detection.augmentations import transform
from detection.model import build_model


@pytest.mark.skip("Fix model tensor mismatched shapes")
def test_dummy(fake_dataset):
    df = read_data(fake_dataset / "train.csv")
    print(df.head())
    train = DetectionDatasetV3(df, fake_dataset, transforms=transform())

    for x in train:
        print(x)

    model = build_model()
    model.fit(train)
