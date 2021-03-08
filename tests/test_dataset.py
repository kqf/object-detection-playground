import pandas as pd
from detection.dataset import DetectionDatasetV3


def test_dataset(fake_dataset):
    df = pd.read_csv(fake_dataset / "train.csv")
    dataset = DetectionDatasetV3(df, fake_dataset)

    for image, targets in dataset:
        assert len(image.shape) == 3, "There are only 3 dimensions"
        assert image.shape[-1] == 3, "There are only 3 channels"