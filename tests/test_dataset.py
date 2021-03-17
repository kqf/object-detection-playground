import pandas as pd
from detection.dataset import DetectionDatasetV3


def test_dataset(fake_dataset):
    df = pd.read_csv(fake_dataset / "train.csv")
    dataset = DetectionDatasetV3(df, fake_dataset)

    for image, (s1, s2, s3) in dataset:
        assert len(image.shape) == 3, "There are only 3 dimensions"
        assert image.shape[-1] == 3, "There are only 3 channels"

        assert s1.shape == (16, 3, 8, 8, 85)
        assert s2.shape == (16, 3, 16, 16, 85)
        assert s3.shape == (16, 3, 32, 32, 85)
