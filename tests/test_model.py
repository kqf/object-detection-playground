import pandas as pd
from detection.data import DetectionDataset
from detection.model import build_model


def test_dummy(fake_dataset):
    df = pd.read_csv(fake_dataset / "train.csv")
    print(df.head())
    train = DetectionDataset(df, fake_dataset)

    model = build_model()
    model.fit(train)
