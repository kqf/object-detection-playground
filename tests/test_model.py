import pytest
from detection.data import DetectionDataset, read_data
from detection.model import build_model


@pytest.mark.skip("Fix the inputs mismatch")
def test_dummy(fake_dataset):
    df = read_data(fake_dataset / "train.csv")
    print(df.head())
    train = DetectionDataset(df, fake_dataset)

    model = build_model()
    model.fit(train)
