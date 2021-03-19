from detection.data import read_data
from detection.dataset import DetectionDatasetV3
from detection.model import build_model


def test_dummy(fake_dataset):
    df = read_data(fake_dataset / "train.csv")
    print(df.head())
    train = DetectionDatasetV3(df, fake_dataset)

    model = build_model()
    model.fit(train)
