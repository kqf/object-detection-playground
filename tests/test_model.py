
from detection.data import read_data
from detection.datasets.v3 import DetectionDatasetV3
from detection.augmentations import transform
from detection.model import build_model

from detection.plot import plot


def test_dummy(fake_dataset):
    df = read_data(fake_dataset / "train.csv")
    print(df.head())
    train = DetectionDatasetV3(df, fake_dataset, transforms=transform())

    model = build_model(max_epochs=2)
    model.fit(train)

    # TODO: Fix me
    preds = model.predict(train)
    plot([train[0][0], preds[0][:, :4]], ofile="test-dummy.png")
