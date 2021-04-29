# import torch

from detection.data import read_data
from detection.datasets.v3 import DetectionDatasetV3
from detection.augmentations import transform
from detection.model import build_model

from detection.plot import plot


def test_dummy(fake_dataset, fixed_seed):
    df = read_data(fake_dataset / "train.csv")
    print(df.head())
    train = DetectionDatasetV3(df, fake_dataset, transforms=transform())

    model = build_model(max_epochs=10)
    model.fit(train)
    preds = model.predict(train)
    first_image_pred = preds[0][:, :4]
    first_image = train[0][0]

    # first_image = torch.ones((3, 460, 460))
    # for first_image in train:
    #     first_image_pred = torch.zeros(10, 4)
    #     first_image = first_image[0]

    #     # first_image = train[0][0] * 0 + 1
    #     first_image_pred = torch.ones((10, 4))
    #     first_image_pred[:, 0] = 0.4
    #     first_image_pred[:, 1] = 0.4
    #     first_image_pred[:, 2] = 0.2
    #     first_image_pred[:, 3] = 0.2

    plot((first_image, first_image_pred),
         convert_bbox=True, ofile='dummy-test.png')
