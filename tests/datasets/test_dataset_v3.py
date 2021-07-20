import torch
import pytest
import pandas as pd
from detection.dataset import DetectionDataset
from detection.augmentations import transform
from detection.plot import plot


@pytest.mark.parametrize("transforms", [
    transform(train=False),
    transform(train=True),
])
@pytest.mark.parametrize("anchors", [
    None,
    [torch.tensor([(0.28, 0.22)])],
])
def test_dataset(fake_dataset, anchors, transforms):
    df = pd.read_csv(fake_dataset / "train.csv")
    dataset = DetectionDataset(
        df, fake_dataset, anchors=anchors, transforms=transforms)

    for image, (s1, s2, s3) in dataset:
        assert len(image.shape) == 3, "There are only 3 dimensions"
        assert image.shape[0] == 3, f"There are only 3 channels {image.shape}"

        assert s1.shape == (dataset.num_anchors_per_scale, 13, 13, 6)
        assert s2.shape == (dataset.num_anchors_per_scale, 26, 26, 6)
        assert s3.shape == (dataset.num_anchors_per_scale, 52, 52, 6)


def test_augmentations(fake_dataset, block=False):
    df = pd.read_csv(fake_dataset / "train.csv")
    dataset = DetectionDataset(
        df, fake_dataset,
        transforms=transform(train=True),
    )

    for i in range(len(dataset)):
        image, bboxes, _ = dataset.example(i)
        plot([image, [], bboxes], block=block, convert_bbox=True)
