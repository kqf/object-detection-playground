import torch

from detection.data import read_data
from detection.datasets.v3 import DetectionDatasetV3
from detection.augmentations import transform
from detection.inference import to_global
# from detection.plot import plot


def test_targets(fake_dataset, fixed_seed):
    df = read_data(fake_dataset / "train.csv")
    train = DetectionDatasetV3(df, fake_dataset, transforms=transform())

    for i in range(len(train)):
        image, bbox, targets = train.example(i)
        for scale in targets:
            # No need to apply the nonlinearities
            # Convert to the global refenrence frame
            pred = to_global(scale.permute(1, 2, 0, 3), scale.shape[2])

            # "Apply" the NMS
            final_output = pred[pred[..., 0] == 1]

            torch.testing.assert_allclose(final_output[:, 1:5], bbox[0])
            # plot((torch.ones(3, 2000, 2000), final_output[:, 1:5], bbox),
            #      convert_bbox=True)
