import torch

from detection.data import read_data
from detection.datasets.v3 import DetectionDatasetV3
from detection.augmentations import transform
from detection.inference import to_global
# from detection.plot import plot


def test_targets(fake_dataset, fixed_seed):
    df = read_data(fake_dataset / "train.csv")
    train = DetectionDatasetV3(df, fake_dataset, transforms=transform())

    for i in range(len(train))[:1]:
        image, bbox, targets = train.example(i)
        for scale in targets:
            # No need to apply the nonlinearities
            # Convert to the global refenrence frame
            assert torch.all(scale[..., 1:3] >= 0)
            assert torch.all(scale[..., 1:3] <= 1)

            pred = to_global(scale.permute(1, 2, 0, 3))

            # "Apply" the NMS
            final_output = pred[pred[..., 0] == 1]

            target = final_output[:, 1:5].flatten()
            torch.testing.assert_allclose(target, bbox[0])

            # Compare visually
            # img = torch.ones(3, 2000, 2000)
            # plot((img, target.reshape(1, -1), bbox), convert_bbox=True)
