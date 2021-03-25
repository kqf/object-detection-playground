import click
import pandas as pd
import pathlib

from detection.datasets.v3 import DetectionDatasetV3
from detection.augmentations import transform
from detection.plot import plot


@click.command()
@click.option("--datapath")
def main(datapath):
    path = pathlib.Path(datapath)
    df = pd.read_csv(path / "train.csv")
    dataset = DetectionDatasetV3(df, path, transforms=transform(train=True))

    for image, (s1, s2, s3) in dataset:
        plot(image, s1)
        plot(image, s2)
        plot(image, s3)
