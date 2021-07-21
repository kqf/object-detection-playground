import click

from pathlib import Path
from click import Path as cpath

from detection.dataset import read_data
from detection.augmentations import transform
from detection.datasets import DetectionDatasetV3
from detection.model import build_model


@click.command()
@click.option("--fin", type=cpath(exists=True))
@click.option("--logdir", type=str)
def main(fin, logdir):
    fin = Path(fin)
    df = read_data(fin.with_suffix(".csv"))
    # TODO: Fix me, overfit a single example
    df = df.loc[[3] * 60]
    print(df.head())
    train = DetectionDatasetV3(df, fin, transform(train=True))

    model = build_model()
    model.fit(train)


if __name__ == '__main__':
    main()
