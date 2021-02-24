import click

from pathlib import Path
from click import Path as cpath

from models.data import read_data, DetectionDataset
from models.augmentations import transform
from models.model import build_model


@click.command()
@click.option("--fin", type=cpath(exists=True))
@click.option("--logdir", type=str)
def main(fin, logdir):
    fin = Path(fin)
    df = read_data(fin.with_suffix(".csv"))
    print(df.head())
    train = DetectionDataset(df, fin, transform(train=True))

    model = build_model()
    model.fit(train)


if __name__ == '__main__':
    main()
