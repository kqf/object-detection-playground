import click

from pathlib import Path
from click import Path as cpath

from models.data import read_data, Dataset
from models.augmentations import transform


@click.command()
@click.option("--fin", type=cpath(exists=True))
@click.option("--logdir", type=str)
def main(fin, logdir):
    fin = Path(fin)
    df = read_data(fin.with_suffix(".csv"))
    print(df.head())
    train = Dataset(df, fin, transform())
    for x in train():
        print(x)


if __name__ == '__main__':
    main()
