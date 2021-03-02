import click
import pandas as pd

from click import Path as cpath


@click.command()
@click.option("--codes", type=cpath(exists=True), default="data/train.csv")
@click.option("--fin", type=cpath(exists=True))
@click.option("--fout", type=cpath(exists=False))
def main(codes, fin, fout):
    # Combine masks into one
    df = pd.read_csv(codes)
    print(df.head())


if __name__ == "__main__":
    main()
