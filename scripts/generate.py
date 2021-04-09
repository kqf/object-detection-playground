import click
from pathlib import Path
from detection.mc import annotations, generate_to_directory


@click.command()
@click.option("--fout", type=click.Path(exists=False))
def main(fout):
    df = annotations()
    path = Path(fout)
    path.mkdir(parents=True, exist_ok=True)
    generate_to_directory(df, path)
