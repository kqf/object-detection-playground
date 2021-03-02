import click
import cv2
import numpy as np
import pandas as pd
import tqdm
import pydicom
from pathlib import Path

from click import Path as cpath
from pydicom.pixel_data_handlers.util import apply_voi_lut


def read_scan(path):
    dicom = pydicom.read_file(path)
    scan = apply_voi_lut(dicom.pixel_array, dicom)

    # depending on this value, X-ray may look inverted - fix that:
    if dicom.PhotometricInterpretation == "MONOCHROME1":
        scan = np.amax(scan) - scan

    minimum = np.min(scan)
    scaled = (scan - minimum) / (np.max(scan) - minimum)
    return (scaled * 255).astype(np.uint8)


@click.command()
@click.option("--codes", type=cpath(exists=True), default="data/train.csv")
@click.option("--fin", type=cpath(exists=True))
@click.option("--fout", type=cpath(exists=False))
def main(codes, fin, fout):
    # Combine masks into one
    df = pd.read_csv(codes)
    print(df.head())

    fout = Path(fout)
    fout.mkdir()
    for image_id in tqdm.tqdm(df["image_id"].values):
        scan = read_scan(path=(Path(fin) / image_id).with_suffix(".dicom"))
        cv2.imwrite(str((fout / image_id).with_suffix(".png")), scan)


if __name__ == "__main__":
    main()
