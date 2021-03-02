import cv2
import pytest
import tempfile

from pathlib import Path

from models.mc import make_blob, blob2image


@pytest.fixture
def size():
    return 256


@pytest.fixture
def fake_dataset(size=256, nfiles=5):
    with tempfile.TemporaryDirectory() as dirname:
        path = Path(dirname)
        for image_id in range(nfiles):
            img = blob2image(make_blob())
            ifile = f"{image_id}.png"
            cv2.imwrite(str(path / ifile), img)

        yield path
