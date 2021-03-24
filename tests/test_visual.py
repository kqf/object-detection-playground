import pytest
import matplotlib.pyplot as plt
from detection.mc import make_blob, blob2image


@pytest.fixture
def bbox():
    return [400, 400, 600, 800]


def test_mc(bbox):
    output = blob2image(make_blob(*bbox))
    plt.imshow(output)
    plt.show()
