import pytest
import matplotlib.pyplot as plt
from detection.mc import make_blob, blob2image


@pytest.fixture
def bbox():
    return [400, 400, 600, 800]


def rectangle(x1, y1, x2, y2):
    w, h = x2 - x1, y2 - y1
    return plt.Rectangle((x1, y1), w, h, color='r', fill=False)


def test_mc(bbox):
    output = blob2image(make_blob(*bbox))
    plt.imshow(output)
    x1, y1, x2, y2 = bbox
    ax = plt.gca()
    ax.add_patch(rectangle(x1, y1, x2, y2))
    plt.show()
