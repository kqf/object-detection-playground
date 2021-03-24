import pytest
import matplotlib.pyplot as plt
from detection.mc import make_blob, blob2image
from detection.plot import rectangle, plot


@pytest.fixture
def bbox():
    return [400, 400, 1200, 1200]


def test_mc(bbox):
    output = blob2image(make_blob(*bbox))
    plt.imshow(output)
    x1, y1, x2, y2 = bbox
    ax = plt.gca()
    ax.add_patch(rectangle(x1, y1, x2, y2))
    plt.show()


def test_plotting(bbox):
    data = [(blob2image(make_blob(*bbox)), bbox) for i in range(16)]
    plot(*data)
