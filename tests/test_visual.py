import pytest
import matplotlib.pyplot as plt
from detection.mc import make_blob, blob2image
from detection.plot import rectangle, plot


@pytest.fixture
def bbox():
    return [400, 400, 1200, 1200]


def test_mc(bbox, block=False):
    output = blob2image(make_blob(*bbox))
    plt.imshow(output)
    x1, y1, x2, y2 = bbox
    ax = plt.gca()
    ax.add_patch(rectangle(x1, y1, x2, y2))
    plt.show(block=block)


@pytest.mark.parametrize("n_images", [1, 5, 16])
def test_plotting(bbox, n_images, block=False):
    data = [(blob2image(make_blob(*bbox)), [bbox]) for i in range(n_images)]
    plot(*data, block=block)
