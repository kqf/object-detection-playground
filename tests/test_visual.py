import matplotlib.pyplot as plt
from detection.mc import make_blob, blob2image


def test_mc():
    output = blob2image(make_blob())
    plt.imshow(output)
    plt.show()
