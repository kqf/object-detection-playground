import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
from math import sqrt, ceil


def normalize(x):
    std = torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1)
    mu = torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1)
    return x * std + mu


def absolute_bbox(bbox, image_w, image_h):
    x_center, y_center, w, h, *color = bbox
    x1 = (x_center - w / 2) * image_w
    x2 = (x_center + w / 2) * image_w
    y1 = (y_center - h / 2) * image_h
    y2 = (y_center + h / 2) * image_h
    return [x1, y1, x2, y2] + color


def tensor2img(t, padding=0, normalize=True):
    # return t * std + mu if t.shape[0] > 1 else t
    img = to_pil_image(normalize(t) if normalize else t)
    w, h = img.size
    return np.array(img.crop((padding, padding, w - padding, h - padding)))


def plot(*imgs, block=True, normalize=False, convert_bbox=False, ofile=None):
    n_plots = ceil(sqrt(len(imgs)))
    fig, axes = plt.subplots(n_plots, n_plots, figsize=(6, 6))

    for i, (image, bboxes, targets) in enumerate(imgs):
        plt.subplot(n_plots, n_plots, i + 1)
        try:
            plt.imshow(image)
        except TypeError:
            plt.imshow(tensor2img(image, normalize=normalize))
        ax = plt.gca()

        for bbox in targets:
            if convert_bbox:
                bbox = absolute_bbox(bbox, image.shape[1], image.shape[2])
            ax.add_patch(rectangle(*bbox, c=9999, lw=2, ls=':'))

        for bbox in bboxes:
            if convert_bbox:
                bbox = absolute_bbox(bbox, image.shape[1], image.shape[2])
            ax.add_patch(rectangle(*bbox))

    plt.tight_layout()
    if ofile is not None:
        plt.savefig(ofile)
    plt.show(block=block)
    return axes


def batches(dataset, batch_size):
    batch = []
    for pair in dataset:
        if len(batch) < batch_size:
            compare([pair[0]], [pair[1]])
            batch.append(pair)
            continue
        yield list(zip(*batch))
        batch = []


def compare(image, bbox, normalize=False, convert_bbox=False):
    plt.imshow(tensor2img(image, normalize=normalize))
    ax = plt.gca()
    if convert_bbox:
        bbox = absolute_bbox(bbox, image.shape[1], image.shape[2])
    ax.add_patch(rectangle(*bbox))
    plt.show()


def check_boxes(boxes, convert_bbox=True, wsize=640, hsize=640):
    plt.imshow(np.zeros((wsize, hsize)), vmin=0, vmax=1, cmap="binary")
    ax = plt.gca()
    for c, box in enumerate(boxes):
        if convert_bbox:
            box = absolute_bbox(box, wsize, hsize)
        ax.add_patch(rectangle(*box, c + 3, fill=True, alpha=0.1))
    plt.show()


def glance(dataset, batch_size, pfunc=plot):
    for batch in batches(dataset, batch_size):
        pfunc(*batch)


def rectangle(x1, y1, x2, y2, c=0, lw=1, ls='-', fill=False, alpha=0.5):
    w, h = x2 - x1, y2 - y1
    color = plt.cm.RdYlBu(1. / int(c + 1))
    return plt.Rectangle((x1, y1), w, h, color=color, alpha=alpha,
                         fill=fill, lw=lw, ls=ls)
