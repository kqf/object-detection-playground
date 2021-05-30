import cv2
import torch
import numpy as np

from torch.utils.data import Dataset


# Localization, there is a single anchor
# DEFAULT_ANCHORS = [
#     torch.tensor([(0.28, 0.22)]),
# ]

# Ignore the default anchors
DEFAULT_ANCHORS = [
    torch.tensor([(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)]),
    torch.tensor([(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)]),
    torch.tensor([(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)]),
]

DEFAULT_SCALES = [1, 2, 4]


class DetectionDatasetV3(Dataset):

    def __init__(
        self,
        dataframe,
        image_dir,
        anchors=None,
        scales=None,
        iou_threshold=0.5,
        transforms=None,
    ):
        super().__init__()

        self.image_ids = dataframe["image_id"].unique()
        self.df = dataframe
        self.image_dir = image_dir
        self.transforms = transforms

        if anchors is None:
            anchors = DEFAULT_ANCHORS

        # NB: the anchors is the list of tensors:
        # anchors[scale1, scale2, scale3]

        if len(set([len(a) for a in anchors])) != 1:
            raise ValueError("Anchors must be of the same length")

        self.num_anchors_per_scale = len(anchors[0])
        self.anchors = torch.cat(anchors)
        self.iou_threshold = iou_threshold
        self.scales = scales or DEFAULT_SCALES

    def example(self, index):
        image_id = self.image_ids[index]
        records = self.df[(self.df['image_id'] == image_id)]
        records = records.reset_index(drop=True)

        file = f"{self.image_dir}/{image_id}.png"
        image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

        image = np.stack([image, image, image])
        image = image.astype('float32')
        image = image.transpose(1, 2, 0)

        if records.loc[0, "class_id"] == 0:
            records = records.loc[[0], :]

        boxes = records[['x_center', 'y_center', 'width', 'height']].values

        labels = torch.tensor(records["class_id"].values, dtype=torch.int64)

        if self.transforms:
            sample = {
                'image': image,
                'bboxes': boxes,
                'labels': labels
            }
            transformed = self.transforms(**sample)

            image = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['labels']

        _, width, height = image.shape
        targets = build_targets(
            boxes, labels,
            self.anchors,
            self.scales,
            self.iou_threshold,
            num_anchors_per_scale=self.num_anchors_per_scale,
            im_size=width,
        )

        return image, boxes, targets

    def __len__(self):
        return self.image_ids.shape[0]

    def __getitem__(self, index):
        image, boxes, targets = self.example(index)
        return image, targets


def iou(a, b):
    ax, ay = a[..., 0], a[..., 1]
    bx, by = b[..., 0], b[..., 1]

    intersection = torch.min(ax, bx) * torch.min(ay, by)
    union = ax * ay + bx * by - intersection
    return intersection / union


def build_targets(bboxes, labels, anchors,
                  raw_scales, iou_threshold, num_anchors_per_scale, im_size):
    # scale = upscaling factor s times darknet output (image_size // 32)
    scales = [im_size // 32 * s for s in raw_scales]

    # Three anchors per scale
    targets = [torch.zeros((num_anchors_per_scale, s, s, 6))
               for i, s in enumerate(scales)]

    for box, class_label in zip(bboxes, labels):
        if np.isnan(box).any():
            continue

        iou_anchors = iou(torch.tensor(box[2:4]), anchors)
        anchor_indices = iou_anchors.argsort(descending=True, dim=0)
        x, y, width, height = box
        has_anchor = [False, False, False]

        for anchor_idx in anchor_indices:
            scale_idx = int(anchor_idx // num_anchors_per_scale)
            anchor_on_scale = int(anchor_idx % num_anchors_per_scale)
            s = scales[scale_idx]
            i, j = int(s * x), int(s * y)  # which cell
            anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]

            if anchor_taken:
                continue

            if not has_anchor[scale_idx]:
                targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                x_cell, y_cell = s * x - i, s * y - j  # both between [0, 1]
                cbox = torch.tensor([x_cell, y_cell, width * s, height * s])
                targets[scale_idx][anchor_on_scale, i, j, 1:5] = cbox
                targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                has_anchor[scale_idx] = True
                continue

            # Ignore the anchor boxes with high iou if collide with other
            if iou_anchors[anchor_idx] > iou_threshold:
                targets[scale_idx][anchor_on_scale, i, j, 0] = -1

    return tuple(targets)
