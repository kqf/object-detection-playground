import cv2
import torch
import numpy as np

from torch.utils.data import Dataset


DEFAULT_ANCHORS = [
    torch.tensor([(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)]),
    torch.tensor([(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)]),
    torch.tensor([(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)]),
]

DEFAULT_SCALES = [13, 26, 52]


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

        self.anchors = anchors or torch.cat(DEFAULT_ANCHORS)
        self.scales = scales or DEFAULT_SCALES
        self.iou_threshold = iou_threshold

    def __getitem__(self, index):
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

        # Normalize the boundign boxes
        records['x_min'] = records['x_min'] / image.shape[0]
        records['x_max'] = records['x_max'] / image.shape[0]
        records['y_min'] = records['y_min'] / image.shape[1]
        records['y_max'] = records['y_max'] / image.shape[1]

        x1, y1, x2, y2 = records[['x_min', 'y_min', 'x_max', 'y_max']].values.T
        width, height = x2 - x1, y2 - y1
        boxes = records[['x_min', 'y_min', 'x_max', 'y_max']].values

        area = torch.as_tensor(width * height, dtype=torch.float32)
        labels = torch.tensor(records["class_id"].values, dtype=torch.int64)

        # suppose all instances are not crowd
        iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([index])
        target['area'] = area
        target['iscrowd'] = iscrowd

        if self.transforms:
            sample = {
                'image': image,
                'bboxes': target['boxes'],
                'labels': labels
            }
            transformed = self.transforms(**sample)
            image = transformed['image']
            target['boxes'] = torch.tensor(transformed['bboxes'])

        if target["boxes"].shape[0] == 0:
            # Albumentation cuts the target (class 14, 1x1px in the corner)
            target["boxes"] = torch.tensor([0.0, 0.0, 1.0, 1.0])
            target["area"] = torch.tensor([1.0], dtype=torch.float32)
            target["labels"] = torch.tensor([0], dtype=torch.int64)

        targets = build_targets(
            target["boxes"], target["labels"],
            self.anchors, self.scales, self.iou_threshold)

        return image, targets

    def __len__(self):
        return self.image_ids.shape[0]


def iou(a, b):
    ax, ay = a[..., 0], a[..., 1]
    bx, by = b[..., 0], b[..., 1]

    intersection = torch.min(ax, bx) * torch.min(ay, by)
    union = ax * ay + bx * by - intersection
    return intersection / union


def build_targets(bboxes, labels, anchors, scales, iou_threshold):
    # Three anchors per scale
    targets = [torch.zeros((3, s, s, 6)) for i, s in enumerate(scales)]
    num_anchors_per_scale = 3

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
            i, j = int(s * y), int(s * x)  # which cell
            anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]

            if anchor_taken:
                continue

            if not has_anchor[scale_idx]:
                targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                x_cell, y_cell = s * x - j, s * y - i  # both between [0,1]
                cbox = torch.tensor([x_cell, y_cell, width * s, height * s])
                targets[scale_idx][anchor_on_scale, i, j, 1:5] = cbox
                targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                has_anchor[scale_idx] = True
                continue

            # Ignore the anchor boxes with high iou if collide with other
            if iou_anchors[anchor_idx] > iou_threshold:
                targets[scale_idx][anchor_on_scale, i, j, 0] = -1

    return tuple(targets)
