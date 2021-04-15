import torch
from detection.metrics import bbox_iou


def iou(box1, box2):
    return bbox_iou(torch.tensor(box1[2:]), torch.tensor(box2[2:]))

def non_max_suppression(bboxes, iou_threshold, threshold):

    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0] or iou(chosen_box, box) < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms
