import torch
from collections import Counter


def bbox_iou(preds, labels):
    a_x1 = preds[..., 0:1] - preds[..., 2:3] / 2
    a_y1 = preds[..., 1:2] - preds[..., 3:4] / 2
    a_x2 = preds[..., 0:1] + preds[..., 2:3] / 2
    a_y2 = preds[..., 1:2] + preds[..., 3:4] / 2
    b_x1 = labels[..., 0:1] - labels[..., 2:3] / 2
    b_y1 = labels[..., 1:2] - labels[..., 3:4] / 2
    b_x2 = labels[..., 0:1] + labels[..., 2:3] / 2
    b_y2 = labels[..., 1:2] + labels[..., 3:4] / 2

    x1 = torch.max(a_x1, b_x1)
    y1 = torch.max(a_y1, b_y1)
    x2 = torch.min(a_x2, b_x2)
    y2 = torch.min(a_y2, b_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    a_area = abs((a_x2 - a_x1) * (a_y2 - a_y1))
    b_area = abs((b_x2 - b_x1) * (b_y2 - b_y1))

    return intersection / (a_area + b_area - intersection + 1e-6)


def positive_rate_(x, y, iou_threshold):
    overlaps = bbox_iou(x[None], y[:, None]).squeeze(-1)
    same_class = x[None, :, -1] == y[:, None, -1]
    return (overlaps * same_class).sum(0) > 0


def mAP(pred, true_boxes, iou_threshold=0.5, n_classes=20, eps=1e-6):
    # list storing all AP for respective classes
    average_precisions = []

    for c in range(n_classes):
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        for detection in pred:
            if detection[-1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[-1] == c:
                ground_truths.append(true_box)

        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        tp = torch.zeros((len(detections)))
        fp = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            pr = positive_rate_(detection.unsqueeze(0),
                                torch.cat(ground_truths).unsqueeze(0),
                                iou_threshold)
            tp[detection_idx], fp[detection_idx] = pr.sum(), (~pr).sum()

        tp_cumsum = torch.cumsum(tp, dim=0)
        fp_cumsum = torch.cumsum(fp, dim=0)

        recalls = tp_cumsum / (total_true_bboxes + eps)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + eps)
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)
