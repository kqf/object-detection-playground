import torch
from detection.metrics import bbox_iou


def to_global(x, scale):
    cells = torch.arange(scale).to(x.device)

    x_cells = cells.reshape(1, scale, 1, 1, 1)
    y_cells = cells.reshape(1, 1, scale, 1, 1)

    x[..., 0:1] = (x[..., 0:1] + x_cells) / scale
    x[..., 1:2] = (x[..., 1:2] + y_cells) / scale
    x[..., 2:5] = x[..., 2:5] / scale
    return x


def nonlin(batch, anchor_boxes):
    predictions = []

    for i, (pred, anchors) in enumerate(zip(batch, anchor_boxes)):
        # [batch, scale, x, y, labels] -> [batch, x, y, scale, labels]
        pred = pred.permute(0, 2, 3, 4, 1)

        # Copy don't mutate the original batch
        prediction = pred[..., :6].detach().clone() * 0

        # pred [batch_size, n_anchors, s, s, 5 + nclasses]
        scale = pred.shape[2]

        prediction[..., 0] = torch.sigmoid(pred[..., 0])
        prediction[..., 1:3] = torch.sigmoid(pred[..., 1:3])
        prediction[..., 3:5] = torch.exp(pred[..., 3:5]) * anchors * scale
        prediction[..., 5] = torch.argmax(pred[..., 5:], dim=-1)

        final = to_global(prediction, scale=scale)
        predictions.append(final)

    return predictions


def infer(batch, anchor_boxes):
    predictions = nonlin(batch, anchor_boxes)
    merged = merge_scales(predictions)

    # Run over all samples in the dataset
    supressed = [no_nms(sample) for sample in merged]
    return supressed


def merge_scales(predictions):
    # Flatten along the batch dimension
    flat = []
    for scale in predictions:
        flat.append([x.reshape(-1, x.shape[-1]) for x in scale])

    # The results along the batch dimension
    return [torch.cat(x) for x in zip(*flat)]


def iou(box1, box2):
    return bbox_iou(torch.tensor(box1[2:]), torch.tensor(box2[2:]))


def non_max_suppression(bboxes, iou_threshold, threshold):
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


def nms(pred, min_iou=0.5):
    same_object = pred[:, None, 0] == pred[None, :, 0]
    objectness_per_class = same_object * pred[None, :, -1]
    maximum = objectness_per_class.max(-1, keepdim=True).values
    not_maximum = objectness_per_class < maximum
    ious = bbox_iou(pred[:, None, 1:5], pred[None, :, 1:5])
    result = same_object * not_maximum * (ious > min_iou).squeeze(-1)
    return ~result


def no_nms(pred, threshold=0.0):
    positive = pred[:, 1] > threshold
    return pred[positive, 1:]
