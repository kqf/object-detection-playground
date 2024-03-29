import torch
# import matplotlib.pyplot as plt
from detection.metrics import bbox_iou


def to_global(x):
    # x[batch, scale, x_cells, y_cells, 6]
    n_cells = x.shape[2]

    cells = torch.arange(n_cells).to(x.device)

    x_cells = cells.reshape(1, 1, n_cells, 1, 1)
    y_cells = cells.reshape(1, 1, 1, n_cells, 1)

    x[..., 1:2] = (x[..., 1:2] + x_cells) / n_cells
    x[..., 2:3] = (x[..., 2:3] + y_cells) / n_cells
    return x


def nonlin(batch, anchor_boxes):
    predictions = []

    for i, (pred, anchors) in enumerate(zip(batch, anchor_boxes)):
        # [batch, scale, x, y, labels] -> [batch, x, y, scale, labels]

        # Copy don't mutate the original batch
        prediction = pred[..., :6].detach().clone() * 0

        # pred [batch_size, n_anchors, s, s, 5 + nclasses]
        prediction[..., 0] = pred[..., 0]
        prediction[..., 1:3] = torch.sigmoid(pred[..., 1:3])

        aa = anchors[None, :, None, None, :]
        prediction[..., 3:5] = torch.exp(pred[..., 3:5]) * aa

        prediction[..., 5] = torch.argmax(pred[..., 5:], dim=-1)

        final = to_global(prediction)
        predictions.append(final)

    return predictions


def merge_scales(predictions):
    # Flatten along the batch dimension
    flat = []
    for scale in predictions:
        flat.append([x.reshape(-1, x.shape[-1]) for x in scale])

    # The results along the batch dimension
    return [torch.cat(x) for x in zip(*flat)]


def infer(batch, anchor_boxes, top_n, min_iou, threshold):
    predictions = nonlin(batch, anchor_boxes)
    merged = merge_scales(predictions)

    # Run over all samples in the dataset
    supressed = [
        nms(sample, top_n=top_n, min_iou=min_iou, threshold=threshold)
        for sample in merged
    ]

    return supressed


def nms(bboxes, threshold=0.5, min_iou=0.5, top_n=None):
    # Filter out the boxes with low objectness score
    x = bboxes[bboxes[:, 0] > threshold]

    # Ensure everything is calculated per class
    same_object = x[:, None, -1] == x[None, :, -1]

    # Find non-maximum elements
    objectness_per_class = same_object * x[None, :, 0]
    maximum = objectness_per_class.max(-1, keepdim=True).values
    not_maximum = objectness_per_class < maximum

    # IoUs
    ious = bbox_iou(x[:, None, 1:5], x[None, :, 1:5])

    # Putting it all together
    noise = same_object * not_maximum * (ious > min_iou).squeeze(-1)
    suppressed = (~noise).all(0)
    return x[suppressed, 1:]


def no_nms(bboxes, threshold=0.5, min_iou=0.5, top_n=None):

    # plt.hist(bboxes[:, 0])
    # plt.xlabel("objectness")
    # plt.show()
    # plt.savefig("last-objectness.png")

    print(bboxes[:, 0].max())

    # Filter the noisy outputs
    bboxes = bboxes[bboxes[:, 0] > threshold]

    if top_n is not None:
        positive = (-bboxes[:, 0]).argsort()[:top_n]

    # print("The top thresholds are:", bboxes[positive, 0])
    return bboxes[positive, 1:]
