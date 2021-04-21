import torch


def to_global(x, scale):
    cells = (
        torch.arange(scale)
        .repeat(x.shape[0], 3, scale, 1)
        .unsqueeze(-1)
        .to(x.device)
    ).permute(0, 2, 3, 1, 4)

    x[..., 0:1] = (x[..., 0:1] + cells) / scale
    x[..., 1:2] = (x[..., 1:2] + cells.transpose(2, 1)) / scale
    x[..., 2:4] = x[..., 2:4] / scale
    return x


def infer(batch, anchor_boxes):
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


def merge_scales(predictions):
    # Flatten along the batch dimension
    flat = []
    for scale in predictions:
        flat.append([x.reshape(-1, x.shape[-1]) for x in scale])

    # The results along the batch dimension
    return [torch.cat(x) for x in zip(*flat)]
