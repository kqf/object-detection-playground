import torch


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


class CombinedLoss(torch.nn.Module):
    def __init__(self, anchors):
        super().__init__()
        self.anchors = anchors
        self.mse = torch.nn.MSELoss()
        self.bce = torch.nn.BCEWithLogitsLoss(reduction="sum")
        self.objectness = torch.nn.BCEWithLogitsLoss()
        self.entropy = torch.nn.CrossEntropyLoss()
        self.sigmoid = torch.nn.Sigmoid()

        self.classification = 1
        self.noobj = 1
        self.obj = 1
        self.box = 1

    def forward(self, predictions, target):
        loss = torch.tensor(0).float()

        # Calculate the loss at each scale
        for pred, y, anchors in zip(predictions, target, self.anchors):
            loss += self._forward(pred, y, anchors)

        return loss

    def _forward(self, predictions, target, anchors):
        # Check where obj and noobj (we ignore if target == -1)
        obj = target[..., 0] == 1  # in paper this is Iobj_i
        noobj = target[..., 0] == 0  # in paper this is Inoobj_i

        predictions = predictions.transpose(1, -1)

        no_obj_denominator = noobj.shape[0] + (noobj.shape[0] < 1)
        no_detection = self.bce(
            (predictions[..., 0:1][noobj]),
            (target[..., 0:1][noobj]),
        ) / no_obj_denominator

        anchors = anchors.reshape(1, 3, 1, 1, 2)

        # x,y coordinates
        predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3])
        box_preds = torch.cat([
            predictions[..., 1:3],
            torch.exp(predictions[..., 3:5]) * anchors
        ], dim=-1)

        ious = bbox_iou(box_preds[obj], target[..., 1:5][obj]).detach()
        detection = self.objectness(
            (predictions[..., 0:1][obj]),
            (ious * target[..., 0:1][obj])
        )
        # width, height coordinate
        tboxes = torch.cat([
            target[..., 1:3],
            torch.log((1e-16 + target[..., 3:5] / anchors)),
        ], dim=-1)

        box = self.mse(predictions[..., 1:5][obj], tboxes[obj])

        classification = self.entropy(
            (predictions[..., 5:][obj]), (target[..., 5][obj].long()),
        )

        return (
            self.box * box
            + self.obj * detection
            + self.noobj * no_detection
            + self.classification * classification
        )  # noqa
