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

    def forward(self, pred, target):
        loss = torch.tensor(0).float()

        # Calculate the loss at each scale
        for pred, y, anchors in zip(pred, target, self.anchors):
            loss = loss.to(y.device)
            anchors = anchors.to(y.device)
            loss += self._forward(pred, y, anchors)

        return loss

    def _forward(self, pred, target, anchors):
        # [batch, scale, x, y, labels] -> [batch, x, y, scale, labels]
        pred = pred.permute(0, 2, 3, 4, 1)

        # [batch, x, y, scale, labels] -> [batch * x * y, scale, labels]
        pred = pred.reshape(-1, pred.shape[-2], pred.shape[-1])

        # [batch, scale, x, y, labels] -> [batch, x, y, scale, labels]
        target = target.permute(0, 2, 3, 1, 4)

        # [batch, x, y, scale, labels] -> [batch * x * y, scale, labels]
        target = target.reshape(-1, target.shape[-2], target.shape[-1])

        # [scale, 2] -> [1, scale, 2]
        anchors = anchors.reshape(1, 3, 2)

        # x,y coordinates
        pred[..., 1:3] = self.sigmoid(pred[..., 1:3])
        box_preds = torch.cat([
            pred[..., 1:3],
            torch.exp(pred[..., 3:5]) * anchors
        ], dim=-1)

        ious = bbox_iou(box_preds, target[..., 1:5]).detach()
        detection = self.objectness(
            pred[..., 0:1],
            ious * target[..., 0:1]
        )
        tboxes = torch.cat([
            target[..., 1:3],
            torch.log((1e-16 + target[..., 3:5] / anchors)),
        ], dim=-1)

        obj = target[..., 0] == 1  # in paper this is Iobj_i
        box = self.mse(pred[..., 1:5][obj], tboxes[obj])

        classification = self.entropy(
            pred[..., 5:][obj],
            target[..., 5][obj].long(),
        )

        return (
            self.box * box
            + self.obj * detection
            + self.classification * classification
        )  # noqa
