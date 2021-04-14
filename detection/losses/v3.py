import torch

objectness = ..., slice(0, 1)
bbox_xy = ..., slice(1, 3)
bbox_wh = ..., slice(3, 5)
bbox_all = ..., slice(1, 5)


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

        self.lcls = 1
        self.det = 1
        self.box = 1
        self.obj = 1

        pos_weight = torch.tensor([self.obj])
        self.objectness = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.classification = torch.nn.CrossEntropyLoss()
        self.bbox = torch.nn.MSELoss()

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
        pred[bbox_xy] = torch.nn.functional.sigmoid(pred[bbox_xy])
        box_preds = torch.cat([
            pred[bbox_xy],
            torch.exp(pred[bbox_wh]) * anchors
        ], dim=-1)

        ious = bbox_iou(box_preds, target[bbox_all]).detach()
        det = self.objectness(
            pred[objectness],
            ious * target[objectness]
        )
        tboxes = torch.cat([
            target[bbox_xy],
            torch.log((1e-16 + target[bbox_wh] / anchors)),
        ], dim=-1)

        obj = target[..., 0] == 1  # in paper this is Iobj_i
        box = self.bbox(pred[bbox_all][obj], tboxes[obj])

        lcls = self.classification(
            pred[..., 5:][obj],
            target[..., 5][obj].long(),
        )

        return self.det * det + self.box * box + self.lcls * lcls
