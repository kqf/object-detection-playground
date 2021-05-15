import torch
from detection.metrics import bbox_iou

objectness = ..., slice(0, 1)
bbox_xy = ..., slice(1, 3)
bbox_wh = ..., slice(3, 5)
bbox_all = ..., slice(1, 5)


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
        self.regression = torch.nn.MSELoss()

    def forward(self, pred, target):
        loss = torch.tensor(0).float()

        # Calculate the loss at each scale
        for pred, y, anchors in zip(pred, target, self.anchors):
            loss = loss.to(y.device)
            anchors = anchors.to(y.device)
            loss += self._forward(pred, y, anchors)

        return loss

    def _forward(self, pred, target, anchors):
        # # [batch, scale, x, y, labels] -> [batch, x, y, scale, labels]
        pred = pred.permute(0, 4, 2, 3, 1)

        # # [batch, x, y, scale, labels] -> [batch * x * y, scale, labels]
        # pred = pred.reshape(-1, pred.shape[-2], pred.shape[-1])

        # # [batch, scale, x, y, labels] -> [batch, x, y, scale, labels]
        # target = target.permute(0, 2, 3, 1, 4)

        # # [batch, x, y, scale, labels] -> [batch * x * y, scale, labels]
        # target = target.reshape(-1, target.shape[-2], target.shape[-1])

        # [scale, 2] -> [1, scale, 2]
        anchors = anchors.reshape(1, 3, 1, 1, 2)

        noobj = target[..., 0:1] == 0  # in paper this is Iobj_i
        nodet = self.objectness(
            pred[objectness][noobj],
            (target[objectness][noobj])
        )

        # x,y coordinates
        box_preds = torch.cat([
            torch.nn.functional.sigmoid(pred[bbox_xy]),
            torch.exp(pred[bbox_wh]) * anchors
        ], dim=-1)

        obj = target[..., 0] == 1  # in paper this is Iobj_i
        ious = bbox_iou(box_preds, target[bbox_all]).detach()
        det = self.regression(
            torch.sigmoid(pred[objectness][obj]),
            ious[obj] * target[objectness][obj]
        )
        tboxes = torch.cat([
            target[bbox_xy],
            torch.log((1e-16 + target[bbox_wh] / anchors)),
        ], dim=-1)

        pred_boxes = torch.cat([
            torch.nn.functional.sigmoid(pred[bbox_xy]),
            pred[bbox_wh]
        ], dim=-1)

        box = self.regression(pred_boxes[obj], tboxes[obj])

        lcls = self.classification(
            pred[..., 5:][obj],
            target[..., 5][obj].long(),
        )

        loss = self.det * det + self.box * box + self.lcls * lcls + nodet
        return loss
