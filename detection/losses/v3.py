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
        self.nodet = 1

        # pos_weight = torch.tensor([self.obj])
        self.noobjloss = torch.nn.MSELoss()
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
        # pred [batch, scale, x, y, labels]
        # [scale, 2] -> [1, scale, 1, 1, 2]
        anchors = anchors.reshape(1, -1, 1, 1, 2)

        noobj = target[..., 0:1] != 1  # in paper this is Iobj_i
        nodet = self.noobjloss(
            pred[objectness][noobj],
            target[objectness][noobj]
        )

        # x,y coordinates
        box_preds = torch.cat([
            torch.sigmoid(pred[bbox_xy]),
            torch.exp(pred[bbox_wh]) * anchors
        ], dim=-1)

        obj = target[..., 0] == 1  # in paper this is Iobj_i
        ious = bbox_iou(box_preds, target[bbox_all]).detach()
        det = self.regression(
            torch.sigmoid(pred[objectness][obj]),
            target[objectness][obj] * ious[obj],
        )

        coord = self.regression(
            torch.sigmoid(pred[bbox_xy][obj]),
            target[bbox_xy][obj],
        )

        box = self.regression(
            pred[bbox_wh][obj],
            torch.log(1e-16 + target[bbox_wh] / anchors)[obj],
        )

        """
        lcls = self.classification(
            pred[..., 5:][obj],
            target[..., 5][obj].long(),
        )
        """

        loss = \
            self.det * det + \
            self.box * box + \
            self.box * coord + \
            self.nodet * nodet
        # self.lcls * lcls + \

        # print(box.item(), coord.item(), det.item())

        # print(
        #     "detection ", det.item(),
        #     "box ", box.item(),
        #     "coord ", coord.item(),
        #     "cls ", lcls.item(),
        #     "nodet ", nodet.item(),
        # )

        return loss
