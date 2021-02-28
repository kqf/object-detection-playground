# The code is taken from:
# https://github.com/ultralytics/yolov5/blob/master/utils/loss.py
import torch


def smooth_bce(eps=0.1):
    """
    https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    """
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


def bbox_iou(box1, box2, x1y1x2y2=True, eps=1e-9):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    return iou


class ComputeLoss:
    # Compute losses
    def __init__(
        self,
        hyp,
        nc=2,  # number of classes
        nl=3,  # number of detection layers
        anchors=(2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
        stride=[16],
        gr=1.0,  # iou loss ratio (obj_loss = 1.0 or iou)
        autobalance=False,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        self.autobalance = autobalance
        self.hyp = hyp
        self.nl = nl
        self.nc = nc
        self.anchors = anchors
        self.stride = stride
        self.gr = gr
        self.device = device
        # Define criteria

        # TODO:
        # hyp['box'] *= 3. / nl  # scale to layers
        # hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
        # scale to image size and layers
        # hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl

        self.BCEcls = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([self.hyp['cls_pw']], device=self.device))
        self.BCEobj = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([self.hyp['obj_pw']], device=self.device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_bce(eps=0.0)

        self.balance = {3: [4.0, 1.0, 0.4]}.get(
            self.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7

        # stride 16 index
        self.ssi = list(self.stride).index(16) if self.autobalance else 0

    def __call__(self, p, targets):  # predictions, targets, model
        lcls, lbox, lobj = torch.zeros(1, device=self.device), torch.zeros(
            1, device=self.device), torch.zeros(1, device=self.device)

        tcls = targets["labels"]
        tbox = targets["boxes"]
        indices = targets["image_id"]
        anchors = targets["area"]

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            # target obj
            tobj = torch.zeros_like(pi[..., 0], device=self.device)

            n = b.shape[0]  # number of targets
            if n:
                # prediction subset corresponding to targets
                ps = pi[b, a, gj, gi]

                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                # iou(prediction, target)
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness

                riou = iou.detach().clamp(0).type(tobj.dtype)  # iou ratio
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * riou

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(
                        ps[:, 5:], self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * \
                    0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls
        return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()
