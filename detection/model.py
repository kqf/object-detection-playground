import torch
import skorch
import numpy as np


from functools import partial

from detection.models.v3 import YOLO
from detection.losses.v3 import CombinedLoss
from detection.datasets.v3 import DEFAULT_ANCHORS


def init(w):
    if w.dim() < 2:
        return w
    return torch.nn.init.xavier_uniform_(w)


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
    import ipdb; ipdb.set_trace(); import IPython; IPython.embed() # noqa
    pass


class DetectionNet(skorch.NeuralNet):

    def predict_proba(self, X):
        nonlin = self._get_predict_nonlinearity()
        y_probas = [[], [], []]
        for yp in self.forward_iter(X, training=False):
            for yprob, scale in zip(y_probas, nonlin(yp)):
                yprob.append(skorch.utils.to_numpy(scale))
        y_proba = [np.concatenate(yprob, 0) for yprob in y_probas]
        return y_proba


def build_model(max_epochs=2, logdir=".tmp/", train_split=None):
    # scheduler = skorch.callbacks.LRScheduler(
    #     policy=torch.optim.lr_scheduler.CyclicLR,
    #     base_lr=0.00001,
    #     max_lr=0.4,
    #     step_size_up=1900,
    #     step_size_down=3900,
    #     step_every='batch',
    # )

    model = DetectionNet(
        YOLO,
        batch_size=16,
        max_epochs=max_epochs,
        lr=0.0001,
        # optimizer__momentum=0.9,
        criterion=CombinedLoss,
        criterion__anchors=DEFAULT_ANCHORS,
        iterator_train__shuffle=True,
        iterator_train__num_workers=6,
        iterator_valid__shuffle=False,
        iterator_valid__num_workers=6,
        train_split=train_split,
        predict_nonlinearity=partial(infer, anchor_boxes=DEFAULT_ANCHORS),
        callbacks=[
            skorch.callbacks.ProgressBar(),
            # skorch.callbacks.Checkpoint(dirname=logdir),
            skorch.callbacks.TrainEndCheckpoint(dirname=logdir),
            # scheduler,
            skorch.callbacks.Initializer("*", init),
        ],
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    return model
