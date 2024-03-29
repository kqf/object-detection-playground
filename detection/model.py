import torch
import skorch


from functools import partial

from detection.models.v3 import YOLO
from detection.loss import CombinedLoss
from detection.dataset import DEFAULT_ANCHORS
from detection.inference import infer


def init(w):
    if w.dim() < 2:
        return w
    return torch.nn.init.xavier_normal_(w)


class DetectionNet(skorch.NeuralNet):

    def predict_proba(self, X):
        nonlin = self._get_predict_nonlinearity()
        y_probas = []
        for yp in self.forward_iter(X, training=True):
            for scale in nonlin(yp):
                y_probas.append(skorch.utils.to_numpy(scale))
        return y_probas


def build_model(max_epochs=2, logdir=".tmp/", top_n=None, train_split=None):
    # Optimal for box width
    base_lr = 0.0000001

    # Bad results
    # base_lr = 0.001

    # A slight improvement
    # base_lr = 0.0001

    # To noisy
    # base_lr = 0.000001

    batch_size = 16

    scheduler = skorch.callbacks.LRScheduler(
        policy=torch.optim.lr_scheduler.CyclicLR,
        base_lr=base_lr,
        max_lr=0.004,
        step_size_up=batch_size * 10,
        step_size_down=batch_size * 40,
        step_every='batch',
        mode="triangular2",
    )

    model = DetectionNet(
        YOLO,
        module__n_scales=len(DEFAULT_ANCHORS[0]),
        batch_size=batch_size,
        max_epochs=max_epochs,
        lr=base_lr,
        # optimizer=torch.optim.Adam,
        # optimizer__momentum=0.9,
        criterion=CombinedLoss,
        criterion__anchors=DEFAULT_ANCHORS,
        iterator_train__shuffle=True,
        iterator_train__num_workers=6,
        iterator_valid__shuffle=False,
        iterator_valid__num_workers=6,
        train_split=train_split,
        predict_nonlinearity=partial(
            infer,
            anchor_boxes=DEFAULT_ANCHORS,
            top_n=top_n,
            min_iou=0.5,
            threshold=0.5,
        ),
        callbacks=[
            scheduler,
            skorch.callbacks.ProgressBar(),
            skorch.callbacks.TrainEndCheckpoint(dirname=logdir),
            skorch.callbacks.Initializer("*", init),
        ],
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    return model
