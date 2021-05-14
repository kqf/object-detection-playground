import torch
import skorch


from functools import partial

from detection.models.v3 import YOLO
from detection.losses.v3 import CombinedLoss
from detection.datasets.v3 import DEFAULT_ANCHORS
from detection.inference import infer


def init(w):
    if w.dim() < 2:
        return w
    return torch.nn.init.xavier_uniform_(w)


class DetectionNet(skorch.NeuralNet):

    def predict_proba(self, X):
        nonlin = self._get_predict_nonlinearity()
        y_probas = []
        for yp in self.forward_iter(X, training=False):
            for scale in nonlin(yp):
                y_probas.append(skorch.utils.to_numpy(scale))
        return y_probas


def build_model(max_epochs=2, logdir=".tmp/", train_split=None):
    model = DetectionNet(
        YOLO,
        batch_size=16,
        max_epochs=max_epochs,
        lr=0.000002,
        optimizer=torch.optim.Adam,
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
            # skorch.callbacks.Initializer("*", init),
        ],
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    return model
