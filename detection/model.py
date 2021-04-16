import torch
import skorch


from detection.models.v3 import YOLO
from detection.losses.v3 import CombinedLoss
from detection.datasets.v3 import DEFAULT_ANCHORS


def init(w):
    if w.dim() < 2:
        return w
    return torch.nn.init.xavier_uniform_(w)


def infer(batch):
    batch[..., 0:2] = torch.sigmoid(batch[..., 0:2])
    batch[..., 2:5] = torch.exp(batch[..., 2:5])
    batch[..., 0] = torch.sigmoid(batch[..., 0])
    batch[..., 5] = torch.argmax(batch[..., 5:], dim=-1).unsqueeze(-1)
    return batch[..., :6]


class DetectionNet(skorch.NeuralNet):
    pass


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
        predict_nonlinearity=infer,
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
