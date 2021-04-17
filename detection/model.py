import torch
import skorch


from functional import partial

from detection.models.v3 import YOLO
from detection.losses.v3 import CombinedLoss
from detection.datasets.v3 import DEFAULT_ANCHORS


def init(w):
    if w.dim() < 2:
        return w
    return torch.nn.init.xavier_uniform_(w)


def infer(batch, anchor_boxes):
    predictions = []

    for i, (scale, achors) in enumerate(batch, anchor_boxes):
        # Copy don't mutate the original batch
        prediction = batch[..., :6].detach().clone()

        prediction[..., 0:2] = torch.sigmoid(scale[..., 0:2])
        prediction[..., 2:5] = torch.exp(scale[..., 2:5]) * anchor_boxes
        prediction[..., 0] = torch.sigmoid(scale[..., 0])
        prediction[..., 5] = torch.argmax(scale[..., 5:], dim=-1).unsqueeze(-1)
        predictions.append(prediction)

    return predictions


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
