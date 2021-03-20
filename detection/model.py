import torch
import skorch


from detection.models.v3 import YOLO
from detection.losses.v3 import CombinedLoss


def init(w):
    if w.dim() < 2:
        return w
    return torch.nn.init.xavier_uniform_(w)


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

    # hyp = {
    #     'box': (1, 0.02, 0.2),  # box loss gain
    #     'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
    #     'cls': (1, 0.2, 4.0),  # cls loss gain
    #     'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
    #     'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
    #     'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
    # }

    model = DetectionNet(
        YOLO,
        batch_size=1,
        max_epochs=max_epochs,
        # optimizer__momentum=0.9,
        criterion=CombinedLoss,
        # criterion__hyp=hyp,
        iterator_train__shuffle=True,
        iterator_train__num_workers=6,
        iterator_valid__shuffle=False,
        iterator_valid__num_workers=6,
        train_split=train_split,
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
