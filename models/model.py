import torch
import skorch


from models.layers import FasterRCNN


def init(w):
    if w.dim() < 2:
        return w
    return torch.nn.init.xavier_uniform_(w)


def build_model(max_epochs=2, logdir=".tmp/", train_split=None):
    scheduler = skorch.callbacks.LRScheduler(
        policy=torch.optim.lr_scheduler.CyclicLR,
        base_lr=0.00001,
        max_lr=0.4,
        step_size_up=1900,
        step_size_down=3900,
        step_every='batch',
    )

    model = skorch.NeuralNet(
        FasterRCNN,
        module__pretrained=False,
        batch_size=6,
        max_epochs=max_epochs,
        # optimizer__momentum=0.9,
        iterator_train__shuffle=True,
        iterator_train__num_workers=4,
        iterator_valid__shuffle=False,
        iterator_valid__num_workers=4,
        train_split=train_split,
        callbacks=[
            skorch.callbacks.ProgressBar(),
            skorch.callbacks.Checkpoint(dirname=logdir),
            skorch.callbacks.TrainEndCheckpoint(dirname=logdir),
            scheduler,
            skorch.callbacks.Initializer("*", init),
        ],
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    return model
