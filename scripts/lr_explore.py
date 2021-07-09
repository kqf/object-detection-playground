import torch
import skorch

import numpy as np
import matplotlib.pyplot as plt


def build_scheduler(n_epochs, batch_size, base_lr):

    scheduler = skorch.callbacks.LRScheduler(
        policy=torch.optim.lr_scheduler.CyclicLR,
        base_lr=base_lr,
        max_lr=0.004,
        step_size_up=batch_size * 4,
        step_size_down=batch_size * 90,
        step_every='batch',
        mode="triangular2"
    )

    # scheduler = skorch.callbacks.LRScheduler(
    #     policy=torch.optim.lr_scheduler.OneCycleLR,
    #     max_lr=0.04,
    #     step_every='batch',
    #     steps_per_epoch=16,
    #     epochs=n_epochs
    # )

    scheduler = skorch.callbacks.LRScheduler(
        policy=torch.optim.lr_scheduler.CyclicLR,
        base_lr=base_lr,
        max_lr=0.004,
        step_size_up=batch_size * 4,
        step_size_down=batch_size * 90,
        step_every='batch',
        mode="exp_range", gamma=0.4
    )

    scheduler = skorch.callbacks.LRScheduler(
        policy=torch.optim.lr_scheduler.CyclicLR,
        base_lr=base_lr,
        max_lr=0.004,
        step_size_up=batch_size * 1,
        step_size_down=batch_size * 4,
        step_every='batch',
        mode="triangular2",
    )
    return scheduler


def main():
    n_epochs = 40
    batch_size = 16
    base_lr = 0.00000009

    scheduler = build_scheduler(n_epochs, batch_size, base_lr)
    sim = scheduler.simulate(batch_size * n_epochs, base_lr)
    plt.plot(sim)

    epochs = np.arange(0, n_epochs, 4)
    steps = epochs * 16

    plt.xticks(steps, epochs)
    plt.show()


if __name__ == '__main__':
    main()
