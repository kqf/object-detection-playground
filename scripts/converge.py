import torch


def main():
    X = torch.as_tensor([1, 2]).float()
    y = torch.as_tensor([3, 4]).float()
    w = torch.randn(1, requires_grad=True)
    b = torch.randn(1, requires_grad=True)
    n_ephoch = 100
    lr = 0.001
    for i in range(n_ephoch):
        loss = ((y - w @ X + b) ** 2).mean()
        loss.backward()
        w.data -= lr * w.grad
        b.data -= lr * b.grad
        w.grad.zero_()
        b.grad.zero_()

    print(w, b)


if __name__ == '__main__':
    main()
