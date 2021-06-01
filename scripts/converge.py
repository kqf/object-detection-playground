import torch


def main():
    y = torch.as_tensor([0.2, 0.9]).float()
    y_hat = torch.randn(2, requires_grad=True)
    n_ephoch = 100
    lr = 0.1
    optimizer = torch.optim.Adam([y_hat], lr=lr)
    for i in range(n_ephoch):
        optimizer.zero_grad()
        loss = ((y - torch.exp(y_hat) * y) ** 2).mean()
        loss.backward()
        optimizer.step()
        print(f"Epoch {i}, loss {loss.item()}")

    print(y, y_hat)


if __name__ == '__main__':
    main()
