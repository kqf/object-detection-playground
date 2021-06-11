import torch


def main():
    n_grids = 13
    y = torch.as_tensor([0.28, 0.22]).float() * n_grids
    logits = torch.randn(2, requires_grad=True)
    n_ephoch = 1000
    lr = 0.1
    optimizer = torch.optim.Adam([logits], lr=lr)
    for i in range(n_ephoch):
        optimizer.zero_grad()

        trues = torch.log(1e-16 + y / y)
        preds = logits
        loss = ((trues - preds) ** 2).mean()
        loss.backward()
        optimizer.step()
        print(f"Epoch {i}, loss {loss.item()}")

    print(y, torch.exp(logits) * y / n_grids)


if __name__ == '__main__':
    main()
