import torch


def main():
    y = torch.as_tensor([0.2, 0.9]).float()
    logits = torch.randn(2, requires_grad=True)
    n_ephoch = 100
    lr = 0.1
    optimizer = torch.optim.Adam([logits], lr=lr)
    for i in range(n_ephoch):
        optimizer.zero_grad()
        loss = ((y - torch.exp(logits) * y) ** 2).mean()
        loss.backward()
        optimizer.step()
        print(f"Epoch {i}, loss {loss.item()}")

    print(y, torch.exp(logits) * y)


if __name__ == '__main__':
    main()
