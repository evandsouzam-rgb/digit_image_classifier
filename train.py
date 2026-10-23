import argparse

import torch
import torch.nn as nn

from dataset import get_loaders
from model import MLP


def check(model, loader, loss_fn):
    model.eval()
    total_loss, correct, n = 0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            out = model(x)
            total_loss += loss_fn(out, y).item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            n += len(y)
    return total_loss / n, correct / n


def train(model, train_loader, val_loader, epochs, lr):
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            loss = loss_fn(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()

        train_loss, train_acc = check(model, train_loader, loss_fn)
        val_loss, val_acc = check(model, val_loader, loss_fn)
        if epoch % 5 == 0:
            print(f"epoch {epoch:3d}  train loss {train_loss:.3f} acc {train_acc:.3f}  "
                  f"val loss {val_loss:.3f} acc {val_acc:.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["baseline", "finetune", "scratch"], default="baseline")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--hidden", type=int, default=64)
    args = p.parse_args()

    torch.manual_seed(0)

    if args.mode == "baseline":
        train_loader, val_loader, _ = get_loaders(batch_size=32)
        model = MLP(args.hidden)
    else:
        train_loader, val_loader, _ = get_loaders(batch_size=16, shifted=True, n_train=100)
        model = MLP(args.hidden)
        if args.mode == "finetune":
            model.load_state_dict(torch.load("baseline_model.pt"))

    train(model, train_loader, val_loader, args.epochs, args.lr)

    path = f"{args.mode}_model.pt"
    torch.save(model.state_dict(), path)
    print(f"saved {path}")
