"""Обучение: python src/train.py --epochs 10 --out resnet_model.pth"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from data import get_loaders
from model import ResNetTransferLearning


def run_epoch(model, loader, device, criterion, optimizer=None):
    train = optimizer is not None
    model.train(train)
    loss_sum, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_sum += loss.item()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return loss_sum / len(loader), 100 * correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="resnet_model.pth")
    ap.add_argument("--results", default="results")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dl, val_dl, _ = get_loaders(args.data, batch_size=args.batch_size)
    model = ResNetTransferLearning().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.lr)

    hist = {"train_loss": [], "val_loss": [], "val_acc": []}
    for ep in range(1, args.epochs + 1):
        tl, _ = run_epoch(model, train_dl, device, criterion, optimizer)
        vl, va = run_epoch(model, val_dl, device, criterion)
        hist["train_loss"].append(tl); hist["val_loss"].append(vl); hist["val_acc"].append(va)
        print(f"epoch {ep}/{args.epochs}  train_loss {tl:.4f}  val_loss {vl:.4f}  val_acc {va:.2f}%")

    torch.save(model.state_dict(), args.out)
    res = Path(args.results); res.mkdir(exist_ok=True)
    (res / "history.json").write_text(json.dumps(hist, indent=2))

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    xs = range(1, args.epochs + 1)
    ax[0].plot(xs, hist["train_loss"], label="train"); ax[0].plot(xs, hist["val_loss"], label="val")
    ax[0].set(title="Loss", xlabel="epoch"); ax[0].legend(); ax[0].grid()
    ax[1].plot(xs, hist["val_acc"]); ax[1].set(title="Validation accuracy, %", xlabel="epoch"); ax[1].grid()
    fig.tight_layout(); fig.savefig(res / "training_curves.png", dpi=120)


if __name__ == "__main__":
    main()
