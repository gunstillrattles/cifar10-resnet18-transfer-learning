"""Оценка на тесте: python src/evaluate.py --weights resnet_model.pth"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from data import CLASSES, get_test_loader
from model import ResNetTransferLearning


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="resnet_model.pth")
    ap.add_argument("--data", default="data")
    ap.add_argument("--results", default="results")
    ap.add_argument("--limit", type=int, default=0, help="оценить только первые N батчей (для быстрой проверки)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dl = get_test_loader(args.data, download=False)
    model = ResNetTransferLearning(pretrained=False)
    model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    model.to(device).eval()

    preds, labels = [], []
    with torch.no_grad():
        for i, (x, y) in enumerate(test_dl):
            if args.limit and i >= args.limit:
                break
            preds.append(model(x.to(device)).argmax(1).cpu()); labels.append(y)
    preds, labels = torch.cat(preds).numpy(), torch.cat(labels).numpy()

    cm = confusion_matrix(labels, preds, labels=range(10))
    per_class = {c: round(float(cm[i, i] / cm[i].sum() * 100), 2) for i, c in enumerate(CLASSES)}
    metrics = {"test_accuracy": round(float((preds == labels).mean() * 100), 2),
               "n_test": int(len(labels)), "per_class_accuracy": per_class}
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    res = Path(args.results); res.mkdir(exist_ok=True)
    (res / "metrics.json").write_text(json.dumps(metrics, indent=2))
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(cmap="Blues", xticks_rotation=45, ax=ax, colorbar=False)
    ax.set_title("Confusion matrix (test)"); fig.tight_layout(); fig.savefig(res / "confusion_matrix.png", dpi=120)
    plt.close(fig)

    # примеры: первые 6 верных и первые 6 ошибочных предсказаний (без отбора «красивых»)
    raw = test_dl.dataset.data[:len(labels)]   # uint8, N x 32 x 32 x 3 (исходное разрешение)
    ok, bad = np.where(preds == labels)[0][:6], np.where(preds != labels)[0][:6]
    fig, axes = plt.subplots(2, 6, figsize=(13, 6.5))
    for row, idxs in ((0, ok), (1, bad)):
        for ax, i in zip(axes[row], idxs):
            ax.imshow(raw[i]); ax.axis("off")
            ax.set_title(f"pred: {CLASSES[preds[i]]}\ntrue: {CLASSES[labels[i]]}", fontsize=9,
                         color="green" if row == 0 else "red")
    fig.suptitle("Test examples: first 6 correct (top) and first 6 wrong (bottom)")
    fig.tight_layout(h_pad=3); fig.savefig(res / "sample_predictions.png", dpi=120)


if __name__ == "__main__":
    main()
