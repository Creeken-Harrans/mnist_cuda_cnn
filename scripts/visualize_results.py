#!/usr/bin/env python3
"""Visualize outputs exported by the from-scratch CUDA MNIST trainer.

The CUDA program writes CSV files to runs/latest by default:
  - metrics.csv
  - prediction_samples.csv
  - conv1_weights.csv

This script turns them into clean Matplotlib figures:
  - training_curves.png
  - prediction_grid.png
  - conv1_filters.png
  - confidence_histogram.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np


DIGITS = list(range(10))


def setup_style() -> None:
    """Use a pleasant Matplotlib style, falling back safely on old installs."""
    for style in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid"):
        try:
            plt.style.use(style)
            break
        except OSError:
            continue

    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "axes.titleweight": "bold",
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def read_metrics(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")

    rows: List[Dict[str, str]] = list(csv.DictReader(path.open()))
    if not rows:
        raise ValueError(f"Metrics file is empty: {path}")

    return {
        "epoch": np.array([int(r["epoch"]) for r in rows], dtype=np.int32),
        "train_loss": np.array([float(r["train_loss"]) for r in rows], dtype=np.float32),
        "train_acc": np.array([float(r["train_acc"]) for r in rows], dtype=np.float32),
        "test_acc": np.array([float(r["test_acc"]) for r in rows], dtype=np.float32),
        "lr": np.array([float(r["lr"]) for r in rows], dtype=np.float32),
        "elapsed_sec": np.array([float(r["elapsed_sec"]) for r in rows], dtype=np.float32),
    }


def read_prediction_samples(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction sample file: {path}")

    samples: List[Dict[str, object]] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            probs = np.array([float(row[f"p{i}"]) for i in DIGITS], dtype=np.float32)
            pixels = np.array([float(row[f"pixel{i}"]) for i in range(28 * 28)], dtype=np.float32).reshape(28, 28)
            samples.append(
                {
                    "index": int(row["index"]),
                    "label": int(row["label"]),
                    "pred": int(row["pred"]),
                    "confidence": float(row["confidence"]),
                    "probs": probs,
                    "image": pixels,
                }
            )
    if not samples:
        raise ValueError(f"Prediction sample file is empty: {path}")
    return samples


def read_conv1_weights(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing conv1 weight file: {path}")

    filters: Dict[int, np.ndarray] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            oc = int(row["filter"])
            kh = int(row["kh"])
            kw = int(row["kw"])
            filters.setdefault(oc, np.zeros((5, 5), dtype=np.float32))[kh, kw] = float(row["weight"])

    if not filters:
        raise ValueError(f"Conv1 weight file is empty: {path}")
    return np.stack([filters[k] for k in sorted(filters)], axis=0)


def plot_training_curves(metrics: Dict[str, np.ndarray], out_path: Path) -> None:
    epoch = metrics["epoch"]
    train_loss = metrics["train_loss"]
    train_acc = metrics["train_acc"] * 100.0
    test_acc = metrics["test_acc"] * 100.0

    fig, ax_loss = plt.subplots(figsize=(9.8, 5.5))
    ax_acc = ax_loss.twinx()

    loss_line = ax_loss.plot(epoch, train_loss, marker="o", linewidth=2.2, label="train loss")
    train_line = ax_acc.plot(epoch, train_acc, marker="s", linewidth=2.2, label="train acc")
    test_line = ax_acc.plot(epoch, test_acc, marker="^", linewidth=2.2, label="test acc")

    ax_loss.set_title("MNIST CUDA CNN training curves")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Cross-entropy loss")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_loss.set_xticks(epoch)
    ax_acc.set_ylim(max(0, min(train_acc.min(), test_acc.min()) - 5), 100)

    lines = loss_line + train_line + test_line
    labels = [line.get_label() for line in lines]
    ax_loss.legend(lines, labels, loc="center right")

    final_test = test_acc[-1]
    ax_loss.text(
        0.02,
        0.05,
        f"final test acc: {final_test:.2f}%",
        transform=ax_loss.transAxes,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.85},
    )

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_prediction_grid(samples: Sequence[Dict[str, object]], out_path: Path, max_images: int = 36) -> None:
    shown = list(samples[:max_images])
    cols = 6
    rows = int(np.ceil(len(shown) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.25))
    axes = np.array(axes).reshape(rows, cols)

    for ax in axes.flat:
        ax.axis("off")

    for ax, sample in zip(axes.flat, shown):
        image = sample["image"]
        label = int(sample["label"])
        pred = int(sample["pred"])
        conf = float(sample["confidence"])
        correct = pred == label

        ax.imshow(image, cmap="gray_r", vmin=0.0, vmax=1.0)
        ax.set_title(
            f"y={label}  pred={pred}\nconf={conf:.2f}",
            color=("#1b7f3a" if correct else "#b42318"),
            fontsize=10,
        )
        ax.axis("off")

    fig.suptitle("Test-set predictions", fontsize=16, fontweight="bold", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_conv1_filters(weights: np.ndarray, out_path: Path) -> None:
    n = weights.shape[0]
    cols = min(8, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.7, rows * 1.9))
    axes = np.array(axes).reshape(rows, cols)

    absmax = float(np.max(np.abs(weights))) or 1.0
    for ax in axes.flat:
        ax.axis("off")

    for i, ax in enumerate(axes.flat[:n]):
        ax.imshow(weights[i], cmap="coolwarm", vmin=-absmax, vmax=absmax)
        ax.set_title(f"filter {i}", fontsize=10)
        ax.axis("off")

    fig.suptitle("Conv1 learned 5×5 filters", fontsize=16, fontweight="bold", y=0.98)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_confidence_histogram(samples: Sequence[Dict[str, object]], out_path: Path) -> None:
    correct_conf = [float(s["confidence"]) for s in samples if int(s["label"]) == int(s["pred"])]
    wrong_conf = [float(s["confidence"]) for s in samples if int(s["label"]) != int(s["pred"])]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bins = np.linspace(0.0, 1.0, 21)
    if correct_conf:
        ax.hist(correct_conf, bins=bins, alpha=0.75, label="correct")
    if wrong_conf:
        ax.hist(wrong_conf, bins=bins, alpha=0.75, label="wrong")

    ax.set_title("Prediction confidence distribution")
    ax.set_xlabel("Softmax confidence")
    ax.set_ylabel("Sample count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize MNIST CUDA CNN run outputs.")
    parser.add_argument("--run-dir", default="runs/latest", help="Directory containing metrics/prediction/weight CSV files.")
    parser.add_argument("--out-dir", default=None, help="Figure directory. Default: <run-dir>/figures")
    parser.add_argument("--max-images", type=int, default=36, help="Maximum test images in prediction grid.")
    args = parser.parse_args()

    setup_style()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = read_metrics(run_dir / "metrics.csv")
    plot_training_curves(metrics, out_dir / "training_curves.png")
    print(f"[viz] wrote {out_dir / 'training_curves.png'}")

    pred_path = run_dir / "prediction_samples.csv"
    if pred_path.exists():
        samples = read_prediction_samples(pred_path)
        plot_prediction_grid(samples, out_dir / "prediction_grid.png", max_images=args.max_images)
        plot_confidence_histogram(samples, out_dir / "confidence_histogram.png")
        print(f"[viz] wrote {out_dir / 'prediction_grid.png'}")
        print(f"[viz] wrote {out_dir / 'confidence_histogram.png'}")
    else:
        print(f"[viz] skip prediction figures; missing {pred_path}")

    conv_path = run_dir / "conv1_weights.csv"
    if conv_path.exists():
        conv1 = read_conv1_weights(conv_path)
        plot_conv1_filters(conv1, out_dir / "conv1_filters.png")
        print(f"[viz] wrote {out_dir / 'conv1_filters.png'}")
    else:
        print(f"[viz] skip conv1 filter figure; missing {conv_path}")


if __name__ == "__main__":
    main()
