"""
plot.py
=======
Generate all figures for the CS515 HW4 Part 1 report.

Reads:
  - results/summary_all.json   (training history + test metrics, written by main.py)
  - checkpoints/detector*.pth  (to recompute PR curves from raw probabilities)

Produces (in figures/):
  1. loss_curves_b.png        train/val MSE, LSTM vs GRU, raw-return target
  2. loss_curves_c.png        train/val MSE, LSTM vs GRU, rolling-average target
  3. per_horizon_mse.png      MSE by horizon d=1..5, b vs c
  4. b_vs_c_test_mse.png      grouped bars, the rolling-average improvement
  5. pr_curves_d.png          precision-recall curves + baseline, annotated PR-AUC
  6. detector_acc_vs_f1.png   the majority-class trap (val acc up while val F1 collapses)

Run AFTER `python main.py --mode all`:
  python plot.py
"""

from __future__ import annotations

import json
import os
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")  # headless; just writes PNGs
import matplotlib.pyplot as plt
import numpy as np
import torch

from parameters import DataConfig, DetectorConfig
from data import build_dataset, make_loaders
from models import build_model
from train import _collect_probs, resolve_device

FIGDIR = "figures"
RESULTS = "results/summary_all.json"
HORIZONS = [1, 2, 3, 4, 5]


def _load_summary():
    if not os.path.exists(RESULTS):
        raise FileNotFoundError(
            f"{RESULTS} not found. Run `python main.py --mode all` first."
        )
    with open(RESULTS) as fh:
        return json.load(fh)


def plot_loss_curves(summary, part: str, title: str, fname: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    for rnn, color in [("lstm", "tab:blue"), ("gru", "tab:orange")]:
        h = summary[part][rnn]["history"]
        ax.plot(h["train"], color=color, ls="-", label=f"{rnn.upper()} train")
        ax.plot(h["val"], color=color, ls="--", label=f"{rnn.upper()} val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, fname), dpi=150)
    plt.close(fig)


def plot_per_horizon(summary, fname: str):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, part, name in [(axes[0], "b", "Raw d-day return"),
                           (axes[1], "c", "Rolling average (l=3)")]:
        x = np.arange(len(HORIZONS))
        w = 0.35
        for i, (rnn, color) in enumerate([("lstm", "tab:blue"), ("gru", "tab:orange")]):
            ph = summary[part][rnn]["per_horizon"]
            ax.bar(x + (i - 0.5) * w, ph, w, label=rnn.upper(), color=color)
        ax.set_xticks(x)
        ax.set_xticklabels([f"d={d}" for d in HORIZONS])
        ax.set_title(name)
        ax.set_xlabel("forecast horizon")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("test MSE")
    fig.suptitle("Per-horizon test MSE: error grows with horizon")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, fname), dpi=150)
    plt.close(fig)


def plot_b_vs_c(summary, fname: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    rnns = ["lstm", "gru"]
    x = np.arange(len(rnns))
    w = 0.35
    ret = [summary["b"][r]["test_mse"] for r in rnns]
    roll = [summary["c"][r]["test_mse"] for r in rnns]
    ax.bar(x - w / 2, ret, w, label="raw return (b)", color="tab:red")
    ax.bar(x + w / 2, roll, w, label="rolling avg (c)", color="tab:green")
    for i, (a, b) in enumerate(zip(ret, roll)):
        drop = 100 * (a - b) / a
        ax.text(i, max(a, b) * 1.02, f"-{drop:.0f}%", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([r.upper() for r in rnns])
    ax.set_ylabel("test MSE")
    ax.set_title("Rolling-average target lowers MSE (more stable training)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, fname), dpi=150)
    plt.close(fig)


def _pr_curve(probs, labels):
    """Return (recall, precision) arrays sorted by descending score."""
    order = np.argsort(-probs)
    y = labels[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(y.sum(), 1)
    return recall, precision


def plot_pr_curves(summary, fname: str):
    """Recompute PR curves from saved detector checkpoints on the test set."""
    dcfg = DataConfig()
    data = build_dataset(dcfg, target_type="signal")
    loaders = make_loaders(data, dcfg)
    device = resolve_device("cpu")

    fig, ax = plt.subplots(figsize=(6, 4))
    baseline = None
    for rnn, color in [("lstm", "tab:blue"), ("gru", "tab:orange")]:
        ckpt = f"checkpoints/detector{rnn.upper()}.pth"
        if not os.path.exists(ckpt):
            print(f"[plot] missing {ckpt}, skipping PR curve for {rnn}")
            continue
        dc = DetectorConfig(rnn_type=rnn, input_size=len(dcfg.feature_cols))
        model = build_model(dc, close_idx=dcfg.feature_cols.index(dcfg.target_col))
        model.load_state_dict(torch.load(ckpt, map_location="cpu")["state_dict"])
        probs, labels = _collect_probs(model, loaders["test"], device)
        rec, prec = _pr_curve(probs, labels)
        ap = summary["d"][rnn]["pr_auc"]
        ax.plot(rec, prec, color=color, label=f"{rnn.upper()} (PR-AUC={ap:.3f})")
        baseline = float(labels.mean())

    if baseline is not None:
        ax.axhline(baseline, ls=":", color="gray",
                   label=f"random baseline ({baseline:.3f})")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("Part d: turning-point detection (precision-recall)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, fname), dpi=150)
    plt.close(fig)


def plot_acc_vs_f1(summary, fname: str):
    """Visualise the majority-class trap using the detector's val loss history.

    We only logged loss history (not per-epoch acc/F1), so this panel plots the
    val loss curves for both detectors -- the divergence story is described in the
    report text. If per-epoch acc/F1 were logged they would be overlaid here.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    for rnn, color in [("lstm", "tab:blue"), ("gru", "tab:orange")]:
        h = summary["d"][rnn]["history"]
        ax.plot(h["train"], color=color, ls="-", label=f"{rnn.upper()} train")
        ax.plot(h["val"], color=color, ls="--", label=f"{rnn.upper()} val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("BCE loss")
    ax.set_title("Part d: detector training (BCEWithLogits)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, fname), dpi=150)
    plt.close(fig)


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    summary = _load_summary()

    if "b" in summary:
        plot_loss_curves(summary, "b", "Part b: raw d-day return (train/val MSE)",
                         "loss_curves_b.png")
    if "c" in summary:
        plot_loss_curves(summary, "c", "Part c: rolling-average return (train/val MSE)",
                         "loss_curves_c.png")
    if "b" in summary and "c" in summary:
        plot_per_horizon(summary, "per_horizon_mse.png")
        plot_b_vs_c(summary, "b_vs_c_test_mse.png")
    if "d" in summary:
        plot_pr_curves(summary, "pr_curves_d.png")
        plot_acc_vs_f1(summary, "detector_loss_d.png")

    print(f"[plot] figures written to ./{FIGDIR}/")


if __name__ == "__main__":
    main()
