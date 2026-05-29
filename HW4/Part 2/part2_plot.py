"""
part2_plot.py
=============
Generate figures for the Part 2 report section.

Reads results/part2_results.json (written by part2_main.py).

Produces:
  figures/comm_loss.png          training loss curve
  figures/comm_error_rates.png   SER, BER, BLER over training
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIGDIR = "figures"
RESULTS = "results/part2_results.json"


def _load():
    with open(RESULTS) as f:
        return json.load(f)


def plot_loss(data, fname):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(data["history"]["loss"], color="tab:blue")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("Part 2: training loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, fname), dpi=150)
    plt.close(fig)


def plot_error_rates(data, fname):
    h = data["history"]
    # These are logged every log_every epochs; reconstruct the epoch axis
    n = len(h["SER"])
    # Determine log interval from total epochs / number of logged points
    total_epochs = len(h["loss"])
    log_every = max(total_epochs // n, 1) if n > 1 else 1
    # First point is epoch 1, then every log_every
    epochs = [1] + [log_every * i for i in range(1, n)]
    epochs = epochs[:n]  # safety

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, h["SER"], "o-", label="SER", markersize=3)
    ax.plot(epochs, h["BER"], "s-", label="BER", markersize=3)
    ax.plot(epochs, h["BLER"], "^-", label="BLER", markersize=3)
    ax.set_xlabel("epoch")
    ax.set_ylabel("error rate")
    ax.set_yscale("log")
    ax.set_title("Part 2: error rates during training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, fname), dpi=150)
    plt.close(fig)


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    data = _load()
    plot_loss(data, "comm_loss.png")
    plot_error_rates(data, "comm_error_rates.png")
    print(f"[plot] Part 2 figures written to ./{FIGDIR}/")


if __name__ == "__main__":
    main()