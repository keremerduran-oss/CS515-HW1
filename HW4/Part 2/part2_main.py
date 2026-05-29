"""
part2_main.py
=============
Entry point for CS515 HW4 Part 2: Neural Communication Protocol.

Usage:
  python part2_main.py
  python part2_main.py --epochs 300 --device cuda
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from part2_parameters import CommConfig, CommModelConfig, CommTrainConfig
from part2_train import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    args = ap.parse_args()

    cfg = CommConfig()
    mcfg = CommModelConfig()
    tcfg = CommTrainConfig()

    if args.epochs is not None:
        tcfg.epochs = args.epochs
    if args.device is not None:
        tcfg.device = args.device
    if args.batch_size is not None:
        tcfg.batch_size = args.batch_size

    print("=" * 60)
    print("HW4 Part 2: Neural Communication Protocol")
    print(f"  Message: {cfg.num_symbols} symbols × alphabet {cfg.alphabet_size}")
    print(f"  Rounds: T = {cfg.num_rounds}, σ² = {cfg.noise_var}")
    print(f"  Training: {tcfg.epochs} epochs, batch {tcfg.batch_size}")
    print("=" * 60)

    results = train(cfg, mcfg, tcfg)

    # Save results
    def _jsonable(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, dict):
            return {k: _jsonable(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_jsonable(v) for v in o]
        return o

    os.makedirs("results", exist_ok=True)
    with open("results/part2_results.json", "w") as f:
        json.dump(_jsonable(results), f, indent=2)

    print(f"\n[done] results saved to results/part2_results.json")


if __name__ == "__main__":
    main()
