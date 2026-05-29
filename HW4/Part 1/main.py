"""
main.py
=======
Entry point / experiment router for CS515 HW4 Part 1.

Modes
-----
  --mode b   : train StockLSTM and StockGRU on raw d-day return targets
  --mode c   : repeat with rolling-average targets; compare MSE vs (b)
  --mode d   : bi-directional detector for buy/pass turning-point classification
  --mode all : run b, c, d sequentially and print a consolidated summary

Pooled single model
--------------------
Every mode trains ONE model across all tickers pooled together (the
f_theta: R^{N x T x F} -> R^{N x D} formulation). REMEMBER to state this explicitly
in the report's methodology section.

Examples
--------
  python main.py --mode b
  python main.py --mode all --epochs 100
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace

import numpy as np

from parameters import DataConfig, ModelConfig, DetectorConfig, TrainConfig
from data import build_dataset, make_loaders
from models import build_model
from train import train_regression, train_classifier


def _close_idx(dcfg: DataConfig) -> int:
    return dcfg.feature_cols.index(dcfg.target_col)


def run_regression_part(target_type: str, dcfg, mcfg, tcfg, label: str):
    """Train both LSTM and GRU on the given target type; return results dict."""
    data = build_dataset(dcfg, target_type=target_type)
    loaders = make_loaders(data, dcfg)
    n_tr = len(data["train"][0])
    print(f"\n=== Part {label} ({target_type}) | pooled windows: "
          f"train={n_tr} val={len(data['val'][0])} test={len(data['test'][0])} ===")

    results = {}
    for rnn_type in ["lstm", "gru"]:
        mc = replace(mcfg, rnn_type=rnn_type, input_size=len(dcfg.feature_cols), output_size=len(dcfg.horizons))
        model = build_model(mc, close_idx=_close_idx(dcfg))
        tag = f"stock{rnn_type.upper()}_{target_type}"
        results[rnn_type] = train_regression(model, loaders, tcfg, tag=tag)
    return results


def run_detector_part(dcfg, tcfg, label: str = "d"):
    data = build_dataset(dcfg, target_type="signal")
    loaders = make_loaders(data, dcfg)
    pos = int((data["train"][1] == 1).sum())
    tot = len(data["train"][1])
    print(f"\n=== Part {label} (turning-point) | train windows {tot}, "
          f"positives {pos} ({100*pos/max(tot,1):.1f}%) ===")

    results = {}
    for rnn_type in ["lstm", "gru"]:
        dc = DetectorConfig(rnn_type=rnn_type, input_size=len(dcfg.feature_cols))
        model = build_model(dc, close_idx=_close_idx(dcfg))
        results[rnn_type] = train_classifier(model, loaders, tcfg, tag=f"detector{rnn_type.upper()}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["b", "c", "d", "all"], default="all")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    dcfg, mcfg, tcfg = DataConfig(), ModelConfig(), TrainConfig()
    if args.epochs is not None:
        tcfg = replace(tcfg, epochs=args.epochs)
    if args.device is not None:
        tcfg = replace(tcfg, device=args.device)

    summary = {}

    if args.mode in ("b", "all"):
        summary["b"] = run_regression_part("return", dcfg, mcfg, tcfg, "b")

    if args.mode in ("c", "all"):
        summary["c"] = run_regression_part("rolling", dcfg, mcfg, tcfg, "c")

    if args.mode in ("d", "all"):
        summary["d"] = run_detector_part(dcfg, tcfg, "d")

    # ---- consolidated comparison (parts b vs c) ----
    if "b" in summary and "c" in summary:
        print("\n================ b vs c (test MSE) ================")
        for rnn in ["lstm", "gru"]:
            mb = summary["b"][rnn]["test_mse"]
            mc = summary["c"][rnn]["test_mse"]
            tag = "more stable / lower" if mc < mb else "higher"
            print(f"  {rnn.upper():4s}: return={mb:.6e}  rolling={mc:.6e}  -> rolling {tag}")

    # ---- persist results so plot.py can render figures without retraining ----
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
    out_path = f"results/summary_{args.mode}.json"
    with open(out_path, "w") as fh:
        json.dump(_jsonable(summary), fh, indent=2)
    print(f"\n[done] results saved to {out_path}; checkpoints in ./checkpoints.")


if __name__ == "__main__":
    main()