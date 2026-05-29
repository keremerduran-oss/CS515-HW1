"""
train.py
========
Training / evaluation loops for CS515 HW4 Part 1.

Two task modes:
  - "regression"     (parts b, c): MSE loss over D horizons.
  - "classification" (part d):     BCEWithLogits, reports accuracy / precision / recall / F1.

Shared utilities: seeding, optimiser construction, early stopping, grad clipping,
checkpoint save/load.
"""

from __future__ import annotations

import os
import random
from dataclasses import asdict
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from parameters import TrainConfig


# --------------------------------------------------------------------------------------
# Reproducibility / device
# --------------------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_optimizer(model: nn.Module, cfg: TrainConfig):
    if cfg.optimizer.lower() == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)


# --------------------------------------------------------------------------------------
# Regression (parts b, c)
# --------------------------------------------------------------------------------------
def _run_regression_epoch(model, loader, criterion, optimizer, device, cfg, train: bool):
    model.train() if train else model.eval()
    total, n = 0.0, 0
    torch.set_grad_enabled(train)
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = criterion(pred, y)
        if train:
            optimizer.zero_grad()
            loss.backward()
            if cfg.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
        total += loss.item() * len(X)
        n += len(X)
    torch.set_grad_enabled(True)
    return total / max(n, 1)


def train_regression(model, loaders, cfg: TrainConfig, tag: str = "model") -> Dict:
    device = resolve_device(cfg.device)
    model.to(device)
    set_seed(cfg.seed)
    criterion = nn.MSELoss()
    optimizer = build_optimizer(model, cfg)

    best_val, best_state, bad = float("inf"), None, 0
    history = {"train": [], "val": []}

    for ep in range(1, cfg.epochs + 1):
        tr = _run_regression_epoch(model, loaders["train"], criterion, optimizer, device, cfg, True)
        va = _run_regression_epoch(model, loaders["val"], criterion, optimizer, device, cfg, False)
        history["train"].append(tr)
        history["val"].append(va)

        if va < best_val - cfg.min_delta:
            best_val, best_state, bad = va, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
        if ep % 10 == 0 or ep == 1:
            print(f"[{tag}] epoch {ep:3d} | train MSE {tr:.6e} | val MSE {va:.6e}")
        if bad >= cfg.patience:
            print(f"[{tag}] early stop at epoch {ep} (best val {best_val:.6e})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    path = os.path.join(cfg.ckpt_dir, f"{tag}.pth")
    torch.save({"state_dict": model.state_dict(), "best_val": best_val}, path)

    test_mse = _run_regression_epoch(model, loaders["test"], criterion, optimizer, device, cfg, False)
    per_h = per_horizon_mse(model, loaders["test"], device)
    print(f"[{tag}] TEST MSE {test_mse:.6e} | per-horizon {np.round(per_h, 6)}")
    return {"history": history, "best_val": best_val, "test_mse": test_mse, "per_horizon": per_h}


def per_horizon_mse(model, loader, device) -> np.ndarray:
    """MSE broken down by forecast horizon d=1..5 (useful for the report tables)."""
    model.eval()
    se, n = None, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            sq = ((pred - y) ** 2).sum(dim=0).cpu().numpy()
            se = sq if se is None else se + sq
            n += len(X)
    return (se / max(n, 1)) if se is not None else np.array([])


# --------------------------------------------------------------------------------------
# Classification (part d)
# --------------------------------------------------------------------------------------
def compute_pos_weight(loader) -> float:
    pos, neg = 0, 0
    for _, y in loader:
        pos += (y == 1).sum().item()
        neg += (y == 0).sum().item()
    return (neg / max(pos, 1)) if pos else 1.0


def _run_clf_epoch(model, loader, criterion, optimizer, device, cfg, train: bool):
    model.train() if train else model.eval()
    total, n = 0.0, 0
    torch.set_grad_enabled(train)
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = criterion(logits, y)
        if train:
            optimizer.zero_grad()
            loss.backward()
            if cfg.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
        total += loss.item() * len(X)
        n += len(X)
    torch.set_grad_enabled(True)
    return total / max(n, 1)


def _collect_probs(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    """Return (probs, labels) as flat numpy arrays over an entire loader."""
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            p = torch.sigmoid(model(X)).cpu().numpy().ravel()
            probs.append(p)
            labels.append(y.cpu().numpy().ravel())
    if not probs:
        return np.zeros(0), np.zeros(0)
    return np.concatenate(probs), np.concatenate(labels)


def average_precision(probs: np.ndarray, labels: np.ndarray) -> float:
    """
    Threshold-independent PR summary (area under the precision-recall curve),
    computed as the step-wise average precision: sum_k (R_k - R_{k-1}) * P_k.
    Measures whether the model RANKS positives above negatives, separate from
    where the decision threshold is placed. Useful for rare-event classification.
    """
    if labels.sum() == 0 or len(labels) == 0:
        return 0.0
    order = np.argsort(-probs)                 # descending by score
    y = labels[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(y.sum(), 1)
    # prepend recall=0 so the first rectangle is counted
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - recall_prev) * precision))


def tune_threshold(model, loader, device, n_steps: int = 200) -> Tuple[float, float]:
    """
    Sweep decision thresholds on the given (validation) loader and return the
    (threshold, f1) pair that maximises F1. With a rare positive class the optimal
    cutoff is well below 0.5, so this is standard practice rather than a hack.
    """
    probs, labels = _collect_probs(model, loader, device)
    if labels.sum() == 0:
        return 0.5, 0.0
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.01, 0.99, n_steps):
        pred = (probs >= t).astype(np.float32)
        tp = float(((pred == 1) & (labels == 1)).sum())
        fp = float(((pred == 1) & (labels == 0)).sum())
        fn = float(((pred == 0) & (labels == 1)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


def clf_metrics(model, loader, device, thresh: float = 0.5) -> Dict[str, float]:
    model.eval()
    tp = fp = tn = fn = 0
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            prob = torch.sigmoid(model(X)).cpu()
            pred = (prob >= thresh).float()
            y = y.cpu()
            tp += ((pred == 1) & (y == 1)).sum().item()
            fp += ((pred == 1) & (y == 0)).sum().item()
            tn += ((pred == 0) & (y == 0)).sum().item()
            fn += ((pred == 0) & (y == 1)).sum().item()
    acc = (tp + tn) / max(tp + fp + tn + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    return {"acc": acc, "precision": prec, "recall": rec, "f1": f1,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def train_classifier(model, loaders, cfg: TrainConfig, tag: str = "detector") -> Dict:
    device = resolve_device(cfg.device)
    model.to(device)
    set_seed(cfg.seed)

    pw = cfg.pos_weight if cfg.pos_weight is not None else compute_pos_weight(loaders["train"])
    print(f"[{tag}] pos_weight = {pw:.3f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=device))
    optimizer = build_optimizer(model, cfg)

    best_val, best_state, bad = float("inf"), None, 0
    history = {"train": [], "val": []}

    for ep in range(1, cfg.epochs + 1):
        tr = _run_clf_epoch(model, loaders["train"], criterion, optimizer, device, cfg, True)
        va = _run_clf_epoch(model, loaders["val"], criterion, optimizer, device, cfg, False)
        history["train"].append(tr)
        history["val"].append(va)
        if va < best_val - cfg.min_delta:
            best_val, best_state, bad = va, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
        if ep % 10 == 0 or ep == 1:
            m = clf_metrics(model, loaders["val"], device)
            print(f"[{tag}] epoch {ep:3d} | train {tr:.4f} | val {va:.4f} "
                  f"| val F1 {m['f1']:.3f} acc {m['acc']:.3f}")
        if bad >= cfg.patience:
            print(f"[{tag}] early stop at epoch {ep}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, os.path.join(cfg.ckpt_dir, f"{tag}.pth"))

    # --- pick the operating threshold on VALIDATION (never on test) ---
    best_t, val_f1_at_t = tune_threshold(model, loaders["val"], device)

    # --- test metrics at the default 0.5 AND at the val-tuned threshold ---
    test_default = clf_metrics(model, loaders["test"], device, thresh=0.5)
    test_tuned = clf_metrics(model, loaders["test"], device, thresh=best_t)

    # --- threshold-independent ranking quality (PR-AUC / average precision) ---
    p_test, y_test = _collect_probs(model, loaders["test"], device)
    pr_auc = average_precision(p_test, y_test)
    pos_rate = float(y_test.mean()) if len(y_test) else 0.0  # = random-baseline PR-AUC

    print(f"[{tag}] tuned threshold = {best_t:.3f} (val F1 {val_f1_at_t:.3f})")
    print(f"[{tag}] TEST @0.5    acc {test_default['acc']:.3f} | P {test_default['precision']:.3f} "
          f"| R {test_default['recall']:.3f} | F1 {test_default['f1']:.3f}")
    print(f"[{tag}] TEST @tuned  acc {test_tuned['acc']:.3f} | P {test_tuned['precision']:.3f} "
          f"| R {test_tuned['recall']:.3f} | F1 {test_tuned['f1']:.3f}")
    print(f"[{tag}] TEST PR-AUC  {pr_auc:.3f} (random baseline = positive rate {pos_rate:.3f})")

    return {
        "history": history,
        "test_metrics": test_default,        # kept for backward compat
        "test_default": test_default,
        "test_tuned": test_tuned,
        "tuned_threshold": best_t,
        "pr_auc": pr_auc,
        "pos_rate": pos_rate,
        "pos_weight": pw,
    }