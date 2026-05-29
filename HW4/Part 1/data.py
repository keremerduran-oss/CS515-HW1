"""
data.py
=======
Data pipeline for CS515 HW4 Part 1.

Responsibilities
----------------
1. Download daily OHLC data for the chosen tickers via yfinance (cached to parquet).
2. Chronological train / val / test split (NO shuffling -- it's a time series).
3. Per-ticker, leak-free z-score normalisation (statistics fit on TRAIN only).
4. Sliding-window construction for three target types:
     - "return"  : raw d-day return ratio          (part b)
     - "rolling" : weighted rolling-average return  (part c)
     - "signal"  : binary buy/pass label            (part d, uses max/high price)

Design note on pooling
-----------------------
We train a SINGLE shared model across ALL tickers (the f_theta: R^{N x T x F} -> R^{N x D}
formulation in the assignment). Windows from every ticker are pooled into one dataset.
Normalisation is per-ticker so that each stock's scale is handled independently before
pooling. >>> This pooled-single-model choice must be stated explicitly in the report. <<<
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from parameters import DataConfig


# --------------------------------------------------------------------------------------
# Download / cache
# --------------------------------------------------------------------------------------
def download_data(cfg: DataConfig) -> pd.DataFrame:
    """
    Download OHLC for all tickers and return a tidy long DataFrame:
        columns = [Date, Ticker, Open, High, Low, Close]
    Cached to parquet so the (slow, sometimes rate-limited) yfinance call runs once.
    """
    if cfg.use_cache and os.path.exists(cfg.cache_path):
        print(f"[data] loading cached data from {cfg.cache_path}")
        return pd.read_csv(cfg.cache_path, parse_dates=["Date"])

    import yfinance as yf  # imported lazily so the rest of the code runs without network

    frames = []
    for tk in cfg.tickers:
        print(f"[data] downloading {tk} ...")
        raw = yf.download(
            tk,
            start=cfg.start_date,
            end=cfg.end_date,
            progress=False,
            auto_adjust=False,
        )
        if raw.empty:
            raise RuntimeError(f"No data returned for {tk}. Check ticker / network.")
        # yfinance returns a MultiIndex column frame for a single ticker; flatten it.
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        # Bring the datetime index out as a real column FIRST, then standardise its name.
        raw = raw.reset_index()
        # The index column may be called "Date", "Datetime", or "index" across versions.
        rename_map = {"Datetime": "Date", "index": "Date", "Date": "Date"}
        raw = raw.rename(columns={c: rename_map.get(c, c) for c in raw.columns})
        if "Date" not in raw.columns:
            # fallback: first column is the datetime index
            raw = raw.rename(columns={raw.columns[0]: "Date"})
        raw = raw[["Date", "Open", "High", "Low", "Close"]].copy()
        raw["Ticker"] = tk
        frames.append(raw)

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(cfg.cache_path), exist_ok=True)
    df.to_csv(cfg.cache_path, index=False)
    print(f"[data] cached {len(df)} rows to {cfg.cache_path}")
    return df


# --------------------------------------------------------------------------------------
# Split + normalise
# --------------------------------------------------------------------------------------
def split_by_date(
    df_ticker: pd.DataFrame, cfg: DataConfig
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological split for a single ticker's frame (already date-sorted)."""
    d = pd.to_datetime(df_ticker["Date"])
    train = df_ticker[d <= cfg.train_end]
    val = df_ticker[(d > cfg.train_end) & (d <= cfg.val_end)]
    test = df_ticker[d > cfg.val_end]
    return train, val, test


def fit_normaliser(train_df: pd.DataFrame, cfg: DataConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Mean/std of feature columns on the TRAIN slice only (avoids look-ahead leakage)."""
    feats = train_df[cfg.feature_cols].to_numpy(dtype=np.float64)
    mu = feats.mean(axis=0)
    sigma = feats.std(axis=0) + 1e-8
    return mu, sigma


# --------------------------------------------------------------------------------------
# Window construction
# --------------------------------------------------------------------------------------
def _build_windows_for_split(
    prices: np.ndarray,        # (L, F) RAW (un-normalised) feature matrix for this split
    high: np.ndarray,          # (L,)   raw high price (for part d)
    norm_feats: np.ndarray,    # (L, F) normalised features used as model input
    cfg: DataConfig,
    target_type: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Slide a window of length T over one contiguous price series and emit (X, y) pairs.

    X : (n, T, F)   normalised input window
    y : depends on target_type:
          "return"  -> (n, D)  raw d-day return ratios for d in horizons
          "rolling" -> (n, D)  weighted rolling-avg return ratios
          "signal"  -> (n, 1)  binary buy label (1.0 if any horizon return >= gamma-1)

    Returns are computed from RAW prices (economically meaningful), while the model INPUT
    uses normalised features. The close price column index is located via feature_cols.
    """
    T = cfg.lookback
    horizons = cfg.horizons
    max_h = max(horizons)
    close_idx = cfg.feature_cols.index(cfg.target_col)

    close = prices[:, close_idx]                      # raw close, (L,)
    L = len(close)

    X_list, y_list = [], []

    # t indexes the LAST day in the lookback window (0-based). We need:
    #   window  = days [t-T+1 .. t]   -> requires t >= T-1
    #   targets = days [t+1 .. t+max_h] -> requires t + max_h < L
    for t in range(T - 1, L - max_h):
        window = norm_feats[t - T + 1 : t + 1]        # (T, F)
        p_t = close[t]                                # anchor price p^t_i

        if target_type == "return":
            y = np.array(
                [(close[t + d] - p_t) / p_t for d in horizons], dtype=np.float32
            )

        elif target_type == "rolling":
            # weighted rolling average of prices over window l, ending at t+d
            l = cfg.roll_window
            w = np.asarray(cfg.roll_weights, dtype=np.float64)
            assert len(w) == l, "roll_weights length must equal roll_window"
            y_vals = []
            for d in horizons:
                # average of p^{t+d}, p^{t+d-1}, ..., p^{t+d-l+1}
                idxs = [t + d - j for j in range(l)]
                avg = float(np.dot(w, close[idxs]))
                y_vals.append((avg - p_t) / p_t)
            y = np.array(y_vals, dtype=np.float32)

        elif target_type == "signal":
            # part (d): use MAX/high price in the numerator; buy if ANY horizon clears gamma
            buy = 0.0
            for d in horizons:
                ratio = high[t + d] / p_t             # pmax_{t+d} / p_t
                if ratio >= cfg.buy_threshold:        # gamma = 1.1  -> 10% up
                    buy = 1.0
                    break
            y = np.array([buy], dtype=np.float32)

        else:
            raise ValueError(f"unknown target_type {target_type}")

        X_list.append(window.astype(np.float32))
        y_list.append(y)

    if not X_list:
        return (
            np.zeros((0, T, prices.shape[1]), dtype=np.float32),
            np.zeros((0, len(horizons) if target_type != "signal" else 1), dtype=np.float32),
        )
    return np.stack(X_list), np.stack(y_list)


def build_dataset(
    cfg: DataConfig, target_type: str = "return"
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Full pipeline -> pooled windows across all tickers, keyed by split.

    Returns {"train": (X, y), "val": (X, y), "test": (X, y)} with windows from every
    ticker concatenated (pooled single model).
    """
    df = download_data(cfg)
    splits = {"train": ([], []), "val": ([], []), "test": ([], [])}

    for tk in cfg.tickers:
        dft = df[df["Ticker"] == tk].sort_values("Date").reset_index(drop=True)
        train_df, val_df, test_df = split_by_date(dft, cfg)
        if cfg.normalize:
            mu, sigma = fit_normaliser(train_df, cfg)
        else:
            mu, sigma = 0.0, 1.0

        for name, sdf in [("train", train_df), ("val", val_df), ("test", test_df)]:
            if len(sdf) <= cfg.lookback + max(cfg.horizons):
                continue
            raw_feats = sdf[cfg.feature_cols].to_numpy(dtype=np.float64)
            high = sdf[cfg.high_col].to_numpy(dtype=np.float64)
            norm_feats = (raw_feats - mu) / sigma
            X, y = _build_windows_for_split(
                raw_feats, high, norm_feats, cfg, target_type
            )
            if len(X):
                splits[name][0].append(X)
                splits[name][1].append(y)

    out = {}
    for name, (xs, ys) in splits.items():
        if xs:
            out[name] = (np.concatenate(xs), np.concatenate(ys))
        else:
            out[name] = (
                np.zeros((0, cfg.lookback, len(cfg.feature_cols)), np.float32),
                np.zeros((0, 1), np.float32),
            )
    return out


# --------------------------------------------------------------------------------------
# Torch Dataset / DataLoader
# --------------------------------------------------------------------------------------
class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i: int):
        return self.X[i], self.y[i]


def make_loaders(
    data: Dict[str, Tuple[np.ndarray, np.ndarray]], cfg: DataConfig
) -> Dict[str, DataLoader]:
    """Build DataLoaders. Train is shuffled at the WINDOW level (fine: each window is an
    independent sample; the temporal split already prevents look-ahead leakage)."""
    loaders = {}
    for name in ["train", "val", "test"]:
        X, y = data[name]
        ds = WindowDataset(X, y)
        loaders[name] = DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=(name == "train"),
            num_workers=cfg.num_workers,
            drop_last=False,
        )
    return loaders