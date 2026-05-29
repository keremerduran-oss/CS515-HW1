"""
parameters.py
=============
Centralised, typed configuration for CS515 HW4 Part 1 (Sequence Modeling / Financial Forecasting).

All hyper-parameters and experiment settings live here as frozen-ish dataclasses so that
every module (data, models, training, evaluation) shares one source of truth. This mirrors
the structure used in HW2/HW3.

Usage:
    from parameters import DataConfig, ModelConfig, TrainConfig, get_default_configs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


# --------------------------------------------------------------------------------------
# Data configuration
# --------------------------------------------------------------------------------------
@dataclass
class DataConfig:
    """Everything related to downloading, splitting and windowing the stock data."""

    # --- Ticker universe (5 diverse S&P 500 names; assignment requires >= 3) ---
    tickers: List[str] = field(
        default_factory=lambda: ["MSFT", "WMT", "PFE", "CVX", "BA"]
    )

    # --- Date ranges (chronological split, NO shuffling) ---
    start_date: str = "2020-01-01"
    end_date: str = "2025-12-31"

    train_end: str = "2024-07-31"      # train: 2020-01 .. 2024-07
    val_end: str = "2024-12-31"        # val:   2024-08 .. 2024-12
    # test: 2025-01 .. 2025-12 (everything after val_end)

    # --- Feature columns pulled from yfinance (the 4 OHLC prices) ---
    # F = 4 base features. Auxiliary features (e.g. MA via 1D conv) handled in the model,
    # so the effective input dim F_hat >= F.
    feature_cols: List[str] = field(
        default_factory=lambda: ["Open", "High", "Low", "Close"]
    )
    target_col: str = "Close"          # closing price drives the return ratio
    high_col: str = "High"             # used as the "max price" proxy in part (d)

    # --- Windowing ---
    lookback: int = 20                 # T = 20 (assignment-specified)
    horizons: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])  # d = 1..5 -> D = 5

    # --- Rolling-average target (part c) ---
    roll_window: int = 3               # l = 3
    # weights w_j for j = 0..l-1. Default = simple average; can be overridden.
    roll_weights: List[float] = field(default_factory=lambda: [1 / 3, 1 / 3, 1 / 3])

    # --- Turning-point detection (part d) ---
    buy_threshold: float = 1.1         # gamma; "buy" if (1 + return) >= 1.1 i.e. >=10% up

    # --- Normalisation ---
    # Per-ticker z-score on features, fit on the TRAIN portion only (no leakage).
    normalize: bool = True

    # --- Caching ---
    cache_path: str = "data/stock_data.csv"
    use_cache: bool = True

    # --- DataLoader ---
    batch_size: int = 64
    num_workers: int = 0


# --------------------------------------------------------------------------------------
# Model configuration
# --------------------------------------------------------------------------------------
@dataclass
class ModelConfig:
    """Architecture hyper-parameters shared by StockLSTM / StockGRU."""

    rnn_type: str = "lstm"             # "lstm" or "gru"
    input_size: int = 4               # F (base OHLC); raised to F_hat internally if MA conv on
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    output_size: int = 5              # D = 5 (one per horizon d=1..5)

    # Auxiliary feature: moving-average channel produced by a 1D conv over closing price.
    # When True the model appends conv-derived channels, so F_hat > F (footnote 1).
    use_ma_conv: bool = True
    ma_kernel: int = 5                # moving-average window for the 1D conv

    bidirectional: bool = False       # set True only for the part (d) detector


@dataclass
class DetectorConfig(ModelConfig):
    """Part (d): bi-directional recurrent classifier (buy / pass)."""

    bidirectional: bool = True
    output_size: int = 1              # single logit -> BCEWithLogits


# --------------------------------------------------------------------------------------
# Training configuration
# --------------------------------------------------------------------------------------
@dataclass
class TrainConfig:
    """Optimisation settings."""

    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-5
    optimizer: str = "adamw"          # "adam" or "adamw"
    grad_clip: float = 1.0            # clip RNN grads for stability
    seed: int = 42

    # Early stopping on validation loss
    patience: int = 15
    min_delta: float = 1e-6

    device: str = "cuda"              # falls back to cpu automatically if unavailable

    # Checkpointing
    ckpt_dir: str = "checkpoints"

    # Part (d) handles class imbalance with a positive-class weight in BCE.
    # None -> computed automatically from the training labels.
    pos_weight: float | None = None


def get_default_configs():
    """Convenience: return a (DataConfig, ModelConfig, TrainConfig) triple."""
    return DataConfig(), ModelConfig(), TrainConfig()
