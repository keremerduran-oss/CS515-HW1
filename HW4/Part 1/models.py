"""
models.py
=========
Recurrent forecasting models for CS515 HW4 Part 1.

  - StockRNN      : unified LSTM/GRU regressor outputting D=5 horizon returns (parts b, c)
  - StockDetector : bi-directional LSTM/GRU classifier -> single buy/pass logit (part d)

Auxiliary feature (footnote 1 of the assignment)
------------------------------------------------
Optionally we prepend a depthwise-style 1D convolution over the closing-price channel to
produce a moving-average feature, raising the effective input dimension from F to F_hat.
This is a learnable smoothing filter (initialised to a uniform MA kernel).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from parameters import ModelConfig


class MovingAverageConv(nn.Module):
    """
    1D convolution over the closing-price channel that appends `out_channels` smoothed
    features to the input, giving F_hat = F + out_channels. Causal padding so no future
    leakage inside the window.
    """

    def __init__(self, close_idx: int, kernel: int = 5, out_channels: int = 1):
        super().__init__()
        self.close_idx = close_idx
        self.kernel = kernel
        self.pad = kernel - 1  # causal (left) padding
        self.conv = nn.Conv1d(
            in_channels=1, out_channels=out_channels, kernel_size=kernel, bias=False
        )
        # Initialise as a uniform moving average; the model can adapt it during training.
        with torch.no_grad():
            self.conv.weight.fill_(1.0 / kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F). Extract close channel -> (B, 1, T), conv, re-append.
        B, T, F = x.shape
        close = x[:, :, self.close_idx].unsqueeze(1)        # (B, 1, T)
        close = nn.functional.pad(close, (self.pad, 0))     # causal pad on the left
        ma = self.conv(close)                               # (B, out_ch, T)
        ma = ma.transpose(1, 2)                             # (B, T, out_ch)
        return torch.cat([x, ma], dim=-1)                   # (B, T, F + out_ch)


class StockRNN(nn.Module):
    """
    Unified LSTM / GRU regressor.

    Forward: (B, T, F) -> (B, D). Uses the final time-step hidden state through a
    dropout + linear head. Handles bidirectional internally (used by the detector).
    """

    def __init__(self, cfg: ModelConfig, close_idx: int = 3):
        super().__init__()
        self.cfg = cfg
        self.use_ma = cfg.use_ma_conv
        input_size = cfg.input_size

        if self.use_ma:
            self.ma = MovingAverageConv(close_idx, kernel=cfg.ma_kernel, out_channels=1)
            input_size = cfg.input_size + 1  # F_hat = F + 1

        rnn_cls = nn.LSTM if cfg.rnn_type.lower() == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            bidirectional=cfg.bidirectional,
        )
        directions = 2 if cfg.bidirectional else 1
        self.dropout = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(cfg.hidden_size * directions, cfg.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_ma:
            x = self.ma(x)
        out, _ = self.rnn(x)                 # out: (B, T, H * directions)
        last = out[:, -1, :]                 # final time-step representation
        last = self.dropout(last)
        return self.head(last)               # (B, output_size)


def build_model(cfg: ModelConfig, close_idx: int = 3) -> StockRNN:
    """Factory used by main.py so the same constructor serves LSTM, GRU and the detector."""
    return StockRNN(cfg, close_idx=close_idx)
