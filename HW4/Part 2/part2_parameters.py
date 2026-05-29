"""
part2_parameters.py
===================
Configuration for CS515 HW4 Part 2: Interactive Communication Protocol.

A transmitter and receiver (both transformer-based) learn to communicate
a 4-symbol message from alphabet {1..8} over T=4 AWGN rounds with noiseless
feedback.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class CommConfig:
    """System / channel parameters."""

    # Message space
    num_symbols: int = 4          # message length (4 symbols per message)
    alphabet_size: int = 8        # each symbol in {1, ..., 8}

    # Channel
    num_rounds: int = 4           # T = 4 communication rounds
    noise_var: float = 0.25       # sigma^2 = 0.25 (SNR = 4 = 6 dB)

    # Power constraint: E[||x||^2] <= power_constraint per round
    power_constraint: float = 1.0


@dataclass
class CommModelConfig:
    """Transformer architecture for encoder and decoder."""

    d_model: int = 64             # internal representation dimension
    nhead: int = 4                # multi-head attention heads
    num_layers: int = 2           # transformer blocks
    d_ff: int = 128               # feed-forward hidden dim
    dropout: float = 0.1

    # Coded symbol dimension (per symbol, per round)
    d_coded: int = 1              # each coded symbol is a scalar -> x^(t) in R^4


@dataclass
class CommTrainConfig:
    """Training hyper-parameters."""

    epochs: int = 200
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-5
    seed: int = 42
    device: str = "cuda"

    # Evaluation
    eval_batches: int = 100       # number of batches for BER / SER evaluation
    log_every: int = 10           # print every N epochs
