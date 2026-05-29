"""
part2_model.py
==============
Neural communication system for CS515 HW4 Part 2.

Components
----------
  Encoder (TX): per-round transformer that maps (original symbols, history of
                coded symbols, history of feedback) → 4 new coded symbols.
  Decoder (RX): transformer that takes ALL received noisy symbols across T
                rounds and produces 4 × 8-way classification.
  Channel:      AWGN forward channel + noiseless feedback (simple relay).

Key design decisions (matching the hints in the assignment):
  - Hint 1: feedback = noisy received symbols relayed back (no neural encoder).
  - Hint 2: decoder runs ONCE at the end when all coded symbols are collected.
  - Hint 3: MLP before transformer (pre-processing) and after (post-processing
            to map to coded symbols).

The input sequence has 4 positions (one per symbol) and stays fixed at 4
across all rounds. At round t, each position's raw input is constructed by
concatenating: the original symbol embedding, all previously transmitted
coded symbols for that position, and all received feedback for that position.
A pre-processing MLP maps this growing raw input to d_model.

Power constraint is enforced by normalising coded symbols so that
||x^(t)||^2 = num_symbols * power_constraint (deterministic, satisfies the
expectation constraint exactly).
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn

from part2_parameters import CommConfig, CommModelConfig


# -----------------------------------------------------------------------
# Building blocks
# -----------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Standard pre-norm transformer block (Eqs 1-3 in the assignment)."""

    def __init__(self, d_model: int, nhead: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm multi-head self-attention + residual
        h = self.norm1(x)
        h = x + self.attn(h, h, h, need_weights=False)[0]
        # Pre-norm FFN + residual
        h = h + self.ff(self.norm2(h))
        return h


class TransformerStack(nn.Module):
    """Stack of L transformer blocks."""

    def __init__(self, d_model: int, nhead: int, d_ff: int, dropout: float, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, nhead, d_ff, dropout) for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# -----------------------------------------------------------------------
# Positional encoding (fixed sinusoidal, 4 positions)
# -----------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 16):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])  # handle odd d_model
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


# -----------------------------------------------------------------------
# Encoder (TX)
# -----------------------------------------------------------------------

class Encoder(nn.Module):
    """
    Per-round encoder.

    At round t (1-indexed), each of the 4 symbol positions has accumulated:
      - 1 symbol embedding (always present)
      - (t-1) previous coded symbols (scalars)
      - (t-1) feedback values (scalars)

    Raw input dim per position = d_emb + 2*(t-1).
    The pre-MLP handles variable input size by padding to a fixed max dim.
    """

    def __init__(self, cfg: CommConfig, mcfg: CommModelConfig):
        super().__init__()
        self.cfg = cfg
        self.mcfg = mcfg

        # Symbol embedding: {1..8} -> d_model
        # Symbols are 1-indexed; we use num_embeddings = alphabet_size + 1
        # and ignore index 0.
        self.symbol_emb = nn.Embedding(cfg.alphabet_size + 1, mcfg.d_model)

        # Max raw input dim per position at round T:
        #   d_model (embedding) + (T-1) coded + (T-1) feedback
        max_raw = mcfg.d_model + 2 * (cfg.num_rounds - 1)

        # Pre-processing MLP: raw input -> d_model
        self.pre_mlp = nn.Sequential(
            nn.Linear(max_raw, mcfg.d_model),
            nn.ReLU(),
            nn.Linear(mcfg.d_model, mcfg.d_model),
        )

        self.pe = PositionalEncoding(mcfg.d_model)
        self.transformer = TransformerStack(
            mcfg.d_model, mcfg.nhead, mcfg.d_ff, mcfg.dropout, mcfg.num_layers
        )

        # Post-processing MLP: d_model -> 1 coded symbol per position
        self.post_mlp = nn.Sequential(
            nn.Linear(mcfg.d_model, mcfg.d_ff),
            nn.ReLU(),
            nn.Linear(mcfg.d_ff, mcfg.d_coded),
        )

    def _build_input(
        self,
        symbols: torch.Tensor,       # (B, 4) LongTensor, values 1..8
        coded_history: list,          # list of (B, 4, d_coded) tensors, len = t-1
        feedback_history: list,       # list of (B, 4, d_coded) tensors, len = t-1
    ) -> torch.Tensor:
        """Assemble the raw input for round t."""
        B = symbols.size(0)
        S = self.cfg.num_symbols
        T = self.cfg.num_rounds
        d_coded = self.mcfg.d_coded
        d_model = self.mcfg.d_model

        # Symbol embeddings: (B, 4, d_model)
        emb = self.symbol_emb(symbols)

        # History features: pad to max length (T-1) coded + (T-1) feedback
        max_hist = T - 1
        coded_feats = torch.zeros(B, S, max_hist * d_coded, device=symbols.device)
        fb_feats = torch.zeros(B, S, max_hist * d_coded, device=symbols.device)

        for i, c in enumerate(coded_history):
            coded_feats[:, :, i * d_coded : (i + 1) * d_coded] = c
        for i, f in enumerate(feedback_history):
            fb_feats[:, :, i * d_coded : (i + 1) * d_coded] = f

        # Concatenate: (B, 4, d_model + 2*(T-1)*d_coded)
        raw = torch.cat([emb, coded_feats, fb_feats], dim=-1)
        return raw

    def forward(
        self,
        symbols: torch.Tensor,
        coded_history: list,
        feedback_history: list,
    ) -> torch.Tensor:
        """
        Produce coded symbols for one round.

        Returns: (B, 4, d_coded) power-normalised coded symbols.
        """
        raw = self._build_input(symbols, coded_history, feedback_history)
        z = self.pre_mlp(raw)                       # (B, 4, d_model)
        z = self.pe(z)                              # + positional encoding
        z = self.transformer(z)                     # (B, 4, d_model)
        x = self.post_mlp(z)                        # (B, 4, d_coded)

        # Power normalisation: scale so ||x||^2 = num_symbols * power_constraint
        # This is per-sample normalisation (each sample in the batch independently).
        target_power = self.cfg.num_symbols * self.cfg.power_constraint
        norm_sq = (x ** 2).sum(dim=(1, 2), keepdim=True).clamp(min=1e-8)
        x = x * (target_power / norm_sq).sqrt()
        return x


# -----------------------------------------------------------------------
# Decoder (RX)
# -----------------------------------------------------------------------

class Decoder(nn.Module):
    """
    Runs ONCE after all T rounds.

    Input: all T rounds of noisy received symbols -> (B, 4, T * d_coded).
    We project into d_model, run a transformer, then classify each of the
    4 positions into {1..8}.
    """

    def __init__(self, cfg: CommConfig, mcfg: CommModelConfig):
        super().__init__()
        self.cfg = cfg
        self.mcfg = mcfg

        input_dim = cfg.num_rounds * mcfg.d_coded  # T * d_coded per position

        self.pre_mlp = nn.Sequential(
            nn.Linear(input_dim, mcfg.d_model),
            nn.ReLU(),
            nn.Linear(mcfg.d_model, mcfg.d_model),
        )
        self.pe = PositionalEncoding(mcfg.d_model)
        self.transformer = TransformerStack(
            mcfg.d_model, mcfg.nhead, mcfg.d_ff, mcfg.dropout, mcfg.num_layers
        )
        self.classifier = nn.Linear(mcfg.d_model, cfg.alphabet_size)

    def forward(self, received_all: torch.Tensor) -> torch.Tensor:
        """
        received_all: (B, 4, T * d_coded) — all noisy receptions concatenated.
        Returns: (B, 4, alphabet_size) logits.
        """
        z = self.pre_mlp(received_all)
        z = self.pe(z)
        z = self.transformer(z)
        return self.classifier(z)


# -----------------------------------------------------------------------
# Full system
# -----------------------------------------------------------------------

class CommSystem(nn.Module):
    """
    End-to-end trainable interactive communication system.

    Forward pass:
      for t = 1..T:
        1. Encoder produces coded symbols x^(t)
        2. AWGN channel: y^(t) = x^(t) + noise
        3. Noiseless feedback: f^(t) = y^(t)  (Hint 1: simple relay)
      Decoder runs once on all received y^(1)..y^(T) -> classification.
    """

    def __init__(self, cfg: CommConfig, mcfg: CommModelConfig):
        super().__init__()
        self.cfg = cfg
        self.mcfg = mcfg
        self.encoder = Encoder(cfg, mcfg)
        self.decoder = Decoder(cfg, mcfg)

    def forward(self, symbols: torch.Tensor) -> torch.Tensor:
        """
        symbols: (B, 4) LongTensor, values in {1..8}.
        Returns: (B, 4, 8) logits for each symbol position.
        """
        B = symbols.size(0)
        T = self.cfg.num_rounds
        d_coded = self.mcfg.d_coded
        sigma = self.cfg.noise_var ** 0.5

        coded_history = []   # list of (B, 4, d_coded) — what TX sent
        feedback_history = []  # list of (B, 4, d_coded) — what TX got back
        received_list = []   # list of (B, 4, d_coded) — what RX received

        for t in range(T):
            # Encoder produces coded symbols
            x = self.encoder(symbols, coded_history, feedback_history)  # (B, 4, d_coded)

            # AWGN channel
            noise = torch.randn_like(x) * sigma
            y = x + noise  # (B, 4, d_coded)

            # Store
            received_list.append(y)
            coded_history.append(x)

            # Noiseless feedback: relay the noisy received signal back to TX
            # (Hint 1: no neural network needed for feedback)
            if t < T - 1:  # no feedback needed after the last round
                feedback_history.append(y)

        # Decoder: concatenate all received symbols and classify
        received_all = torch.cat(received_list, dim=-1)  # (B, 4, T * d_coded)
        logits = self.decoder(received_all)  # (B, 4, 8)
        return logits
