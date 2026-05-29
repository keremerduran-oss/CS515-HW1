"""
part2_train.py
==============
Training and evaluation for the neural communication system (CS515 HW4 Part 2).

Training
--------
- Random messages are generated on-the-fly (no fixed dataset — the message
  space is 8^4 = 4096 but we train on random samples per batch).
- Loss: cross-entropy summed over the 4 symbol positions.
- Gradients flow through the AWGN channel naturally (additive noise,
  reparameterisation trick is implicit).

Evaluation metrics
------------------
- SER (Symbol Error Rate): fraction of individual symbols decoded incorrectly.
- BER (Bit Error Rate): fraction of bits wrong (each symbol = 3 bits since
  log2(8) = 3; we compare bit-level XOR).
- Block Error Rate: fraction of entire messages (all 4 symbols) decoded wrong.
"""

from __future__ import annotations

import os
import random
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from part2_parameters import CommConfig, CommModelConfig, CommTrainConfig
from part2_model import CommSystem


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def sample_messages(batch_size: int, cfg: CommConfig, device: torch.device) -> torch.Tensor:
    """Generate random messages: (B, num_symbols) with values in {1..alphabet_size}."""
    return torch.randint(1, cfg.alphabet_size + 1, (batch_size, cfg.num_symbols), device=device)


@torch.no_grad()
def evaluate(model: CommSystem, cfg: CommConfig, tcfg: CommTrainConfig, device: torch.device) -> Dict[str, float]:
    """Compute SER, BER, and block error rate over eval_batches random messages."""
    model.eval()
    total_symbols = 0
    total_bits = 0
    sym_errors = 0
    bit_errors = 0
    block_errors = 0
    total_blocks = 0
    bits_per_symbol = int(np.log2(cfg.alphabet_size))  # = 3

    for _ in range(tcfg.eval_batches):
        m = sample_messages(tcfg.batch_size, cfg, device)
        logits = model(m)                              # (B, 4, 8)
        pred = logits.argmax(dim=-1) + 1               # back to 1-indexed

        # Symbol errors
        sym_err = (pred != m)
        sym_errors += sym_err.sum().item()
        total_symbols += m.numel()

        # Bit errors (XOR the 3-bit representations)
        # Convert to 0-indexed for bitwise ops
        m0 = (m - 1).cpu().numpy().astype(np.uint8)
        p0 = (pred - 1).cpu().numpy().astype(np.uint8)
        xor = np.bitwise_xor(m0, p0)
        bit_errors += sum(bin(v).count("1") for v in xor.ravel())
        total_bits += m.numel() * bits_per_symbol

        # Block errors (any symbol wrong -> whole message wrong)
        block_errors += sym_err.any(dim=1).sum().item()
        total_blocks += m.size(0)

    model.train()
    return {
        "SER": sym_errors / max(total_symbols, 1),
        "BER": bit_errors / max(total_bits, 1),
        "BLER": block_errors / max(total_blocks, 1),
    }


def train(cfg: CommConfig, mcfg: CommModelConfig, tcfg: CommTrainConfig) -> Dict:
    """End-to-end training loop."""
    device = resolve_device(tcfg.device)
    set_seed(tcfg.seed)

    model = CommSystem(cfg, mcfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tcfg.epochs)
    criterion = nn.CrossEntropyLoss()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[comm] model parameters: {n_params:,}")
    print(f"[comm] channel SNR = {1.0 / cfg.noise_var:.1f} ({10 * np.log10(1.0 / cfg.noise_var):.1f} dB)")

    history = {"loss": [], "SER": [], "BER": [], "BLER": []}
    best_ser = 1.0
    best_state = None

    for ep in range(1, tcfg.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 50  # inner loop batches per epoch

        for _ in range(n_batches):
            m = sample_messages(tcfg.batch_size, cfg, device)
            logits = model(m)  # (B, 4, 8)

            # CE loss: logits are (B, 4, 8), targets are (B, 4) in {1..8}
            # nn.CrossEntropyLoss expects (B, C) so we reshape.
            loss = criterion(logits.reshape(-1, cfg.alphabet_size), (m - 1).reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / n_batches
        history["loss"].append(avg_loss)

        # Evaluate periodically
        if ep % tcfg.log_every == 0 or ep == 1:
            metrics = evaluate(model, cfg, tcfg, device)
            history["SER"].append(metrics["SER"])
            history["BER"].append(metrics["BER"])
            history["BLER"].append(metrics["BLER"])
            print(f"[comm] epoch {ep:4d} | loss {avg_loss:.4f} | "
                  f"SER {metrics['SER']:.4f} | BER {metrics['BER']:.4f} | "
                  f"BLER {metrics['BLER']:.4f}")
            if metrics["SER"] < best_ser:
                best_ser = metrics["SER"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation
    final = evaluate(model, cfg, tcfg, device)
    print(f"\n[comm] FINAL  SER {final['SER']:.4f} | BER {final['BER']:.4f} | "
          f"BLER {final['BLER']:.4f}")

    # Save checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "cfg": cfg, "mcfg": mcfg},
               "checkpoints/comm_system.pth")

    return {"history": history, "final": final, "n_params": n_params}
