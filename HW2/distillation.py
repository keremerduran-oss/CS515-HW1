"""
distillation.py — Knowledge Distillation for CIFAR-10

Implements:
  - KDLoss: combined soft (teacher) + hard (true label) loss
  - SoftTargetLoss: custom variant where only the true-class probability
      comes from the teacher; all other classes share the remainder equally
  - run_distillation: full training loop for student under a frozen teacher
  - count_flops: wrapper around ptflops for FLOPs comparison
"""

import copy
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from typing import Tuple

from parameters import DataConfig, ModelConfig, TrainingConfig
from train import get_loaders, build_criterion, build_scheduler
from results_logger import log_result


# ── Loss functions ────────────────────────────────────────────────────────────

class KDLoss(nn.Module):
    """Knowledge distillation loss combining soft and hard targets.

    Soft loss: KL divergence between student and teacher outputs after
    temperature scaling. Captures inter-class similarity information.

    Hard loss: Standard cross entropy against the true one-hot label.

    Final loss = alpha * soft_loss + (1 - alpha) * hard_loss

    Args:
        temperature: Softmax temperature T. Higher T produces softer
            probability distributions, amplifying signals from non-target
            classes. Typical values: 3-7.
        alpha: Weight of the soft (distillation) loss. (1 - alpha) is
            applied to the hard loss. Typical values: 0.5-0.9.
    """

    def __init__(self, temperature: float = 4.0, alpha: float = 0.7) -> None:
        super().__init__()
        self.T     = temperature
        self.alpha = alpha

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the combined KD loss.

        Args:
            student_logits: Raw logits from the student model. Shape: (N, C).
            teacher_logits: Raw logits from the frozen teacher. Shape: (N, C).
            labels: Ground truth class indices. Shape: (N,).

        Returns:
            Scalar loss tensor.
        """
        # Soft loss — scale by T^2 to keep gradient magnitudes consistent
        # across different temperature values (standard practice from Hinton et al.)
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / self.T, dim=1),
            F.softmax(teacher_logits   / self.T, dim=1),
            reduction="batchmean",
        ) * (self.T ** 2)

        # Hard loss
        hard_loss = F.cross_entropy(student_logits, labels)

        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss


class SoftTargetLoss(nn.Module):
    """Custom distillation loss using teacher confidence on the true class only.

    Instead of using the full soft distribution from the teacher, this loss
    assigns the teacher's probability for the true class to that class, and
    distributes the remaining probability equally across all other classes.

    Example (10 classes, teacher is 70% confident the image is a cat):
        target = [0.033, 0.033, 0.033, 0.70, 0.033, 0.033, 0.033, 0.033, 0.033, 0.033]

    This acts as a form of dynamic label smoothing where the smoothing amount
    is determined per-example by how confident the teacher is, rather than
    being a fixed epsilon. Harder examples (low teacher confidence) get more
    smoothing; easy examples (high teacher confidence) get less.

    Args:
        temperature: Temperature for extracting teacher probabilities.
    """

    def __init__(self, temperature: float = 4.0) -> None:
        super().__init__()
        self.T = temperature

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the soft target loss.

        Args:
            student_logits: Raw logits from the student model. Shape: (N, C).
            teacher_logits: Raw logits from the frozen teacher. Shape: (N, C).
            labels: Ground truth class indices. Shape: (N,).

        Returns:
            Scalar loss tensor.
        """
        N, C = student_logits.shape

        # Get teacher's probability for the true class per sample
        teacher_probs  = F.softmax(teacher_logits / self.T, dim=1)  # (N, C)
        true_class_prob = teacher_probs[torch.arange(N), labels]     # (N,)

        # Build soft target distribution:
        # true class gets teacher's confidence, rest share (1 - confidence) equally
        targets = torch.full((N, C), 0.0, device=student_logits.device)
        remainder_per_class = (1.0 - true_class_prob) / (C - 1)      # (N,)
        targets += remainder_per_class.unsqueeze(1)                   # broadcast
        targets[torch.arange(N), labels] = true_class_prob            # overwrite true class

        # Cross entropy against the soft targets
        log_probs = F.log_softmax(student_logits, dim=1)
        loss = -(targets * log_probs).sum(dim=1).mean()
        return loss


# ── FLOPs counter ─────────────────────────────────────────────────────────────

def count_flops(model: nn.Module, input_size: Tuple[int, int, int] = (3, 32, 32)) -> None:
    """Print GMACs and parameter count for a model using ptflops.

    Args:
        model: The PyTorch model to analyse.
        input_size: Input tensor shape excluding batch dimension (C, H, W).
    """
    try:
        from ptflops import get_model_complexity_info
        macs, params = get_model_complexity_info(
            model, input_size,
            as_strings=True,
            print_per_layer_stat=False,
            verbose=False,
        )
        print(f"  GMACs : {macs}")
        print(f"  Params: {params}")
    except ImportError:
        print("  ptflops not installed. Run: pip install ptflops")


# ── Training loop ─────────────────────────────────────────────────────────────

def _train_one_epoch_kd(
    student:   nn.Module,
    teacher:   nn.Module,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device:    torch.device,
    log_interval: int,
) -> Tuple[float, float]:
    """Run one distillation training epoch.

    Args:
        student: Student model being trained.
        teacher: Frozen teacher model providing soft targets.
        loader: Training DataLoader.
        optimizer: Optimizer for the student.
        criterion: KDLoss or SoftTargetLoss instance.
        device: Device to run on.
        log_interval: How often to print batch-level progress.

    Returns:
        Tuple of (average loss, accuracy) over the epoch.
    """
    student.train()
    teacher.eval()
    total_loss, correct, n = 0.0, 0, 0

    with torch.no_grad():
        # We'll get teacher logits inside the loop but no_grad only for teacher
        pass

    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)

        with torch.no_grad():
            teacher_logits = teacher(imgs)

        optimizer.zero_grad()
        student_logits = student(imgs)
        loss = criterion(student_logits, teacher_logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item() * imgs.size(0)
        correct    += student_logits.argmax(1).eq(labels).sum().item()
        n          += imgs.size(0)

        if (batch_idx + 1) % log_interval == 0:
            print(f"  [{batch_idx+1}/{len(loader)}] "
                  f"loss: {total_loss/n:.4f}  acc: {correct/n:.4f}")

    return total_loss / n, correct / n


def _validate_student(
    student:   nn.Module,
    loader:    DataLoader,
    device:    torch.device,
) -> Tuple[float, float]:
    """Evaluate student on the validation set using standard cross entropy.

    Args:
        student: Student model to evaluate.
        loader: Validation DataLoader.
        device: Device to run on.

    Returns:
        Tuple of (average cross entropy loss, accuracy).
    """
    student.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, n = 0.0, 0, 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out  = student(imgs)
            loss = criterion(out, labels)
            total_loss += loss.detach().item() * imgs.size(0)
            correct    += out.argmax(1).eq(labels).sum().item()
            n          += imgs.size(0)

    return total_loss / n, correct / n


def run_distillation(
    student:    nn.Module,
    teacher:    nn.Module,
    data_cfg:   DataConfig,
    model_cfg:  ModelConfig,
    train_cfg:  TrainingConfig,
    device:     torch.device,
) -> float:
    """Full knowledge distillation training loop.

    Trains the student model under guidance of the frozen teacher.
    Selects KDLoss for standard distillation or SoftTargetLoss for the
    custom teacher-confidence variant based on train_cfg.distill_mode.

    Args:
        student: Student model to train (SimpleCNN or MobileNetV2).
        teacher: Pre-trained teacher model (ResNet-18), will be frozen.
        data_cfg: Dataset configuration.
        model_cfg: Model configuration (for logging).
        train_cfg: Training hyperparameters including temperature and alpha.
        device: Device to run on.

    Returns:
        Best validation accuracy achieved during training.
    """
    os.makedirs("models/saved", exist_ok=True)

    # Freeze teacher completely
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    train_loader, val_loader = get_loaders(data_cfg, train_cfg)

    # Select loss function based on distill_mode
    if train_cfg.distill_mode == "soft_targets_only":
        criterion = SoftTargetLoss(temperature=train_cfg.temperature)
        print(f"Using SoftTargetLoss  (T={train_cfg.temperature})")
    else:
        criterion = KDLoss(temperature=train_cfg.temperature, alpha=train_cfg.alpha)
        print(f"Using KDLoss  (T={train_cfg.temperature}, alpha={train_cfg.alpha})")

    optimizer = torch.optim.Adam(
        student.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )
    scheduler = build_scheduler(optimizer, train_cfg)

    best_acc     = 0.0
    best_weights = None
    patience     = 10
    no_improve   = 0
    train_losses: list[float] = []
    val_losses:   list[float] = []

    for epoch in range(1, train_cfg.epochs + 1):
        print(f"\nEpoch {epoch}/{train_cfg.epochs}")

        tr_loss, tr_acc   = _train_one_epoch_kd(
            student, teacher, train_loader, optimizer, criterion, device, train_cfg.log_interval
        )
        val_loss, val_acc = _validate_student(student, val_loader, device)

        if train_cfg.scheduler == "plateau":
            scheduler.step(val_loss)
        else:
            scheduler.step()

        train_losses.append(tr_loss)
        val_losses.append(val_loss)

        print(f"  Train loss: {tr_loss:.4f}  acc: {tr_acc:.4f}")
        print(f"  Val   loss: {val_loss:.4f}  acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc     = val_acc
            best_weights = copy.deepcopy(student.state_dict())
            no_improve   = 0
            torch.save(best_weights, train_cfg.save_path)
            print(f"  Saved best student (val_acc={best_acc:.4f})")
        else:
            no_improve += 1
            print(f"  No improvement {no_improve}/{patience}")
            if no_improve >= patience:
                print("  Early stopping triggered!")
                break

    student.load_state_dict(best_weights)
    print(f"\nDistillation done. Best val accuracy: {best_acc:.4f}")

    # ── Loss curve ────────────────────────────────────────────────────────────
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses,   label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Distillation Loss — {train_cfg.run_name}")
    plt.legend()
    plt.savefig(f"plots/loss_{train_cfg.run_name}.png")
    plt.close()
    print(f"Loss curve saved to plots/loss_{train_cfg.run_name}.png")

    log_result("results.csv", train_cfg, data_cfg, model_cfg,
               best_val_acc=best_acc, test_acc=best_acc)

    return best_acc
