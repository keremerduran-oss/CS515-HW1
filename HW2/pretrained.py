"""
pretrained.py — Transfer Learning with ResNet-18 on CIFAR-10

Option 1: Resize CIFAR-10 images to 224×224, freeze all pretrained layers,
          replace and train only the final FC head.

Option 2: Keep images at 32×32, replace the aggressive first conv (7×7 stride 2)
          with a CIFAR-friendly one (3×3 stride 1), unfreeze early layers + head
          and fine-tune them together.
"""

import os
import torch
import torch.nn as nn
from torchvision import models
from parameters import DataConfig, ModelConfig, TrainingConfig
from train import run_training
from test import run_test


# ── Model builders ────────────────────────────────────────────────────────────

def build_option1(num_classes: int = 10) -> nn.Module:
    """Load pretrained ResNet-18 and prepare it for Option 1 transfer learning.

    Freezes all pretrained layers and replaces the final FC layer with a
    new linear head mapping to num_classes. Images must be resized to 224×224
    before being fed to this model.

    Args:
        num_classes: Number of output classes (10 for CIFAR-10).

    Returns:
        Modified ResNet-18 with all layers frozen except the new FC head.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Freeze all pretrained weights
    for param in model.parameters():
        param.requires_grad = False

    # Replace FC head — this is the only part that will be trained
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    # model.fc parameters have requires_grad=True by default

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[Option 1] Trainable params: {trainable:,} / {total:,}  "
          f"({100*trainable/total:.1f}%)")
    return model


def build_option2(num_classes: int = 10) -> nn.Module:
    """Load pretrained ResNet-18 and prepare it for Option 2 transfer learning.

    Replaces the first conv layer (designed for 224×224) with a CIFAR-friendly
    3×3 conv with stride 1 and removes the aggressive max-pool, so the model
    can accept 32×32 inputs without destroying spatial information.
    Unfreezes the first layer and the FC head for fine-tuning while keeping
    layers 2-4 frozen.

    Args:
        num_classes: Number of output classes (10 for CIFAR-10).

    Returns:
        Modified ResNet-18 suitable for 32×32 inputs with partial fine-tuning.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Replace the first conv: 7×7 stride 2 → 3×3 stride 1
    # The original was designed for 224×224; on 32×32 it would halve resolution
    # twice (conv stride 2 + maxpool stride 2), leaving only 8×8 feature maps.
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # conv1 is new so its params have requires_grad=True by default

    # Remove the max-pool that follows conv1 — not needed for small images
    model.maxpool = nn.Identity()

    # Unfreeze layer1 (first residual block) for fine-tuning
    for param in model.layer1.parameters():
        param.requires_grad = True

    # Replace and unfreeze the FC head
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    # model.fc params have requires_grad=True by default

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[Option 2] Trainable params: {trainable:,} / {total:,}  "
          f"({100*trainable/total:.1f}%)")
    return model


# ── Entry point ───────────────────────────────────────────────────────────────

def run_transfer_learning(
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
    train_cfg: TrainingConfig,
    device: torch.device,
) -> None:
    """Build the appropriate transfer learning model and run training + testing.

    Dispatches to Option 1 (freeze + resize) or Option 2 (modify conv + fine-tune)
    based on train_cfg.transfer_option.

    Args:
        data_cfg: Dataset configuration.
        model_cfg: Model configuration (used for logging only in this path).
        train_cfg: Training configuration including transfer_option.
        device: Device to run on.
    """
    os.makedirs("models/saved", exist_ok=True)

    option = train_cfg.transfer_option
    print(f"\n{'='*60}")
    print(f"Transfer Learning — {option.upper()}")
    print(f"{'='*60}")

    if option == "option1":
        model  = build_option1(num_classes=data_cfg.num_classes).to(device)
        resize = True   # images will be upscaled 32→224 in the data pipeline
    elif option == "option2":
        model  = build_option2(num_classes=data_cfg.num_classes).to(device)
        resize = False  # images stay at 32×32
    else:
        raise ValueError(f"Unknown transfer_option: '{option}'. Choose 'option1' or 'option2'.")

    if train_cfg.mode in ("train", "both"):
        run_training(model, data_cfg, model_cfg, train_cfg, device, resize=resize)

    if train_cfg.mode in ("test", "both"):
        run_test(model, data_cfg, train_cfg, device, resize=resize)
