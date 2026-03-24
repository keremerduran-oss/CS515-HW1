import copy
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Tuple

from parameters import DataConfig, ModelConfig, TrainingConfig
from results_logger import log_result

os.makedirs("plots_v2", exist_ok=True)
os.makedirs("models/saved_v2", exist_ok=True)


def get_transforms(data_cfg: DataConfig, train: bool = True) -> transforms.Compose:
    """Return the appropriate transform pipeline for the given dataset and split."""
    mean, std = data_cfg.mean, data_cfg.std

    if data_cfg.dataset == "mnist":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def get_loaders(
    data_cfg: DataConfig,
    train_cfg: TrainingConfig,
) -> Tuple[DataLoader, DataLoader]:
    """Build and return train and validation DataLoaders."""
    train_tf = get_transforms(data_cfg, train=True)
    val_tf   = get_transforms(data_cfg, train=False)

    if data_cfg.dataset == "mnist":
        train_ds = datasets.MNIST(data_cfg.data_dir, train=True,  download=True, transform=train_tf)
        val_ds   = datasets.MNIST(data_cfg.data_dir, train=False, download=True, transform=val_tf)
    else:
        train_ds = datasets.CIFAR10(data_cfg.data_dir, train=True,  download=True, transform=train_tf)
        val_ds   = datasets.CIFAR10(data_cfg.data_dir, train=False, download=True, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size,
                              shuffle=True,  num_workers=data_cfg.num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=train_cfg.batch_size,
                              shuffle=False, num_workers=data_cfg.num_workers)
    return train_loader, val_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    train_cfg: TrainingConfig,
) -> tuple[float, float]:
    """Run one training epoch and return (avg_loss, accuracy)."""
    model.train()
    total_loss, correct, n = 0.0, 0, 0

    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)

        if train_cfg.reg_type == "l1":
            l1_penalty = sum(p.abs().sum() for p in model.parameters())
            loss = loss + train_cfg.reg_lambda * l1_penalty

        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item() * imgs.size(0)
        correct    += out.argmax(1).eq(labels).sum().item()
        n          += imgs.size(0)

        if (batch_idx + 1) % train_cfg.log_interval == 0:
            print(f"  [{batch_idx+1}/{len(loader)}] "
                  f"loss: {total_loss/n:.4f}  acc: {correct/n:.4f}")

    return total_loss / n, correct / n


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate the model on the validation set and return (avg_loss, accuracy)."""
    model.eval()
    total_loss, correct, n = 0.0, 0, 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out  = model(imgs)
            loss = criterion(out, labels)
            total_loss += loss.detach().item() * imgs.size(0)
            correct    += out.argmax(1).eq(labels).sum().item()
            n          += imgs.size(0)

    return total_loss / n, correct / n


def run_training(
    model: nn.Module,
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
    train_cfg: TrainingConfig,
    device: torch.device,
) -> None:
    """Full training loop with early stopping, LR scheduling, and loss curve saving."""
    train_loader, val_loader = get_loaders(data_cfg, train_cfg)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr           = train_cfg.learning_rate,
        weight_decay = train_cfg.weight_decay,
    )

    if train_cfg.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg.epochs)
    elif train_cfg.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_acc          = 0.0
    best_weights      = None
    patience          = 5
    epochs_no_improve = 0
    train_losses: list[float] = []
    val_losses:   list[float] = []

    for epoch in range(1, train_cfg.epochs + 1):
        print(f"\nEpoch {epoch}/{train_cfg.epochs}")
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, train_cfg
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        if train_cfg.scheduler == "plateau":
            scheduler.step(val_loss)
        else:
            scheduler.step()

        train_losses.append(tr_loss)
        val_losses.append(val_loss)

        print(f"  Train loss: {tr_loss:.4f}  acc: {tr_acc:.4f}")
        print(f"  Val   loss: {val_loss:.4f}  acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc          = val_acc
            best_weights      = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            torch.save(best_weights, train_cfg.save_path)
            print(f"  Saved best model (val_acc={best_acc:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve}/{patience} epochs")
            if epochs_no_improve >= patience:
                print("  Early stopping triggered!")
                break

    model.load_state_dict(best_weights)
    print(f"\nTraining done. Best val accuracy: {best_acc:.4f}")

    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses,   label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.savefig(f"plots_v2/loss_curve_{train_cfg.run_name}.png")
    plt.show()
    print(f"Loss curve saved to plots_v2/loss_curve_{train_cfg.run_name}.png")

    log_result("results_v2.csv", train_cfg, data_cfg, model_cfg, best_val_acc=best_acc, test_acc=best_acc)
