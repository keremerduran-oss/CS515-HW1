import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from parameters import DataConfig, TrainingConfig
from train import get_transforms


@torch.no_grad()
def run_test(
    model: torch.nn.Module,
    data_cfg: DataConfig,
    train_cfg: TrainingConfig,
    device: torch.device,
    resize: bool = False,
) -> float:
    """Load the best saved model and evaluate it on the test set with per-class accuracy.

    Args:
        model: The neural network to evaluate.
        data_cfg: Dataset configuration.
        train_cfg: Training configuration (used for save_path, batch_size, num_workers).
        device: Device to run on.
        resize: Whether to resize images to 224x224 (Option 1 transfer learning).

    Returns:
        Overall test accuracy.
    """
    tf = get_transforms(data_cfg, train=False, resize=resize)

    if data_cfg.dataset == "mnist":
        test_ds = datasets.MNIST(data_cfg.data_dir,  train=False, download=True, transform=tf)
    else:
        test_ds = datasets.CIFAR10(data_cfg.data_dir, train=False, download=True, transform=tf)

    loader = DataLoader(
        test_ds,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=data_cfg.num_workers,
        pin_memory=True,
    )

    model.load_state_dict(torch.load(train_cfg.save_path, map_location=device))
    model.eval()

    correct, n = 0, 0
    class_correct: list[int] = [0] * data_cfg.num_classes
    class_total:   list[int] = [0] * data_cfg.num_classes

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs).argmax(1)
        correct += preds.eq(labels).sum().item()
        n       += imgs.size(0)
        for p, t in zip(preds, labels):
            class_correct[t] += (p == t).item()
            class_total[t]   += 1

    overall = correct / n
    print(f"\n=== Test Results ({train_cfg.run_name}) ===")
    print(f"Overall accuracy: {overall:.4f}  ({correct}/{n})\n")

    cifar_classes = ["airplane", "automobile", "bird", "cat", "deer",
                     "dog", "frog", "horse", "ship", "truck"]
    for i in range(data_cfg.num_classes):
        label = cifar_classes[i] if data_cfg.dataset == "cifar10" else str(i)
        acc   = class_correct[i] / class_total[i]
        print(f"  {label:>12}: {acc:.4f}  ({class_correct[i]}/{class_total[i]})")

    return overall
