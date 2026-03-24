"""
visualize.py — t-SNE feature visualization for trained models.

Extracts penultimate-layer features from any trained model using a forward
hook, then projects them to 2D with t-SNE for visualization.

Usage:
    python visualize.py --model resnet --save_path models/saved/resnet_scratch_ls.pth
    python visualize.py --model cnn --save_path models/saved/cnn_baseline_50.pth
    python visualize.py --model cnn --save_path models/saved/cnn_distilled_v2.pth
    python visualize.py --model mobilenet --save_path models/saved/mobilenet_softtargets.pth
"""

import argparse
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Tuple

from models.CNN import SimpleCNN
from models.ResNet import ResNet, BasicBlock
from models.MobileNet import MobileNetV2


CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)


def get_args() -> argparse.Namespace:
    """Parse command-line arguments for the visualization script.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(description="t-SNE feature visualization")
    parser.add_argument("--model",     choices=["cnn", "resnet", "mobilenet"], required=True)
    parser.add_argument("--save_path", type=str, required=True,
                        help="Path to the saved .pth model weights.")
    parser.add_argument("--n_samples", type=int, default=2000,
                        help="Number of test samples to use for t-SNE (default: 2000).")
    parser.add_argument("--device",    type=str, default="cuda")
    parser.add_argument("--run_name",  type=str, default=None,
                        help="Name for the output plot file. Defaults to model name.")
    return parser.parse_args()


def build_model(model_name: str) -> nn.Module:
    """Instantiate the requested model architecture with 10 output classes.

    Args:
        model_name: One of 'cnn', 'resnet', 'mobilenet'.

    Returns:
        Instantiated model (weights not loaded yet).
    """
    if model_name == "cnn":
        return SimpleCNN(num_classes=10)
    if model_name == "resnet":
        return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)
    if model_name == "mobilenet":
        return MobileNetV2(num_classes=10)
    raise ValueError(f"Unknown model: {model_name}")


def get_penultimate_layer(model: nn.Module, model_name: str) -> nn.Module:
    """Return the penultimate layer to attach the hook to.

    This is the layer just before the final classification head,
    whose output gives us the richest learned feature representation.

    Args:
        model: The instantiated model.
        model_name: One of 'cnn', 'resnet', 'mobilenet'.

    Returns:
        The target layer module.
    """
    if model_name == "cnn":
        return model.fc1        # output: (N, 128)
    if model_name == "resnet":
        return model.avgpool    # output: (N, 512, 1, 1)
    if model_name == "mobilenet":
        return model.classifier[2]  # Dropout before final Linear, output: (N, 1280)
    raise ValueError(f"Unknown model: {model_name}")


def extract_features(
    model:      nn.Module,
    model_name: str,
    loader:     DataLoader,
    device:     torch.device,
    n_samples:  int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract penultimate-layer features using a forward hook.

    Attaches a hook to the penultimate layer, runs a forward pass,
    and collects the captured activations up to n_samples.

    Args:
        model: Trained model to extract features from.
        model_name: Architecture name for selecting the hook layer.
        loader: DataLoader for the test set.
        device: Device to run inference on.
        n_samples: Maximum number of samples to collect.

    Returns:
        Tuple of (features array of shape (n_samples, D),
                  labels array of shape (n_samples,)).
    """
    captured = {}

    def hook_fn(module: nn.Module, input: torch.Tensor, output: torch.Tensor) -> None:
        """Forward hook that captures and flattens the layer output."""
        captured["features"] = output.detach().cpu()

    # Register hook on the penultimate layer
    target_layer = get_penultimate_layer(model, model_name)
    handle = target_layer.register_forward_hook(hook_fn)

    model.eval()
    all_features, all_labels = [], []
    collected = 0

    with torch.no_grad():
        for imgs, labels in loader:
            if collected >= n_samples:
                break
            imgs = imgs.to(device)
            model(imgs)  # hook fires automatically during forward pass

            feats = captured["features"]
            feats = feats.view(feats.size(0), -1)  # flatten (handles avgpool's 4D output)

            remaining = n_samples - collected
            all_features.append(feats[:remaining].numpy())
            all_labels.append(labels[:remaining].numpy())
            collected += feats.size(0)

    handle.remove()  # always clean up the hook

    return np.concatenate(all_features), np.concatenate(all_labels)


def run_tsne(
    features:  np.ndarray,
    labels:    np.ndarray,
    title:     str,
    save_path: str,
) -> None:
    """Run t-SNE on features and save the resulting scatter plot.

    Args:
        features: Feature array of shape (N, D).
        labels: Class index array of shape (N,).
        title: Plot title.
        save_path: Path to save the output PNG.
    """
    print(f"Running t-SNE on {len(features)} samples of dimension {features.shape[1]}...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    embedded = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(10, 8))
    for class_idx, class_name in enumerate(CIFAR10_CLASSES):
        mask = labels == class_idx
        ax.scatter(
            embedded[mask, 0],
            embedded[mask, 1],
            s=5,
            label=class_name,
            alpha=0.7,
        )

    ax.legend(markerscale=3, bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_title(title)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"t-SNE plot saved to {save_path}")


def main() -> None:
    """Entry point: load model, extract features, run and save t-SNE plot."""
    args   = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    model = build_model(args.model).to(device)
    model.load_state_dict(torch.load(args.save_path, map_location=device))
    print(f"Loaded weights from: {args.save_path}")

    # ── Data ──────────────────────────────────────────────────────────────────
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    dataset = datasets.CIFAR10("./data", train=False, download=True, transform=tf)
    loader  = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=2)

    # ── Extract features ──────────────────────────────────────────────────────
    features, labels = extract_features(model, args.model, loader, device, args.n_samples)
    print(f"Extracted features: {features.shape}")

    # ── t-SNE + plot ──────────────────────────────────────────────────────────
    os.makedirs("plots", exist_ok=True)
    run_name  = args.run_name or args.model
    title     = f"t-SNE — {run_name}"
    save_path = f"plots/tsne_{run_name}.png"
    run_tsne(features, labels, title, save_path)


if __name__ == "__main__":
    main()
