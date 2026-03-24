import random
import ssl
import numpy as np
import torch

from parameters import get_params, DataConfig, ModelConfig, TrainingConfig
from models.MLP import MLP
from models.CNN import MNIST_CNN, SimpleCNN
from models.VGG import VGG
from models.ResNet import ResNet, BasicBlock
from models.MobileNet import MobileNetV2
from train import run_training
from test import run_test
from pretrained import run_transfer_learning
from distillation import run_distillation, count_flops

ssl._create_default_https_context = ssl._create_unverified_context


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across all libraries.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def build_model(data_cfg: DataConfig, model_cfg: ModelConfig) -> torch.nn.Module:
    """Instantiate and return the model specified in model_cfg.

    Args:
        data_cfg: Dataset configuration (used for input size and num_classes).
        model_cfg: Model architecture configuration.

    Returns:
        An instantiated PyTorch model.
    """
    model_name = model_cfg.model
    dataset    = data_cfg.dataset
    nc         = data_cfg.num_classes

    if model_name == "mlp":
        return MLP(
            input_size   = data_cfg.input_size,
            hidden_sizes = model_cfg.hidden_sizes,
            num_classes  = nc,
            dropout      = model_cfg.dropout,
            activation   = model_cfg.activation,
            batch_norm   = model_cfg.batch_norm,
        )

    if model_name == "cnn":
        if dataset == "mnist":
            return MNIST_CNN(num_classes=nc)
        return SimpleCNN(num_classes=nc)

    if model_name == "vgg":
        if dataset == "mnist":
            raise ValueError("VGG is designed for 3-channel images; use cifar10.")
        return VGG(dept=model_cfg.vgg_depth, num_class=nc)

    if model_name == "resnet":
        if dataset == "mnist":
            raise ValueError("ResNet is designed for 3-channel images; use cifar10.")
        return ResNet(BasicBlock, model_cfg.resnet_layers, num_classes=nc)

    if model_name == "mobilenet":
        if dataset == "mnist":
            raise ValueError("MobileNetV2 is designed for 3-channel images; use cifar10.")
        return MobileNetV2(num_classes=nc)

    raise ValueError(f"Unknown model: {model_name}")


def main() -> None:
    """Entry point: parse config, build model, run training and/or testing."""
    data_cfg, model_cfg, train_cfg = get_params()

    set_seed(train_cfg.seed)
    print(f"Seed: {train_cfg.seed}")
    print(f"Dataset: {data_cfg.dataset} | Model: {model_cfg.model}")

    device = torch.device(
        train_cfg.device if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Device: {device}")

    # ── Route to transfer learning if --transfer_option is set ───────────────
    if train_cfg.transfer_option is not None:
        run_transfer_learning(data_cfg, model_cfg, train_cfg, device)
        return

    # ── Route to knowledge distillation if --teacher_path is set ─────────────
    if train_cfg.teacher_path is not None:
        student = build_model(data_cfg, model_cfg).to(device)

        # Load teacher (always ResNet-18)
        teacher = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=data_cfg.num_classes).to(device)
        teacher.load_state_dict(torch.load(train_cfg.teacher_path, map_location=device))
        print(f"Teacher loaded from: {train_cfg.teacher_path}")

        # FLOPs comparison before training
        print("\n── Teacher FLOPs ──")
        count_flops(teacher)
        print("── Student FLOPs ──")
        count_flops(student)

        run_distillation(student, teacher, data_cfg, model_cfg, train_cfg, device)
        return

    # ── Standard training path ────────────────────────────────────────────────
    model = build_model(data_cfg, model_cfg).to(device)
    print(model)

    if train_cfg.mode in ("train", "both"):
        run_training(model, data_cfg, model_cfg, train_cfg, device)

    if train_cfg.mode in ("test", "both"):
        run_test(model, data_cfg, train_cfg, device)


if __name__ == "__main__":
    main()
