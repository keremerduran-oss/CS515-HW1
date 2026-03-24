import random
import ssl
import numpy as np
import torch

from parameters import get_params, DataConfig, ModelConfig, TrainingConfig
from models.MLP import MLP
from models.CNN import MNIST_CNN, SimpleCNN
from models.VGG import VGG
from models.ResNet import ResNet, BasicBlock
from train import run_training
from test import run_test

ssl._create_default_https_context = ssl._create_unverified_context


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(data_cfg: DataConfig, model_cfg: ModelConfig) -> torch.nn.Module:
    """Instantiate and return the model specified in model_cfg."""
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
            raise ValueError("VGG is designed for 3-channel images; use cifar10 with vgg.")
        return VGG(dept=model_cfg.vgg_depth, num_class=nc)

    if model_name == "resnet":
        if dataset == "mnist":
            raise ValueError("ResNet is designed for 3-channel images; use cifar10 with resnet.")
        return ResNet(BasicBlock, model_cfg.resnet_layers, num_classes=nc)

    raise ValueError(f"Unknown model: {model_name}")


def main() -> None:
    """Entry point: parse config, build model, run training and/or testing."""
    data_cfg, model_cfg, train_cfg = get_params()

    set_seed(train_cfg.seed)
    print(f"Seed set to: {train_cfg.seed}")
    print(f"Dataset: {data_cfg.dataset}  |  Model: {model_cfg.model}")

    device = torch.device(
        train_cfg.device if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Using device: {device}")

    model = build_model(data_cfg, model_cfg).to(device)
    print(model)

    if train_cfg.mode in ("train", "both"):
        run_training(model, data_cfg, model_cfg, train_cfg, device)

    if train_cfg.mode in ("test", "both"):
        run_test(model, data_cfg, train_cfg, device)


if __name__ == "__main__":
    main()
