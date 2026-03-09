import argparse
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class DataConfig:
    """Dataset configuration and normalization parameters."""
    dataset: str
    data_dir: str
    num_workers: int
    mean: Tuple
    std: Tuple
    input_size: int
    num_classes: int


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    model: str
    hidden_sizes: List[int]
    dropout: float
    activation: str
    batch_norm: bool
    vgg_depth: str
    resnet_layers: List[int]


@dataclass
class TrainingConfig:
    """Training hyperparameters and run settings."""
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    scheduler: str
    reg_type: str
    reg_lambda: float
    seed: int
    device: str
    save_path: str
    log_interval: int
    mode: str
    run_name: str


def get_params() -> Tuple[DataConfig, ModelConfig, TrainingConfig]:
    """Parse command-line arguments and return structured config dataclasses."""
    parser = argparse.ArgumentParser(description="Deep Learning on MNIST / CIFAR-10")

    parser.add_argument("--mode",       choices=["train", "test", "both"], default="both")
    parser.add_argument("--dataset",    choices=["mnist", "cifar10"],      default="mnist")
    parser.add_argument("--model",      choices=["mlp", "cnn", "vgg", "resnet"], default="mlp")
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--device",     type=str,   default="cpu")
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--activation", choices=["relu", "gelu"], default="relu")
    parser.add_argument("--batch_norm", type=lambda x: x.lower() != "false", default=True)
    parser.add_argument("--hidden_sizes", type=int, nargs="+", default=[512, 256, 128])
    parser.add_argument("--dropout",    type=float, default=0.3)
    parser.add_argument("--reg_type",   choices=["l1", "l2", "none"], default="none")
    parser.add_argument("--reg_lambda", type=float, default=1e-4)
    parser.add_argument("--scheduler",  choices=["step", "cosine", "plateau"], default="step")
    parser.add_argument("--vgg_depth",  choices=["11", "13", "16", "19"], default="16")
    parser.add_argument("--resnet_layers", type=int, nargs=4, default=[2, 2, 2, 2],
                        metavar=("L1", "L2", "L3", "L4"),
                        help="Number of blocks per ResNet layer (default: 2 2 2 2 = ResNet-18)")

    args = parser.parse_args()

    if args.dataset == "mnist":
        input_size = 784
        mean, std  = (0.1307,), (0.3081,)
    else:
        input_size = 3072
        mean       = (0.4914, 0.4822, 0.4465)
        std        = (0.2023, 0.1994, 0.2010)

    run_name = (
        f"mlp_h{'-'.join(str(h) for h in args.hidden_sizes)}"
        f"_act{args.activation}_do{args.dropout}"
        f"_bn{args.batch_norm}_reg{args.reg_type}_sch{args.scheduler}"
    )

    data_cfg = DataConfig(
        dataset     = args.dataset,
        data_dir    = "./data",
        num_workers = 4,
        mean        = mean,
        std         = std,
        input_size  = input_size,
        num_classes = 10,
    )

    model_cfg = ModelConfig(
        model         = args.model,
        hidden_sizes  = args.hidden_sizes,
        dropout       = args.dropout,
        activation    = args.activation,
        batch_norm    = args.batch_norm,
        vgg_depth     = args.vgg_depth,
        resnet_layers = args.resnet_layers,
    )

    train_cfg = TrainingConfig(
        epochs        = args.epochs,
        batch_size    = args.batch_size,
        learning_rate = args.lr,
        weight_decay  = 1e-4,
        scheduler     = args.scheduler,
        reg_type      = args.reg_type,
        reg_lambda    = args.reg_lambda,
        seed          = 42,
        device        = args.device,
        save_path     = (
            f"models/saved_v2/best_model_{args.activation}"
            f"_do{args.dropout}_bn{args.batch_norm}"
            f"_reg{args.reg_type}_sch{args.scheduler}.pth"
        ),
        log_interval  = 100,
        mode          = args.mode,
        run_name      = run_name,
    )

    return data_cfg, model_cfg, train_cfg
