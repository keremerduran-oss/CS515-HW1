import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional


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
    # Transfer learning
    transfer_option: Optional[str]
    # Knowledge distillation / label smoothing
    label_smoothing: float
    temperature: float
    alpha: float
    teacher_path: Optional[str]
    distill_mode: str


def get_params() -> Tuple[DataConfig, ModelConfig, TrainingConfig]:
    """Parse command-line arguments and return structured config dataclasses."""
    parser = argparse.ArgumentParser(description="CS515 HW2 - Transfer Learning & Knowledge Distillation")

    # ── Mode & data ──────────────────────────────────────────────────────────
    parser.add_argument("--mode", choices=["train", "test", "both"], default="both")
    parser.add_argument("--run_name", type=str, default=None, help="Optional name for this run. Overrides the auto-generated name for saving/logging.",)
    parser.add_argument("--dataset", choices=["mnist", "cifar10"], default="cifar10")
    parser.add_argument("--model", choices=["mlp", "cnn", "vgg", "resnet", "mobilenet"], default="resnet")

    # ── Training hyperparameters ─────────────────────────────────────────────
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--activation", choices=["relu", "gelu"], default="relu")
    parser.add_argument("--batch_norm", type=lambda x: x.lower() != "false", default=True)
    parser.add_argument("--hidden_sizes", type=int, nargs="+", default=[512, 256, 128])
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--reg_type", choices=["l1", "l2", "none"], default="none")
    parser.add_argument("--reg_lambda", type=float, default=1e-4)
    parser.add_argument("--scheduler", choices=["step", "cosine", "plateau"], default="cosine")
    parser.add_argument("--vgg_depth", choices=["11", "13", "16", "19"], default="16")
    parser.add_argument("--resnet_layers", type=int, nargs=4, default=[2, 2, 2, 2],
                        metavar=("L1", "L2", "L3", "L4"),
                        help="Number of blocks per ResNet layer (default: 2 2 2 2 = ResNet-18)")

    # ── Part A: Transfer learning ────────────────────────────────────────────
    parser.add_argument(
        "--transfer_option",
        choices=["option1", "option2"],
        default=None,
        help=(
            "Transfer learning mode. "
            "option1: resize CIFAR-10 to 224x224, freeze early layers, train FC only. "
            "option2: modify early conv for 32x32 input, fine-tune early layers + FC."
        ),
    )

    # ── Part B: Label smoothing & knowledge distillation ────────────────────
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.0,
        help="Label smoothing epsilon for CrossEntropyLoss (0.0 = disabled).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=4.0,
        help="Softmax temperature for knowledge distillation.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="Weight of the distillation loss vs hard-label loss (0.0–1.0).",
    )
    parser.add_argument(
        "--teacher_path",
        type=str,
        default=None,
        help="Path to saved teacher model weights (.pth) for knowledge distillation.",
    )
    parser.add_argument(
        "--distill_mode",
        choices=["standard", "soft_targets_only"],
        default="standard",
        help=(
            "standard: full KDLoss (soft + hard targets). "
            "soft_targets_only: use teacher confidence on true class only "
            "(dynamic label smoothing variant)."
        ),
    )

    args = parser.parse_args()

    # ── Derived dataset config ───────────────────────────────────────────────
    if args.dataset == "mnist":
        input_size = 784
        mean, std = (0.1307,), (0.3081,)
    else:
        input_size = 3072
        mean = (0.4914, 0.4822, 0.4465)
        std  = (0.2023, 0.1994, 0.2010)

    # ── Run name for saving/logging ──────────────────────────────────────────
    if args.run_name:
        run_name = args.run_name
    elif args.transfer_option:
        run_name = f"transfer_{args.transfer_option}_resnet18"
    else:
        run_name = (
            f"{args.model}_act{args.activation}_do{args.dropout}"
            f"_bn{args.batch_norm}_reg{args.reg_type}_sch{args.scheduler}"
            f"_ls{args.label_smoothing}"
        )

    # ── Save path ────────────────────────────────────────────────────────────
    save_path = f"models/saved/{run_name}.pth"

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
        epochs          = args.epochs,
        batch_size      = args.batch_size,
        learning_rate   = args.lr,
        weight_decay    = 1e-4,
        scheduler       = args.scheduler,
        reg_type        = args.reg_type,
        reg_lambda      = args.reg_lambda,
        seed            = 42,
        device          = args.device,
        save_path       = save_path,
        log_interval    = 100,
        mode            = args.mode,
        run_name        = run_name,
        transfer_option = args.transfer_option,
        label_smoothing = args.label_smoothing,
        temperature     = args.temperature,
        alpha           = args.alpha,
        teacher_path    = args.teacher_path,
        distill_mode    = args.distill_mode,
    )

    return data_cfg, model_cfg, train_cfg
