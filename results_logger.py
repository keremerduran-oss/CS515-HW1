import csv
import os

from parameters import DataConfig, ModelConfig, TrainingConfig


def log_result(
    filepath: str,
    train_cfg: TrainingConfig,
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
    best_val_acc: float,
    test_acc: float,
) -> None:
    """Append a single experiment result to a CSV file, creating it if needed."""
    file_exists = os.path.isfile(filepath)

    with open(filepath, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "hidden_sizes", "activation", "dropout", "batch_norm",
            "reg_type", "reg_lambda", "learning_rate", "batch_size",
            "scheduler", "epochs", "best_val_acc", "test_acc",
        ])
        if not file_exists:
            writer.writeheader()

        writer.writerow({
            "hidden_sizes":  str(model_cfg.hidden_sizes) if model_cfg else "N/A",
            "activation":    model_cfg.activation        if model_cfg else "N/A",
            "dropout":       model_cfg.dropout           if model_cfg else "N/A",
            "batch_norm":    model_cfg.batch_norm        if model_cfg else "N/A",
            "reg_type":      train_cfg.reg_type,
            "reg_lambda":    train_cfg.reg_lambda,
            "learning_rate": train_cfg.learning_rate,
            "batch_size":    train_cfg.batch_size,
            "scheduler":     train_cfg.scheduler,
            "epochs":        train_cfg.epochs,
            "best_val_acc":  f"{best_val_acc:.4f}",
            "test_acc":      f"{test_acc:.4f}",
        })
    print(f"Result logged to {filepath}")
