"""Fine-tune and evaluate VSN transfer learning on SDNET2018."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from evaluate_vsn_sdnet_zero_shot import (  # noqa: E402
    binary_metrics,
    group_bootstrap_ci,
    sigmoid,
)
from train_vsn_binary import build_model, build_transforms  # noqa: E402
from train_vsn_student_baseline import VsnStudentDwCnn  # noqa: E402


RUN_KINDS = (
    "student_scratch",
    "teacher_imagenet_finetune",
    "teacher_source_finetune",
)


class ManifestDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], transform: Callable) -> None:
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        row = self.rows[index]
        with Image.open(PROJECT_ROOT / row["path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, torch.tensor(float(row["label"]), dtype=torch.float32), index


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_rows(index_path: Path) -> list[dict[str, str]]:
    with index_path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No rows found in {index_path}")
    return rows


def build_run_model(
    run_kind: str,
    source_checkpoint: Path,
    device: torch.device,
) -> tuple[nn.Module, int, int, float]:
    if run_kind == "student_scratch":
        model = VsnStudentDwCnn(width=1.0, dropout=0.1)
        return model.to(device), 128, 256, 1e-3
    if run_kind == "teacher_imagenet_finetune":
        model = build_model("mobilenet_v3_small", pretrained=True)
        return model.to(device), 224, 128, 3e-4
    if run_kind == "teacher_source_finetune":
        model = build_model("mobilenet_v3_small", pretrained=False)
        checkpoint = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        return model.to(device), 224, 128, 1e-4
    raise ValueError(f"Unsupported run kind: {run_kind}")


def make_loader(
    rows: list[dict[str, str]],
    transform: Callable,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        ManifestDataset(rows, transform),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


def collect_logits(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    logits = np.empty(len(loader.dataset), dtype=np.float32)
    labels = np.empty(len(loader.dataset), dtype=np.int64)
    total_loss = 0.0
    with torch.inference_mode():
        for images, batch_labels, indices in loader:
            images = images.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            batch_logits = model(images).flatten()
            loss = criterion(batch_logits, batch_labels)
            index_values = indices.numpy()
            logits[index_values] = batch_logits.detach().cpu().numpy()
            labels[index_values] = batch_labels.detach().cpu().numpy().astype(np.int64)
            total_loss += float(loss.item()) * len(indices)
    return logits, labels, total_loss / max(len(loader.dataset), 1)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> float:
    model.train()
    total_loss = 0.0
    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images).flatten()
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * images.size(0)
    return total_loss / max(len(loader.dataset), 1)


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    logits_tensor = torch.tensor(logits, dtype=torch.float64)
    labels_tensor = torch.tensor(labels, dtype=torch.float64)
    log_temperature = nn.Parameter(torch.zeros((), dtype=torch.float64))
    optimiser = torch.optim.LBFGS(
        [log_temperature],
        lr=0.1,
        max_iter=100,
        tolerance_grad=1e-10,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )
    criterion = nn.BCEWithLogitsLoss()

    def closure() -> torch.Tensor:
        optimiser.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = criterion(logits_tensor / temperature, labels_tensor)
        loss.backward()
        return loss

    optimiser.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 20.0).item())


def write_prediction_csv(
    path: Path,
    rows: list[dict[str, str]],
    logits: np.ndarray,
    raw_probabilities: np.ndarray,
    calibrated_probabilities: np.ndarray,
) -> None:
    fields = [
        "path",
        "label",
        "label_name",
        "surface",
        "parent_group",
        "component_id",
        "split",
        "logit",
        "probability_raw",
        "probability_target_val_calibrated",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow(
                {
                    **{field: row[field] for field in fields[:7]},
                    "logit": f"{float(logits[index]):.9g}",
                    "probability_raw": f"{float(raw_probabilities[index]):.9g}",
                    "probability_target_val_calibrated": (
                        f"{float(calibrated_probabilities[index]):.9g}"
                    ),
                }
            )


def train_run(args: argparse.Namespace, run_kind: str) -> dict[str, object]:
    set_seed(args.seed)
    device = torch.device(args.device)
    all_rows = read_rows(args.index)
    split_rows = {
        split: [row for row in all_rows if row["split"] == split]
        for split in ("train", "val", "test")
    }
    if any(not rows for rows in split_rows.values()):
        raise RuntimeError("Grouped train/val/test rows are required")

    model, image_size, default_batch_size, default_lr = build_run_model(
        run_kind,
        args.source_checkpoint,
        device,
    )
    batch_size = args.batch_size or default_batch_size
    learning_rate = args.lr or default_lr
    train_transform, eval_transform = build_transforms(image_size)
    loaders = {
        "train": make_loader(
            split_rows["train"],
            train_transform,
            batch_size,
            args.num_workers,
            device,
            shuffle=True,
        ),
        "val": make_loader(
            split_rows["val"],
            eval_transform,
            batch_size,
            args.num_workers,
            device,
            shuffle=False,
        ),
        "test": make_loader(
            split_rows["test"],
            eval_transform,
            batch_size,
            args.num_workers,
            device,
            shuffle=False,
        ),
    }

    positives = sum(int(row["label"]) == 1 for row in split_rows["train"])
    negatives = len(split_rows["train"]) - positives
    positive_weight = negatives / max(positives, 1)
    train_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(positive_weight, dtype=torch.float32, device=device)
    )
    eval_criterion = nn.BCEWithLogitsLoss()
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser,
        T_max=max(args.epochs, 1),
    )

    run_id = f"{run_kind}_seed{args.seed}"
    output_dir = args.output_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best.pt"
    history: list[dict[str, object]] = []
    best_val_macro_f1 = -math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    start_time = time.perf_counter()

    print(
        f"Run={run_id} device={device} image_size={image_size} "
        f"batch_size={batch_size} lr={learning_rate}",
        flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, loaders["train"], device, train_criterion, optimiser)
        val_logits, val_labels, val_loss = collect_logits(
            model,
            loaders["val"],
            device,
            eval_criterion,
        )
        val_metrics = binary_metrics(val_labels, sigmoid(val_logits))
        scheduler.step()
        history.append(
            {
                "epoch": epoch,
                "train_loss_weighted": train_loss,
                "val_loss_unweighted": val_loss,
                "val": val_metrics,
                "learning_rate": optimiser.param_groups[0]["lr"],
            }
        )
        print(
            f"{run_id} epoch={epoch:02d} train_loss={train_loss:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"val_positive_f1={val_metrics['f1_positive']:.4f} "
            f"val_bal_acc={val_metrics['balanced_accuracy']:.4f}",
            flush=True,
        )
        current = float(val_metrics["macro_f1"])
        if current > best_val_macro_f1 + args.min_delta:
            best_val_macro_f1 = current
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "run_kind": run_kind,
                    "image_size": image_size,
                    "args": vars(args),
                    "val_metrics": val_metrics,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch}", flush=True)
            break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    val_logits, val_labels, val_loss = collect_logits(
        model,
        loaders["val"],
        device,
        eval_criterion,
    )
    temperature = fit_temperature(val_logits, val_labels)
    test_logits, test_labels, test_loss = collect_logits(
        model,
        loaders["test"],
        device,
        eval_criterion,
    )
    raw_probabilities = sigmoid(test_logits)
    calibrated_probabilities = sigmoid(test_logits / temperature)
    raw_metrics = binary_metrics(test_labels, raw_probabilities)
    calibrated_metrics = binary_metrics(test_labels, calibrated_probabilities)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    elapsed_seconds = time.perf_counter() - start_time

    result: dict[str, object] = {
        "run_id": run_id,
        "run_kind": run_kind,
        "architecture": (
            "VsnStudentDwCnn" if run_kind == "student_scratch" else "MobileNetV3-Small"
        ),
        "initialisation": {
            "student_scratch": "random",
            "teacher_imagenet_finetune": "ImageNet",
            "teacher_source_finetune": "existing mixed-source crack teacher",
        }[run_kind],
        "source_checkpoint": (
            str(args.source_checkpoint) if run_kind == "teacher_source_finetune" else None
        ),
        "protocol": {
            "index": str(args.index),
            "grouped_split": True,
            "test_used_during_training": False,
            "selection_metric": "validation macro-F1 at fixed threshold 0.5",
            "calibration_data": "grouped target validation split",
            "threshold_selection_data": None,
            "decision_threshold": 0.5,
            "seed": args.seed,
        },
        "training": {
            "epochs_requested": args.epochs,
            "epochs_completed": len(history),
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_val_macro_f1,
            "patience": args.patience,
            "min_delta": args.min_delta,
            "batch_size": batch_size,
            "image_size": image_size,
            "learning_rate": learning_rate,
            "weight_decay": args.weight_decay,
            "positive_weight": positive_weight,
            "sample_counts": {split: len(rows) for split, rows in split_rows.items()},
            "elapsed_seconds": elapsed_seconds,
        },
        "model": {
            "parameters": parameter_count,
            "fp32_weight_mb_estimate": parameter_count * 4 / (1024 * 1024),
            "int8_weight_mb_estimate": parameter_count / (1024 * 1024),
        },
        "calibration": {
            "temperature": temperature,
            "validation_loss_unweighted": val_loss,
        },
        "test": {
            "loss_unweighted": test_loss,
            "raw": raw_metrics,
            "target_val_calibrated": calibrated_metrics,
            "group_bootstrap_95ci": {
                "raw": group_bootstrap_ci(
                    split_rows["test"],
                    test_labels,
                    raw_probabilities,
                    args.bootstrap_repetitions,
                    args.seed,
                ),
                "target_val_calibrated": group_bootstrap_ci(
                    split_rows["test"],
                    test_labels,
                    calibrated_probabilities,
                    args.bootstrap_repetitions,
                    args.seed,
                ),
            },
        },
        "claims_boundary": (
            "SDNET2018 crack classification is a visual damage-evidence task, not a certified "
            "structural risk or collapse-probability estimate."
        ),
    }
    (output_dir / "history.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in history),
        encoding="utf-8",
    )
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    write_prediction_csv(
        output_dir / "test_predictions.csv",
        split_rows["test"],
        test_logits,
        raw_probabilities,
        calibrated_probabilities,
    )
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return result


def write_comparison(output_dir: Path, results: list[dict[str, object]]) -> None:
    fields = [
        "run_id",
        "run_kind",
        "architecture",
        "initialisation",
        "parameters",
        "image_size",
        "best_epoch",
        "balanced_accuracy",
        "macro_f1",
        "f1_positive",
        "precision_positive",
        "recall_positive",
        "specificity",
        "roc_auc",
        "average_precision",
        "brier",
        "nll",
        "ece_15",
        "temperature",
    ]
    with (output_dir / "summary_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            metrics = result["test"]["target_val_calibrated"]
            writer.writerow(
                {
                    "run_id": result["run_id"],
                    "run_kind": result["run_kind"],
                    "architecture": result["architecture"],
                    "initialisation": result["initialisation"],
                    "parameters": result["model"]["parameters"],
                    "image_size": result["training"]["image_size"],
                    "best_epoch": result["training"]["best_epoch"],
                    **{field: metrics[field] for field in fields[7:18]},
                    "temperature": result["calibration"]["temperature"],
                }
            )
    (output_dir / "summary_metrics.json").write_text(
        json.dumps(results, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    lines = [
        "# SDNET2018 Grouped Transfer Results",
        "",
        "## Protocol",
        "",
        "- All optimisation uses the grouped SDNET2018 train split.",
        "- Checkpoints and temperature scaling use the grouped validation split.",
        "- The grouped test split is opened only after model selection.",
        "- The decision threshold remains fixed at 0.5; no test threshold tuning is performed.",
        "",
        "## Results",
        "",
        "| Run | Initialisation | Params | Balanced acc. | Macro-F1 | Positive F1 | AUROC | AUPRC | ECE |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        metrics = result["test"]["target_val_calibrated"]
        lines.append(
            f"| {result['run_kind']} | {result['initialisation']} | "
            f"{result['model']['parameters']:,} | {metrics['balanced_accuracy']:.4f} | "
            f"{metrics['macro_f1']:.4f} | {metrics['f1_positive']:.4f} | "
            f"{metrics['roc_auc']:.4f} | {metrics['average_precision']:.4f} | "
            f"{metrics['ece_15']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Claims Boundary",
            "",
            "These results measure grouped target-domain crack classification. They do not establish "
            "post-disaster structural safety, severity grading, or collapse probability.",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run grouped SDNET2018 VSN transfer baselines.")
    parser.add_argument("--run-kinds", nargs="+", choices=RUN_KINDS, default=list(RUN_KINDS))
    parser.add_argument(
        "--index",
        type=Path,
        default=PROJECT_ROOT
        / "experiments"
        / "vsn_sdnet2018"
        / "sdnet2018_index_grouped.csv",
    )
    parser.add_argument(
        "--source-checkpoint",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "vsn_binary_mixed_full"
        / "mobilenet_v3_small_pretrained"
        / "best.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "vsn_sdnet2018_transfer",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = [train_run(args, run_kind) for run_kind in args.run_kinds]
    write_comparison(args.output_dir, results)
    print(f"Wrote transfer comparison to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
