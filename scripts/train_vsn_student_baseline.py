"""Train the compact VSN student architecture without distillation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from train_vsn_binary import PROJECT_ROOT, VsnBinaryDataset, build_transforms, read_samples


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class VsnStudentDwCnn(nn.Module):
    def __init__(self, width: float = 1.0, dropout: float = 0.1) -> None:
        super().__init__()
        base_channels = [16, 24, 32, 48, 64, 96]
        channels = [max(8, int(round(value * width))) for value in base_channels]
        self.features = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
            DepthwiseSeparableBlock(channels[0], channels[1], stride=2),
            DepthwiseSeparableBlock(channels[1], channels[2], stride=2),
            DepthwiseSeparableBlock(channels[2], channels[3], stride=2),
            DepthwiseSeparableBlock(channels[3], channels[4], stride=2),
            DepthwiseSeparableBlock(channels[4], channels[5], stride=1),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels[-1], 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x)).flatten()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def size_estimates(parameters: int) -> dict[str, float]:
    return {
        "fp32_weight_mb": parameters * 4 / (1024 * 1024),
        "int8_weight_est_mb": parameters / (1024 * 1024),
    }


def binary_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float | int]:
    probs = torch.sigmoid(logits).detach().cpu()
    y = labels.detach().cpu().int()
    pred = (probs >= 0.5).int()
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    try:
        roc_auc = float(roc_auc_score(y.numpy(), probs.numpy()))
    except ValueError:
        roc_auc = float("nan")
    return {
        "accuracy": accuracy,
        "precision_positive": precision,
        "recall_positive": recall,
        "specificity": specificity,
        "f1_positive": f1,
        "roc_auc": roc_auc,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float | int]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        if is_train:
            loss.backward()
            optimizer.step()
        total_loss += float(loss.item()) * images.size(0)
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())
    logits_cat = torch.cat(all_logits)
    labels_cat = torch.cat(all_labels)
    metrics = binary_metrics(logits_cat, labels_cat)
    metrics["loss"] = total_loss / max(len(loader.dataset), 1)
    return metrics


def benchmark_inference(model: nn.Module, loader: DataLoader, device: torch.device, batches: int = 30) -> float:
    model.eval()
    times: list[float] = []
    with torch.no_grad():
        for idx, (images, _) in enumerate(loader):
            if idx >= batches:
                break
            images = images.to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) / images.size(0) * 1000.0)
    return float(sum(times) / max(len(times), 1))


def build_loss(train_labels: Iterable[int], device: torch.device) -> nn.Module:
    labels = list(train_labels)
    positives = sum(label == 1 for label in labels)
    negatives = sum(label == 0 for label in labels)
    weight = negatives / max(positives, 1)
    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight, dtype=torch.float32, device=device))


def save_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def plot_history(history: list[dict[str, object]], output_path: Path) -> None:
    epochs = [int(row["epoch"]) for row in history]
    val_f1 = [float(row["val"]["f1_positive"]) for row in history]  # type: ignore[index]
    val_recall = [float(row["val"]["recall_positive"]) for row in history]  # type: ignore[index]
    train_loss = [float(row["train"]["loss"]) for row in history]  # type: ignore[index]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=140)
    axes[0].plot(epochs, train_loss, marker="o")
    axes[0].set_title("Train loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(True, alpha=0.25)
    axes[1].plot(epochs, val_f1, marker="o", label="Val F1")
    axes[1].plot(epochs, val_recall, marker="s", label="Val recall")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Validation metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def train_one(args: argparse.Namespace, image_size: int) -> dict[str, object]:
    set_seed(args.seed)
    device = torch.device(args.device)
    sources = set(args.sources) if args.sources else None
    train_tf, eval_tf = build_transforms(image_size)
    train_samples = read_samples(args.index, "train", sources)
    val_samples = read_samples(args.index, "val", sources)
    test_samples = read_samples(args.index, "test", sources)
    train_loader = DataLoader(
        VsnBinaryDataset(train_samples, train_tf),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        VsnBinaryDataset(val_samples, eval_tf),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        VsnBinaryDataset(test_samples, eval_tf),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = VsnStudentDwCnn(width=args.width, dropout=args.dropout).to(device)
    parameters = count_parameters(model)
    criterion = build_loss((sample.label for sample in train_samples), device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    run_id = f"vsn_student_dwcnn_s{image_size}_w{args.width:g}_scratch"
    output_dir = args.output_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pt"
    history: list[dict[str, object]] = []
    best_val_f1 = -math.inf
    best_epoch = 0

    print(f"\nRun: {run_id}", flush=True)
    print(f"Device: {device} parameters={parameters}", flush=True)
    print(f"Samples: train={len(train_samples)} val={len(val_samples)} test={len(test_samples)}", flush=True)

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device, None)
        scheduler.step()
        row: dict[str, object] = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(
            f"{run_id} epoch={epoch:02d} train_loss={float(train_metrics['loss']):.4f} "
            f"val_f1={float(val_metrics['f1_positive']):.4f} val_recall={float(val_metrics['recall_positive']):.4f}",
            flush=True,
        )
        if float(val_metrics["f1_positive"]) > best_val_f1:
            best_val_f1 = float(val_metrics["f1_positive"])
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "args": vars(args), "image_size": image_size, "val_metrics": val_metrics}, best_path)

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    test_metrics = run_epoch(model, test_loader, criterion, device, None)
    inference_ms = benchmark_inference(model, test_loader, device)
    estimates = size_estimates(parameters)
    result: dict[str, object] = {
        "model": "vsn_student_dwcnn",
        "run_id": run_id,
        "student_type": "FP32_from_scratch",
        "image_size": image_size,
        "width": args.width,
        "dropout": args.dropout,
        "parameters": parameters,
        "parameters_m": parameters / 1_000_000,
        "fp32_weight_mb": estimates["fp32_weight_mb"],
        "int8_weight_est_mb": estimates["int8_weight_est_mb"],
        "average_inference_ms": inference_ms,
        "best_val_f1": best_val_f1,
        "best_epoch": best_epoch,
        "test": test_metrics,
        "training": {
            "index": str(args.index),
            "sources": sorted(sources) if sources else "all",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "sample_counts": {"train": len(train_samples), "val": len(val_samples), "test": len(test_samples)},
            "device": str(device),
            "seed": args.seed,
        },
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    save_jsonl(output_dir / "history.jsonl", history)
    plot_history(history, output_dir / "history.png")
    return result


def write_summary(results: list[dict[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "summary_metrics.csv"
    fieldnames = [
        "run_id",
        "model",
        "student_type",
        "image_size",
        "accuracy",
        "precision_positive",
        "recall_positive",
        "f1_positive",
        "roc_auc",
        "tn",
        "fp",
        "fn",
        "tp",
        "parameters_m",
        "fp32_weight_mb",
        "int8_weight_est_mb",
        "average_inference_ms",
        "best_epoch",
        "best_val_f1",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            test = result["test"]  # type: ignore[assignment]
            row = {
                "run_id": result["run_id"],
                "model": result["model"],
                "student_type": result["student_type"],
                "image_size": result["image_size"],
                "accuracy": test["accuracy"],
                "precision_positive": test["precision_positive"],
                "recall_positive": test["recall_positive"],
                "f1_positive": test["f1_positive"],
                "roc_auc": test["roc_auc"],
                "tn": test["tn"],
                "fp": test["fp"],
                "fn": test["fn"],
                "tp": test["tp"],
                "parameters_m": result["parameters_m"],
                "fp32_weight_mb": result["fp32_weight_mb"],
                "int8_weight_est_mb": result["int8_weight_est_mb"],
                "average_inference_ms": result["average_inference_ms"],
                "best_epoch": result["best_epoch"],
                "best_val_f1": result["best_val_f1"],
            }
            writer.writerow(row)

    lines = [
        "# VSN Student FP32 From-Scratch Baseline",
        "",
        "## Scope",
        "",
        "- This is the first TinyML student baseline.",
        "- No knowledge distillation, PTQ, or QAT is used here.",
        "- The model is a small depthwise separable CNN trained from scratch.",
        "- Results are intended as the baseline needed before claiming that KD helps.",
        "",
        "## Results",
        "",
        "| Run | Image size | Accuracy | Recall | F1 | ROC-AUC | Params | FP32 weights MB | INT8 est MB | Avg ms/image |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        test = result["test"]  # type: ignore[assignment]
        lines.append(
            "| {run} | {size} | {acc:.4f} | {rec:.4f} | {f1:.4f} | {auc:.4f} | {params:.4f}M | {fp32:.4f} | {int8:.4f} | {ms:.4f} |".format(
                run=result["run_id"],
                size=result["image_size"],
                acc=float(test["accuracy"]),
                rec=float(test["recall_positive"]),
                f1=float(test["f1_positive"]),
                auc=float(test["roc_auc"]),
                params=float(result["parameters_m"]),
                fp32=float(result["fp32_weight_mb"]),
                int8=float(result["int8_weight_est_mb"]),
                ms=float(result["average_inference_ms"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "These runs answer whether a very small VSN student can learn the current visual risk task without a teacher. KD should only be interpreted as useful if it improves on this from-scratch baseline under the same image size and architecture.",
            "",
            "## Caution",
            "",
            "- This is still a dataset-level visual crack/damage proxy, not real collapse prediction.",
            "- Memory values are weight-size estimates, not MCU tensor-arena profiling.",
            "- These models are FP32 students; deployment quantisation has not been run yet.",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    (output_dir / "summary_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {output_dir / 'REPORT.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train VSN TinyML student FP32 baselines from scratch.")
    parser.add_argument("--index", type=Path, default=PROJECT_ROOT / "experiments" / "vsn_binary" / "vsn_binary_index.csv")
    parser.add_argument("--image-sizes", type=int, nargs="+", default=[128, 96])
    parser.add_argument("--sources", nargs="*", default=None)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--width", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "vsn_student_baselines")
    args = parser.parse_args()

    results = [train_one(args, image_size) for image_size in args.image_sizes]
    write_summary(results, args.output_dir)


if __name__ == "__main__":
    main()
