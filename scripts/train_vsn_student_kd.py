"""Train the compact VSN student using hard labels and teacher soft targets.

The teacher and student see the same grouped split. Only the student is updated;
the teacher is frozen and used to expose class similarity through softened
binary logits. Validation F1 selects the checkpoint and the test split is read
only after model selection.
"""

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
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from train_vsn_binary import PROJECT_ROOT, VsnBinaryDataset, build_model, build_transforms, read_samples
from train_vsn_student_baseline import VsnStudentDwCnn, count_parameters, size_estimates


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def build_hard_loss(train_labels: Iterable[int], device: torch.device) -> nn.Module:
    labels = list(train_labels)
    positives = sum(label == 1 for label in labels)
    negatives = sum(label == 0 for label in labels)
    weight = negatives / max(positives, 1)
    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight, dtype=torch.float32, device=device))


def binary_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    hard_criterion: nn.Module,
    alpha: float,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Blend class-weighted supervision with temperature-scaled distillation."""
    hard_loss = hard_criterion(student_logits, labels)
    t = temperature
    # The T^2 factor keeps the distillation gradient on a comparable scale when
    # a larger temperature flattens the teacher distribution.
    teacher_prob = torch.sigmoid(teacher_logits / t)
    kd_loss = F.binary_cross_entropy_with_logits(student_logits / t, teacher_prob, reduction="mean") * (t * t)
    loss = alpha * hard_loss + (1.0 - alpha) * kd_loss
    return loss, hard_loss.detach(), kd_loss.detach()


def run_eval(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> dict[str, float | int]:
    model.eval()
    total_loss = 0.0
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += float(loss.item()) * images.size(0)
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())
    logits_cat = torch.cat(all_logits)
    labels_cat = torch.cat(all_labels)
    metrics = binary_metrics(logits_cat, labels_cat)
    metrics["loss"] = total_loss / max(len(loader.dataset), 1)
    return metrics


def run_kd_epoch(
    student: nn.Module,
    teacher: nn.Module,
    loader: DataLoader,
    hard_criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    alpha: float,
    temperature: float,
) -> dict[str, float | int]:
    student.train()
    teacher.eval()
    total_loss = 0.0
    total_hard = 0.0
    total_kd = 0.0
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        student_logits = student(images)
        with torch.no_grad():
            teacher_logits = teacher(images).flatten()
        loss, hard_loss, kd_loss = binary_kd_loss(student_logits, teacher_logits, labels, hard_criterion, alpha, temperature)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * images.size(0)
        total_hard += float(hard_loss.item()) * images.size(0)
        total_kd += float(kd_loss.item()) * images.size(0)
        all_logits.append(student_logits.detach().cpu())
        all_labels.append(labels.detach().cpu())
    logits_cat = torch.cat(all_logits)
    labels_cat = torch.cat(all_labels)
    metrics = binary_metrics(logits_cat, labels_cat)
    denom = max(len(loader.dataset), 1)
    metrics["loss"] = total_loss / denom
    metrics["hard_loss"] = total_hard / denom
    metrics["kd_loss"] = total_kd / denom
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


def load_teacher(args: argparse.Namespace, device: torch.device) -> nn.Module:
    """Restore and freeze the selected FP32 teacher checkpoint."""
    teacher = build_model(args.teacher_model, pretrained=False).to(device)
    checkpoint = torch.load(args.teacher_checkpoint, map_location=device)
    teacher.load_state_dict(checkpoint["model"])
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    return teacher


def save_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def plot_history(history: list[dict[str, object]], output_path: Path) -> None:
    epochs = [int(row["epoch"]) for row in history]
    train_loss = [float(row["train"]["loss"]) for row in history]  # type: ignore[index]
    train_hard = [float(row["train"]["hard_loss"]) for row in history]  # type: ignore[index]
    train_kd = [float(row["train"]["kd_loss"]) for row in history]  # type: ignore[index]
    val_f1 = [float(row["val"]["f1_positive"]) for row in history]  # type: ignore[index]
    val_recall = [float(row["val"]["recall_positive"]) for row in history]  # type: ignore[index]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=140)
    axes[0].plot(epochs, train_loss, marker="o", label="Total")
    axes[0].plot(epochs, train_hard, marker="s", label="Hard")
    axes[0].plot(epochs, train_kd, marker="^", label="KD")
    axes[0].set_title("Train losses")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()
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
    """Train one input-resolution candidate and persist its full audit trail."""
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

    student = VsnStudentDwCnn(width=args.width, dropout=args.dropout).to(device)
    teacher = load_teacher(args, device)
    parameters = count_parameters(student)
    hard_criterion = build_hard_loss((sample.label for sample in train_samples), device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    run_id = f"vsn_student_dwcnn_s{image_size}_w{args.width:g}_kd_t{args.temperature:g}_a{args.alpha:g}"
    output_dir = args.output_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pt"
    history: list[dict[str, object]] = []
    best_val_f1 = -math.inf
    best_epoch = 0

    print(f"\nRun: {run_id}", flush=True)
    print(f"Device: {device} student_parameters={parameters}", flush=True)
    print(f"Teacher: {args.teacher_model} checkpoint={args.teacher_checkpoint}", flush=True)
    print(f"Samples: train={len(train_samples)} val={len(val_samples)} test={len(test_samples)}", flush=True)
    print(f"KD config: temperature={args.temperature} alpha={args.alpha}", flush=True)

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_kd_epoch(student, teacher, train_loader, hard_criterion, device, optimizer, args.alpha, args.temperature)
        val_metrics = run_eval(student, val_loader, hard_criterion, device)
        scheduler.step()
        row: dict[str, object] = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(
            f"{run_id} epoch={epoch:02d} train_loss={float(train_metrics['loss']):.4f} "
            f"hard={float(train_metrics['hard_loss']):.4f} kd={float(train_metrics['kd_loss']):.4f} "
            f"val_f1={float(val_metrics['f1_positive']):.4f} val_recall={float(val_metrics['recall_positive']):.4f}",
            flush=True,
        )
        if float(val_metrics["f1_positive"]) > best_val_f1:
            best_val_f1 = float(val_metrics["f1_positive"])
            best_epoch = epoch
            torch.save(
                {
                    "model": student.state_dict(),
                    "args": vars(args),
                    "image_size": image_size,
                    "val_metrics": val_metrics,
                    "kd": {"temperature": args.temperature, "alpha": args.alpha},
                },
                best_path,
            )

    # Test data are deliberately evaluated once, after validation has fixed the
    # epoch and all optimisation choices.
    checkpoint = torch.load(best_path, map_location=device)
    student.load_state_dict(checkpoint["model"])
    test_metrics = run_eval(student, test_loader, hard_criterion, device)
    inference_ms = benchmark_inference(student, test_loader, device)
    estimates = size_estimates(parameters)
    result: dict[str, object] = {
        "model": "vsn_student_dwcnn",
        "run_id": run_id,
        "student_type": "FP32_KD",
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
        "kd": {
            "teacher_model": args.teacher_model,
            "teacher_checkpoint": str(args.teacher_checkpoint),
            "temperature": args.temperature,
            "alpha": args.alpha,
        },
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
    csv_path = output_dir / "summary_metrics.csv"
    fieldnames = [
        "run_id",
        "student_type",
        "image_size",
        "temperature",
        "alpha",
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
            kd = result["kd"]  # type: ignore[assignment]
            writer.writerow(
                {
                    "run_id": result["run_id"],
                    "student_type": result["student_type"],
                    "image_size": result["image_size"],
                    "temperature": kd["temperature"],
                    "alpha": kd["alpha"],
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
            )

    lines = [
        "# VSN Student Knowledge Distillation",
        "",
        "## Scope",
        "",
        "- Teacher: MobileNetV3-Small visual risk model.",
        "- Student: depthwise separable CNN used in the from-scratch VSN student baseline.",
        "- Purpose: test whether teacher soft targets improve TinyML student clean performance without changing inference memory.",
        "",
        "## Results",
        "",
        "| Run | Image size | T | Alpha | Accuracy | Recall | F1 | ROC-AUC | Params | FP32 MB | INT8 est MB | Avg ms/image |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        test = result["test"]  # type: ignore[assignment]
        kd = result["kd"]  # type: ignore[assignment]
        lines.append(
            "| {run} | {size} | {temp:.2f} | {alpha:.2f} | {acc:.4f} | {rec:.4f} | {f1:.4f} | {auc:.4f} | {params:.4f}M | {fp32:.4f} | {int8:.4f} | {ms:.4f} |".format(
                run=result["run_id"],
                size=result["image_size"],
                temp=float(kd["temperature"]),
                alpha=float(kd["alpha"]),
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
            "## Interpretation Template",
            "",
            "Compare these KD runs against `outputs/vsn_student_baselines/REPORT.md`. Because the from-scratch student is already strong, KD should be judged not only by clean F1, but also by recall, calibration, robustness, and later INT8 behavior.",
            "",
            "## Caution",
            "",
            "- This is still a visual crack/damage proxy task, not real collapse prediction.",
            "- KD does not change inference memory; any gain is a training-time compression benefit.",
            "- Robustness and calibration should be evaluated before claiming the KD student is better for risk-score generation.",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    (output_dir / "summary_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {csv_path}", flush=True)
    print(f"Wrote {output_dir / 'REPORT.md'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train VSN TinyML student with knowledge distillation.")
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
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for hard-label BCE loss. 1-alpha weights KD loss.")
    parser.add_argument("--teacher-model", choices=["mobilenet_v3_small", "mobilenet_v3_large", "resnet18", "efficientnet_b0"], default="mobilenet_v3_small")
    parser.add_argument(
        "--teacher-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "vsn_binary_mixed_full" / "mobilenet_v3_small_pretrained" / "best.pt",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "vsn_student_kd")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = [train_one(args, image_size) for image_size in args.image_sizes]
    write_summary(results, args.output_dir)


if __name__ == "__main__":
    main()
