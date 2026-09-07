"""Train the initial binary VSN model-selection baselines.

Candidate backbones share the same transforms, splits and reporting code so the
comparison reflects architecture choice rather than a different data pipeline.
These larger FP32 models establish the teacher/reference point for later tiny
students; they are not assumed to be MCU-deployable.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Sample:
    path: Path
    label: int
    source: str
    split: str


class VsnBinaryDataset(Dataset):
    def __init__(self, samples: list[Sample], transform: transforms.Compose) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        image = Image.open(sample.path).convert("RGB")
        return self.transform(image), torch.tensor(sample.label, dtype=torch.float32)


def read_samples(index_path: Path, split: str, sources: set[str] | None) -> list[Sample]:
    rows: list[Sample] = []
    with index_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] != split:
                continue
            if sources and row["source"] not in sources:
                continue
            rows.append(
                Sample(
                    path=PROJECT_ROOT / row["path"],
                    label=int(row["label"]),
                    source=row["source"],
                    split=row["split"],
                )
            )
    return rows


def limit_samples(samples: list[Sample], limit: int | None, seed: int) -> list[Sample]:
    """Subsample an index without changing its class balance substantially."""
    if limit is None or limit <= 0 or len(samples) <= limit:
        return samples
    by_label: dict[int, list[Sample]] = {0: [], 1: []}
    for sample in samples:
        by_label[sample.label].append(sample)
    rng = random.Random(seed)
    for group in by_label.values():
        rng.shuffle(group)
    selected: list[Sample] = []
    for label in (0, 1):
        quota = max(1, round(limit * len(by_label[label]) / max(len(samples), 1)))
        selected.extend(by_label[label][:quota])
    rng.shuffle(selected)
    return selected[:limit]


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    """Return stochastic training and deterministic evaluation transforms."""
    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(8),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_tf, eval_tf


def build_model(name: str, pretrained: bool) -> nn.Module:
    weights = "DEFAULT" if pretrained else None
    # Replace the final layer to output a single logit (unnormalized probability)
    if name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
    elif name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
    elif name == "resnet18":
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 1)
    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
    else:
        raise ValueError(f"Unsupported model: {name}")
    return model


def binary_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    """Calculates evaluation metrics. Uses max(..., 1) to avoid ZeroDivisionError."""
    probs = torch.sigmoid(logits).detach().cpu()
    y = labels.detach().cpu().int()
    # Threshold at 0.5 for hard predictions
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
        roc_auc = float("nan")  # Handles cases where a batch only contains one class
    return {
        "accuracy": accuracy,
        "precision_positive": precision,
        "recall_positive": recall,
        "specificity": specificity,
        "f1_positive": f1,
        "roc_auc": roc_auc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    """
    Executes a single pass over the dataset.
    Acts as training loop if an optimizer is provided, otherwise acts as evaluation.
    """
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
        logits = model(images).flatten()
        loss = criterion(logits, labels)
        if is_train:
            loss.backward()
            optimizer.step()
        total_loss += float(loss.item()) * images.size(0)
        all_logits.append(logits.detach())
        all_labels.append(labels.detach())
    logits_cat = torch.cat(all_logits)
    labels_cat = torch.cat(all_labels)
    metrics = binary_metrics(logits_cat, labels_cat)
    metrics["loss"] = total_loss / max(len(loader.dataset), 1)
    return metrics


def benchmark_inference(model: nn.Module, loader: DataLoader, device: torch.device, batches: int = 20) -> float:
    """Measures inference speed (ms/image). Requires torch.cuda.synchronize() for accurate GPU timings."""
    model.eval()
    times: list[float] = []
    with torch.no_grad():
        for idx, (images, _) in enumerate(loader):
            if idx >= batches:
                break
            images = images.to(device, non_blocking=True)
            # Sync before starting the timer to avoid counting data transfer time
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(images)
            # Sync before stopping the timer to ensure GPU execution is fully complete
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            times.append(elapsed / images.size(0) * 1000.0)
    return float(sum(times) / max(len(times), 1))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def build_loss(train_samples: list[Sample], device: torch.device, pos_weight: str) -> nn.Module:
    '''Constructs BCE loss. Dynamically calculates positive weight for highly imbalanced datasets.'''
    if pos_weight == "none":
        return nn.BCEWithLogitsLoss()
    if pos_weight == "auto":
        positives = sum(sample.label == 1 for sample in train_samples)
        negatives = sum(sample.label == 0 for sample in train_samples)
        # Weight formula: N_negative / N_positive. Prevents the model from just guessing 0.
        weight = negatives / max(positives, 1)
    else:
        weight = float(pos_weight)
    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight, dtype=torch.float32, device=device))


def save_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train VSN binary visual risk model.")
    parser.add_argument("--index", type=Path, default=PROJECT_ROOT / "experiments" / "vsn_binary" / "vsn_binary_index.csv")
    parser.add_argument("--model", choices=["mobilenet_v3_small", "mobilenet_v3_large", "resnet18", "efficientnet_b0"], required=True)
    parser.add_argument("--sources", nargs="*", default=None, help="Optional source filter, e.g. mendeley_concrete_crack")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--pos-weight", default="auto", help="Use 'auto', 'none', or a numeric BCE positive-class weight.")
    parser.add_argument("--max-train", type=int, default=None, help="Optional stratified train sample limit for quick screening.")
    parser.add_argument("--max-val", type=int, default=None, help="Optional stratified validation sample limit for quick screening.")
    parser.add_argument("--max-test", type=int, default=None, help="Optional stratified test sample limit for quick screening.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "vsn_binary")
    args = parser.parse_args()

    # 1. Data Preparation
    sources = set(args.sources) if args.sources else None
    train_tf, eval_tf = build_transforms(args.image_size)
    train_samples = read_samples(args.index, "train", sources)
    val_samples = read_samples(args.index, "val", sources)
    test_samples = read_samples(args.index, "test", sources)
    train_samples = limit_samples(train_samples, args.max_train, args.seed)
    val_samples = limit_samples(val_samples, args.max_val, args.seed)
    test_samples = limit_samples(test_samples, args.max_test, args.seed)
    if not train_samples or not val_samples or not test_samples:
        raise RuntimeError("Empty train/val/test split. Run scripts/build_vsn_index.py first.")

    device = torch.device(args.device)
    # pin_memory=True speeds up host-to-device transfers (CPU to GPU)
    train_loader = DataLoader(VsnBinaryDataset(train_samples, train_tf), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(VsnBinaryDataset(val_samples, eval_tf), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    test_loader = DataLoader(VsnBinaryDataset(test_samples, eval_tf), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")

    # 2. Model & Optimizer Initialization
    model = build_model(args.model, args.pretrained).to(device)
    criterion = build_loss(train_samples, device, args.pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    run_id = f"{args.model}_{'pretrained' if args.pretrained else 'scratch'}"
    output_dir = args.output_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, object]] = []
    best_val_f1 = -math.inf
    best_path = output_dir / "best.pt"

    print(f"Device: {device}", flush=True)
    print(f"Model: {args.model}", flush=True)
    print(f"Parameters: {count_parameters(model)}", flush=True)
    print(f"Samples: train={len(train_samples)} val={len(val_samples)} test={len(test_samples)}", flush=True)

    # 3. Training Loop
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device, None)
        row: dict[str, object] = {"epoch": epoch, "model": args.model, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(f"epoch={epoch} train_loss={train_metrics['loss']:.4f} val_f1={val_metrics['f1_positive']:.4f} val_recall={val_metrics['recall_positive']:.4f}", flush=True)
        # Checkpointing: Save model weights ONLY if it beats the previous best F1 score
        if val_metrics["f1_positive"] > best_val_f1:
            best_val_f1 = val_metrics["f1_positive"]
            torch.save({"model": model.state_dict(), "args": vars(args), "val_metrics": val_metrics}, best_path)

    # 4. Final Evaluation & Export
    # Load the best weights from the entire run before testing
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    test_metrics = run_epoch(model, test_loader, criterion, device, None)
    inference_ms = benchmark_inference(model, test_loader, device)
    result = {
        "model": args.model,
        "pretrained": args.pretrained,
        "parameters": count_parameters(model),
        "average_inference_ms": inference_ms,
        "best_val_f1": best_val_f1,
        "test": test_metrics,
        "sources": sorted(sources) if sources else "all",
        "training": {
            "index": str(args.index),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "image_size": args.image_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "pos_weight": args.pos_weight,
            "max_train": args.max_train,
            "max_val": args.max_val,
            "max_test": args.max_test,
            "sample_counts": {
                "train": len(train_samples),
                "val": len(val_samples),
                "test": len(test_samples),
            },
            "device": str(device),
            "seed": args.seed,
        },
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    save_jsonl(output_dir / "history.jsonl", history)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
