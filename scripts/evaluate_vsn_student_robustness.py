"""Evaluate the distilled VSN student under controlled image corruption.

Low light, blur and related transforms are applied only at evaluation time.
These controlled tests isolate sensitivity to input quality but do not claim to
reproduce the full distribution of a post-disaster environment.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import random
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from train_vsn_binary import PROJECT_ROOT, Sample, build_transforms, read_samples
from train_vsn_student_baseline import VsnStudentDwCnn


def as_namespace(data: dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(**data)


def source_set(value: object) -> set[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return set(str(item) for item in value)
    if value == "all":
        return None
    return None


def load_temperature(path: Path | None) -> float:
    if path is None:
        return 1.0
    data = json.loads(path.read_text(encoding="utf-8"))
    return float(data["temperature"])


def build_student_from_checkpoint(checkpoint: dict[str, object], device: torch.device) -> nn.Module:
    args = as_namespace(checkpoint["args"])  # type: ignore[arg-type]
    model = VsnStudentDwCnn(width=float(getattr(args, "width", 1.0)), dropout=float(getattr(args, "dropout", 0.1))).to(device)
    model.load_state_dict(checkpoint["model"])  # type: ignore[arg-type]
    model.eval()
    return model


def checkpoint_image_size(checkpoint: dict[str, object]) -> int:
    if "image_size" in checkpoint:
        return int(checkpoint["image_size"])
    args = as_namespace(checkpoint["args"])  # type: ignore[arg-type]
    return int(getattr(args, "image_size"))


def apply_corruption(image: Image.Image, corruption: str, severity: int, seed: int) -> Image.Image:
    if corruption == "clean":
        return image
    if corruption == "low_light":
        return ImageEnhance.Brightness(image).enhance({1: 0.7, 2: 0.45, 3: 0.25}[severity])
    if corruption == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius={1: 1.0, 2: 2.0, 3: 3.5}[severity]))
    if corruption == "gaussian_noise":
        rng = np.random.default_rng(seed)
        arr = np.asarray(image).astype(np.float32) / 255.0
        arr = np.clip(arr + rng.normal(0.0, {1: 0.03, 2: 0.07, 3: 0.12}[severity], size=arr.shape), 0.0, 1.0)
        return Image.fromarray((arr * 255).astype(np.uint8))
    if corruption == "occlusion":
        rng = random.Random(seed)
        arr = np.asarray(image.copy()).copy()
        h, w = arr.shape[:2]
        side = int(min(h, w) * {1: 0.12, 2: 0.22, 3: 0.35}[severity])
        x0 = rng.randint(0, max(w - side, 0))
        y0 = rng.randint(0, max(h - side, 0))
        arr[y0 : y0 + side, x0 : x0 + side] = 0
        return Image.fromarray(arr)
    if corruption == "jpeg":
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality={1: 60, 2: 35, 3: 15}[severity])
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")
    raise ValueError(f"Unsupported corruption: {corruption}")


class RobustVsnDataset(Dataset):
    def __init__(self, samples: list[Sample], transform, corruption: str, severity: int, seed: int) -> None:
        self.samples = samples
        self.transform = transform
        self.corruption = corruption
        self.severity = severity
        self.seed = seed

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        image = Image.open(sample.path).convert("RGB")
        image = apply_corruption(image, self.corruption, self.severity, self.seed + index)
        return self.transform(image), torch.tensor(sample.label, dtype=torch.float32)


def calibrated_metrics(logits: torch.Tensor, labels: torch.Tensor, temperature: float, bins: int) -> dict[str, float | int]:
    logits = logits.cpu()
    labels = labels.cpu().float()
    probs = torch.sigmoid(logits / temperature).clamp(1e-7, 1 - 1e-7)
    pred = (probs >= 0.5).int()
    y = labels.int()
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
    brier = float(((probs - labels) ** 2).mean())
    nll = float((-(labels * probs.log() + (1 - labels) * (1 - probs).log())).mean())
    entropy = -(probs * torch.log2(probs) + (1 - probs) * torch.log2(1 - probs))
    confidence = 1.0 - entropy
    ece = 0.0
    for idx in range(bins):
        lo = idx / bins
        hi = (idx + 1) / bins
        mask = (probs >= lo) & (probs <= hi) if idx == bins - 1 else (probs >= lo) & (probs < hi)
        if not int(mask.sum()):
            continue
        ece += int(mask.sum()) / len(labels) * abs(float(probs[mask].mean()) - float(labels[mask].mean()))
    return {
        "accuracy": accuracy,
        "precision_positive": precision,
        "recall_positive": recall,
        "specificity": specificity,
        "f1_positive": f1,
        "roc_auc": roc_auc,
        "brier": brier,
        "nll": nll,
        "ece": ece,
        "mean_risk_score_positive": float(probs[labels == 1].mean()) if int((labels == 1).sum()) else math.nan,
        "mean_risk_score_negative": float(probs[labels == 0].mean()) if int((labels == 0).sum()) else math.nan,
        "mean_confidence": float(confidence.mean()),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def evaluate_condition(
    model: nn.Module,
    samples: list[Sample],
    transform,
    corruption: str,
    severity: int,
    temperature: float,
    device: torch.device,
    batch_size: int,
    bins: int,
    seed: int,
) -> dict[str, object]:
    loader = DataLoader(
        RobustVsnDataset(samples, transform, corruption, severity, seed),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            all_logits.append(model(images).flatten().detach().cpu())
            all_labels.append(labels.detach().cpu())
    metrics = calibrated_metrics(torch.cat(all_logits), torch.cat(all_labels), temperature, bins)
    metrics["corruption"] = corruption
    metrics["severity"] = severity
    metrics["sample_count"] = len(samples)
    return metrics


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    fieldnames = [
        "corruption",
        "severity",
        "sample_count",
        "accuracy",
        "precision_positive",
        "recall_positive",
        "specificity",
        "f1_positive",
        "roc_auc",
        "brier",
        "nll",
        "ece",
        "mean_risk_score_positive",
        "mean_risk_score_negative",
        "mean_confidence",
        "tn",
        "fp",
        "fn",
        "tp",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def plot_robustness(rows: list[dict[str, object]], output_path: Path) -> None:
    corruptions = [name for name in ["low_light", "blur", "gaussian_noise", "occlusion", "jpeg"] if any(r["corruption"] == name for r in rows)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=140)
    for ax, metric, label in zip(axes, ["f1_positive", "recall_positive", "ece"], ["F1", "Recall", "ECE"]):
        for corruption in corruptions:
            subset = sorted([r for r in rows if r["corruption"] == corruption], key=lambda r: r["severity"])
            ax.plot([r["severity"] for r in subset], [r[metric] for r in subset], marker="o", label=corruption)
        ax.set_xlabel("Severity")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def write_markdown(rows: list[dict[str, object]], output_path: Path, metadata: dict[str, object]) -> None:
    clean = next(row for row in rows if row["corruption"] == "clean")
    worst = min([row for row in rows if row["corruption"] != "clean"], key=lambda row: row["f1_positive"])
    lines = [
        "# VSN Student Robustness Evaluation",
        "",
        f"- Run: `{metadata['run_id']}`",
        f"- Image size: {metadata['image_size']}",
        f"- Sources: {metadata['sources']}",
        f"- Split: {metadata['split']}",
        f"- Temperature: {metadata['temperature']:.4f}",
        "- Risk score: calibrated `sigmoid(logit / T)`",
        "",
        "## Summary",
        "",
        f"- Clean F1: {clean['f1_positive']:.4f}, clean ECE: {clean['ece']:.6f}",
        f"- Worst condition by F1: {worst['corruption']} severity {worst['severity']} with F1={worst['f1_positive']:.4f}, recall={worst['recall_positive']:.4f}, ECE={worst['ece']:.6f}",
        "",
        "## Results",
        "",
        "| Corruption | Severity | Accuracy | Recall | F1 | ECE | Mean risk + | Mean risk - | Confusion matrix (TN/FP/FN/TP) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {corr} | {sev} | {acc:.4f} | {rec:.4f} | {f1:.4f} | {ece:.6f} | {pos:.4f} | {neg:.4f} | {tn}/{fp}/{fn}/{tp} |".format(
                corr=row["corruption"],
                sev=row["severity"],
                acc=row["accuracy"],
                rec=row["recall_positive"],
                f1=row["f1_positive"],
                ece=row["ece"],
                pos=row["mean_risk_score_positive"],
                neg=row["mean_risk_score_negative"],
                tn=row["tn"],
                fp=row["fp"],
                fn=row["fn"],
                tp=row["tp"],
            )
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VSN student robustness under visual corruptions.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--calibration-json", type=Path, default=None)
    parser.add_argument("--sources", nargs="*", default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "vsn_student_robustness")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.run_dir / "best.pt", map_location=device)
    model_args = as_namespace(checkpoint["args"])
    sources = set(args.sources) if args.sources else source_set(getattr(model_args, "sources", None))
    index = Path(model_args.index)
    if not index.is_absolute():
        index = PROJECT_ROOT / index
    image_size = checkpoint_image_size(checkpoint)
    samples = read_samples(index, args.split, sources)
    if not samples:
        raise RuntimeError(f"No samples found for split={args.split} sources={sources}")

    _, eval_tf = build_transforms(image_size)
    model = build_student_from_checkpoint(checkpoint, device)
    temperature = load_temperature(args.calibration_json)
    conditions = [("clean", 0)]
    for corruption in ["low_light", "blur", "gaussian_noise", "occlusion", "jpeg"]:
        for severity in [1, 2, 3]:
            conditions.append((corruption, severity))

    rows = []
    for corruption, severity in conditions:
        result = evaluate_condition(model, samples, eval_tf, corruption, severity, temperature, device, args.batch_size, args.bins, args.seed)
        rows.append(result)
        print(
            f"{args.run_dir.name} {corruption}:{severity} accuracy={result['accuracy']:.4f} "
            f"f1={result['f1_positive']:.4f} recall={result['recall_positive']:.4f} ece={result['ece']:.6f}",
            flush=True,
        )

    stem = f"{args.run_dir.parent.name}_{args.run_dir.name}_{args.split}"
    metadata = {
        "run_id": args.run_dir.name,
        "image_size": image_size,
        "sources": ", ".join(sorted(sources)) if sources else "all",
        "split": args.split,
        "temperature": temperature,
    }
    csv_path = args.output_dir / f"{stem}_robustness.csv"
    json_path = args.output_dir / f"{stem}_robustness.json"
    md_path = args.output_dir / f"{stem}_robustness.md"
    plot_path = args.output_dir / f"{stem}_robustness.png"
    write_csv(rows, csv_path)
    json_path.write_text(json.dumps({"metadata": metadata, "results": rows}, indent=2), encoding="utf-8")
    plot_robustness(rows, plot_path)
    write_markdown(rows, md_path, metadata)
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {plot_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
