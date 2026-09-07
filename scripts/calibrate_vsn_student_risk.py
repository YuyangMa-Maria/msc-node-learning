"""Fit temperature scaling for the selected VSN student.

Temperature is fitted on validation logits only, then frozen before reporting
test ECE, Brier score and NLL. The transform changes score calibration but not
the ordering of predictions or the trained model weights.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader

from train_vsn_binary import PROJECT_ROOT, VsnBinaryDataset, build_transforms, read_samples
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


def collect_logits(
    run_dir: Path,
    split: str,
    sources: set[str] | None,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, SimpleNamespace, int]:
    checkpoint = torch.load(run_dir / "best.pt", map_location=device)
    args = as_namespace(checkpoint["args"])
    image_size = checkpoint_image_size(checkpoint)
    index = Path(args.index)
    if not index.is_absolute():
        index = PROJECT_ROOT / index
    _, eval_tf = build_transforms(image_size)
    samples = read_samples(index, split, sources)
    if not samples:
        raise RuntimeError(f"No samples found for split={split} sources={sources}")
    loader = DataLoader(
        VsnBinaryDataset(samples, eval_tf),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model = build_student_from_checkpoint(checkpoint, device)
    logits: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    with torch.no_grad():
        for images, y in loader:
            images = images.to(device, non_blocking=True)
            logits.append(model(images).flatten().cpu())
            labels.append(y.float().cpu())
    return torch.cat(logits), torch.cat(labels), args, image_size


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, device: torch.device) -> float:
    """Optimise one positive scalar on held-out validation predictions."""
    logits = logits.to(device)
    labels = labels.to(device)
    log_temperature = torch.zeros((), device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.05, max_iter=100)
    criterion = nn.BCEWithLogitsLoss()

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        temperature = torch.exp(log_temperature).clamp(0.05, 20.0)
        loss = criterion(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_temperature).detach().cpu().clamp(0.05, 20.0))


def reliability_bins(probs: torch.Tensor, labels: torch.Tensor, bins: int) -> list[dict[str, float]]:
    rows = []
    for idx in range(bins):
        lo = idx / bins
        hi = (idx + 1) / bins
        mask = (probs >= lo) & (probs <= hi) if idx == bins - 1 else (probs >= lo) & (probs < hi)
        count = int(mask.sum())
        if count == 0:
            rows.append({"bin_low": lo, "bin_high": hi, "count": 0, "mean_risk_score": math.nan, "positive_rate": math.nan})
            continue
        rows.append({"bin_low": lo, "bin_high": hi, "count": count, "mean_risk_score": float(probs[mask].mean()), "positive_rate": float(labels[mask].mean())})
    return rows


def calibration_metrics(logits: torch.Tensor, labels: torch.Tensor, temperature: float, bins: int) -> dict[str, object]:
    """Compute discrimination and calibration measures from fixed logits."""
    probs = torch.sigmoid(logits / temperature).clamp(1e-7, 1 - 1e-7)
    pred = (probs >= 0.5).float()
    tp = int(((pred == 1) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    brier = float(((probs - labels) ** 2).mean())
    nll = float((-(labels * probs.log() + (1 - labels) * (1 - probs).log())).mean())
    entropy = -(probs * torch.log2(probs) + (1 - probs) * torch.log2(1 - probs))
    confidence = 1.0 - entropy
    rows = reliability_bins(probs, labels, bins)
    ece = 0.0
    for row in rows:
        if row["count"] == 0:
            continue
        ece += row["count"] / len(labels) * abs(row["mean_risk_score"] - row["positive_rate"])
    return {
        "temperature": temperature,
        "accuracy": accuracy,
        "precision_positive": precision,
        "recall_positive": recall,
        "f1_positive": f1,
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
        "bins": rows,
    }


def plot_reliability(before: dict[str, object], after: dict[str, object], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=140, sharex=True, sharey=True)
    for ax, title, result in zip(axes, ["Before calibration", "After temperature scaling"], [before, after]):
        rows = [row for row in result["bins"] if row["count"] > 0]  # type: ignore[index]
        x = [row["mean_risk_score"] for row in rows]
        y = [row["positive_rate"] for row in rows]
        sizes = [max(20, row["count"] * 0.8) for row in rows]
        ax.scatter(x, y, s=sizes, alpha=0.75)
        ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("Mean predicted risk score")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Observed positive rate")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def write_markdown(path: Path, run_id: str, image_size: int, temperature: float, test_before: dict[str, object], test_after: dict[str, object]) -> None:
    lines = [
        "# VSN Student Risk Score Calibration",
        "",
        f"- Run: `{run_id}`",
        f"- Student input size: {image_size}",
        f"- Temperature: {temperature:.4f}",
        "- Risk score: `sigmoid(logit / T)`",
        "- Confidence: `1 - binary_entropy(risk_score)`",
        "",
        "## Test Metrics",
        "",
        "| State | Accuracy | Recall | F1 | Brier | NLL | ECE | Mean confidence | Confusion matrix (TN/FP/FN/TP) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, result in [("Before", test_before), ("After", test_after)]:
        lines.append(
            "| {name} | {acc:.4f} | {rec:.4f} | {f1:.4f} | {brier:.6f} | {nll:.6f} | {ece:.6f} | {conf:.4f} | {tn}/{fp}/{fn}/{tp} |".format(
                name=name,
                acc=result["accuracy"],
                rec=result["recall_positive"],
                f1=result["f1_positive"],
                brier=result["brier"],
                nll=result["nll"],
                ece=result["ece"],
                conf=result["mean_confidence"],
                tn=result["tn"],
                fp=result["fp"],
                fn=result["fn"],
                tp=result["tp"],
            )
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Fit on the calibration split and apply the frozen value to the test split."""
    parser = argparse.ArgumentParser(description="Calibrate VSN student risk scores with temperature scaling.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--sources", nargs="*", default=None)
    parser.add_argument("--calib-split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--test-split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "vsn_student_calibration")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.run_dir / "best.pt", map_location="cpu")
    checkpoint_args = as_namespace(checkpoint["args"])
    sources = set(args.sources) if args.sources else source_set(getattr(checkpoint_args, "sources", None))
    calib_logits, calib_labels, model_args, image_size = collect_logits(args.run_dir, args.calib_split, sources, device, args.batch_size)
    temperature = fit_temperature(calib_logits, calib_labels, device)
    # No test label participates in fitting the temperature.
    test_logits, test_labels, _, _ = collect_logits(args.run_dir, args.test_split, sources, device, args.batch_size)

    calib_before = calibration_metrics(calib_logits, calib_labels, 1.0, args.bins)
    calib_after = calibration_metrics(calib_logits, calib_labels, temperature, args.bins)
    test_before = calibration_metrics(test_logits, test_labels, 1.0, args.bins)
    test_after = calibration_metrics(test_logits, test_labels, temperature, args.bins)

    stem = f"{args.run_dir.parent.name}_{args.run_dir.name}"
    output = {
        "model": "vsn_student_dwcnn",
        "run_dir": str(args.run_dir),
        "run_id": args.run_dir.name,
        "student_type": getattr(model_args, "student_type", "student"),
        "image_size": image_size,
        "sources": sorted(sources) if sources else "all",
        "calib_split": args.calib_split,
        "test_split": args.test_split,
        "temperature": temperature,
        "risk_score_definition": "temperature_scaled_sigmoid(logit / T), interpreted as P(visual crack/damage)",
        "confidence_definition": "1 - binary_entropy(risk_score), range 0-1",
        "calibration": {"before": calib_before, "after": calib_after},
        "test": {"before": test_before, "after": test_after},
    }
    json_path = args.output_dir / f"{stem}_calibration.json"
    plot_path = args.output_dir / f"{stem}_reliability.png"
    md_path = args.output_dir / f"{stem}_calibration.md"
    json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    plot_reliability(test_before, test_after, plot_path)
    write_markdown(md_path, args.run_dir.name, image_size, temperature, test_before, test_after)
    print(f"Wrote {json_path}")
    print(f"Wrote {plot_path}")
    print(f"Wrote {md_path}")
    print(f"temperature={temperature:.4f} test_ece_before={test_before['ece']:.6f} test_ece_after={test_after['ece']:.6f}")


if __name__ == "__main__":
    main()
