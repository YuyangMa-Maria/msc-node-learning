"""Calibrate the ASN risk score and operating threshold on validation data.

The scalar temperature corrects probability sharpness, while a separate
validation-selected threshold preserves the binary decision operating point.
Both values are frozen before the independent test split is evaluated.
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

from train_asn_audio import PROJECT_ROOT, read_samples
from train_asn_student import AsnStudentDataset, AsnStudentModel


def as_namespace(data: dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(**data)


def source_set(value: object) -> set[str] | None:
    if value is None or value == "all":
        return None
    if isinstance(value, list):
        return set(str(item) for item in value)
    return None


def build_model(checkpoint: dict[str, object], device: torch.device) -> tuple[AsnStudentModel, SimpleNamespace]:
    args = as_namespace(checkpoint["args"])  # type: ignore[arg-type]
    model = AsnStudentModel(
        sample_rate=int(getattr(args, "sample_rate")),
        n_fft=int(getattr(args, "n_fft")),
        hop_length=int(getattr(args, "hop_length")),
        n_mels=int(getattr(args, "n_mels")),
        width=float(getattr(args, "width")),
        dropout=float(getattr(args, "dropout")),
    ).to(device)
    model.load_state_dict(checkpoint["model"])  # type: ignore[arg-type]
    model.eval()
    return model, args


def collect_logits(
    run_dir: Path,
    split: str,
    sources: set[str] | None,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, SimpleNamespace]:
    checkpoint = torch.load(run_dir / "best.pt", map_location=device)
    model, args = build_model(checkpoint, device)
    index = Path(args.index)
    if not index.is_absolute():
        index = PROJECT_ROOT / index
    samples = read_samples(index, split, sources)
    if not samples:
        raise RuntimeError(f"No samples found for split={split} sources={sources}")
    dataset = AsnStudentDataset(
        samples=samples,
        sample_rate=int(args.sample_rate),
        student_duration=float(args.duration),
        teacher_duration=float(getattr(args, "teacher_duration", 10.0)),
        train=False,
        seed=int(args.seed),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    logits: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    with torch.no_grad():
        for student_waveform, _, y in loader:
            student_waveform = student_waveform.to(device, non_blocking=True)
            logits.append(model(student_waveform).flatten().cpu())
            labels.append(y.float().cpu())
    return torch.cat(logits), torch.cat(labels), args


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, device: torch.device) -> float:
    """Fit a positive temperature without updating the acoustic model."""
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
        rows.append(
            {
                "bin_low": lo,
                "bin_high": hi,
                "count": count,
                "mean_risk_score": float(probs[mask].mean()),
                "positive_rate": float(labels[mask].mean()),
            }
        )
    return rows


def best_f1_threshold(probs: torch.Tensor, labels: torch.Tensor) -> float:
    """Choose the decision threshold from validation predictions only."""
    probs_cpu = probs.detach().cpu()
    y = labels.detach().cpu().int()
    candidates = torch.unique(probs_cpu).tolist()
    if not candidates:
        return 0.5
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in candidates:
        pred = (probs_cpu >= threshold).int()
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
    return best_threshold


def calibration_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    decision_threshold: float,
    bins: int,
) -> dict[str, object]:
    probs = torch.sigmoid(logits / temperature).clamp(1e-7, 1 - 1e-7)
    pred = (probs >= decision_threshold).float()
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
        "decision_threshold": decision_threshold,
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
        rows = [row for row in result["bins"] if row["count"] > 0]
        x = [row["mean_risk_score"] for row in rows]
        y = [row["positive_rate"] for row in rows]
        sizes = [max(20, row["count"] * 1.2) for row in rows]
        ax.scatter(x, y, s=sizes, alpha=0.75)
        ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("Mean predicted acoustic risk score")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Observed positive rate")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def write_markdown(output: dict[str, object], md_path: Path) -> None:
    test_before = output["test"]["before"]  # type: ignore[index]
    test_after = output["test"]["after"]  # type: ignore[index]
    lines = [
        "# ASN Student Risk Score Calibration",
        "",
        f"- Run: `{Path(str(output['run_dir'])).name}`",
        f"- Temperature: {output['temperature']:.4f}",
        f"- Original decision threshold: {output['original_decision_threshold']:.4f}",
        f"- Calibrated decision threshold: {output['decision_threshold']:.4f}",
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
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Temperature scaling adjusts the ASN student risk-score distribution while keeping the ranking of samples unchanged. The calibrated temperature is used by the compressed fusion experiment to compute `risk_score` and entropy-based `confidence`.",
            "",
            "## Caution",
            "",
            "- This calibrates the selected ASN student checkpoint. Hardware-specific INT8 calibration may still differ slightly.",
            "- The output remains an acoustic abnormality / structural-risk proxy, not a certified structural failure probability.",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Fit calibration parameters and write test-set reliability evidence."""
    parser = argparse.ArgumentParser(description="Calibrate ASN student acoustic risk scores with temperature scaling.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--sources", nargs="*", default=None)
    parser.add_argument("--calib-split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--test-split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "asn_student_calibration")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.run_dir / "best.pt", map_location="cpu")
    checkpoint_args = as_namespace(checkpoint["args"])
    sources = set(args.sources) if args.sources else source_set(getattr(checkpoint_args, "sources", None))
    result_path = args.run_dir / "result.json"
    original_decision_threshold = 0.5
    if result_path.exists():
        original_decision_threshold = float(json.loads(result_path.read_text(encoding="utf-8")).get("decision_threshold", 0.5))

    calib_logits, calib_labels, _ = collect_logits(args.run_dir, args.calib_split, sources, device, args.batch_size)
    temperature = fit_temperature(calib_logits, calib_labels, device)
    # Test labels are used for final reporting only, not temperature or
    # threshold selection.
    test_logits, test_labels, _ = collect_logits(args.run_dir, args.test_split, sources, device, args.batch_size)
    calibrated_val_probs = torch.sigmoid(calib_logits / temperature)
    calibrated_decision_threshold = best_f1_threshold(calibrated_val_probs, calib_labels)

    calib_before = calibration_metrics(calib_logits, calib_labels, 1.0, original_decision_threshold, args.bins)
    calib_after = calibration_metrics(calib_logits, calib_labels, temperature, calibrated_decision_threshold, args.bins)
    test_before = calibration_metrics(test_logits, test_labels, 1.0, original_decision_threshold, args.bins)
    test_after = calibration_metrics(test_logits, test_labels, temperature, calibrated_decision_threshold, args.bins)

    output = {
        "model": "asn_student_logmel_cnn",
        "run_dir": str(args.run_dir),
        "sources": sorted(sources) if sources else "all",
        "calib_split": args.calib_split,
        "test_split": args.test_split,
        "temperature": temperature,
        "original_decision_threshold": original_decision_threshold,
        "decision_threshold": calibrated_decision_threshold,
        "risk_score_definition": "temperature_scaled_sigmoid(logit / T), interpreted as acoustic abnormality risk proxy",
        "confidence_definition": "1 - binary_entropy(risk_score), range 0-1",
        "calibration": {"before": calib_before, "after": calib_after},
        "test": {"before": test_before, "after": test_after},
    }

    stem = f"{args.run_dir.parent.name}_{args.run_dir.name}"
    json_path = args.output_dir / f"{stem}_calibration.json"
    plot_path = args.output_dir / f"{stem}_reliability.png"
    md_path = args.output_dir / f"{stem}_calibration.md"
    json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    plot_reliability(test_before, test_after, plot_path)
    write_markdown(output, md_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {plot_path}")
    print(f"Wrote {md_path}")
    print(f"temperature={temperature:.6f}")
    print(f"test_ece_before={test_before['ece']:.6f} test_ece_after={test_after['ece']:.6f}")


if __name__ == "__main__":
    main()
