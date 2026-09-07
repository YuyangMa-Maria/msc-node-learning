"""Compare ASN FP32 and PTQ predictions under controlled audio degradation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.ao.quantization import convert, get_default_qconfig, prepare

from evaluate_asn_robustness import (
    collect_scores,
    confidence_from_risk,
    metrics,
    source_set,
)
from export_asn_student_ptq import (
    QuantizedEncoderWrapper,
    build_student_from_checkpoint,
    calibrate,
    fuse_student_encoder,
    make_loader,
)
from train_asn_audio import PROJECT_ROOT, read_samples


def as_namespace(data: dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(**data)


def expected_calibration_error(scores: torch.Tensor, labels: torch.Tensor, bins: int) -> float:
    scores = scores.detach().cpu()
    labels = labels.detach().cpu().float()
    predicted = (scores >= 0.5).float()
    confidence = torch.maximum(scores, 1.0 - scores)
    correctness = (predicted == labels).float()
    edges = torch.linspace(0.0, 1.0, bins + 1)
    ece = torch.tensor(0.0)
    for idx in range(bins):
        lower = edges[idx]
        upper = edges[idx + 1]
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            ece += mask.float().mean() * (confidence[mask].mean() - correctness[mask].mean()).abs()
    return float(ece)


def evaluate_pair(
    fp32_model: torch.nn.Module,
    int8_model: torch.nn.Module,
    samples,
    sample_rate: int,
    target_samples: int,
    temperature: float,
    threshold: float,
    condition: str,
    severity: int,
    batch_size: int,
    seed: int,
    bins: int,
) -> dict[str, object]:
    common = (
        samples,
        sample_rate,
        target_samples,
        temperature,
        condition,
        severity,
        batch_size,
        torch.device("cpu"),
        seed,
    )
    fp32_scores, labels = collect_scores(fp32_model, *common)
    int8_scores, int8_labels = collect_scores(int8_model, *common)
    if not torch.equal(labels, int8_labels):
        raise RuntimeError("FP32 and INT8 evaluation labels are not aligned.")

    fp32 = metrics(fp32_scores, labels, threshold)
    int8 = metrics(int8_scores, labels, threshold)
    fp32["ece"] = expected_calibration_error(fp32_scores, labels, bins)
    int8["ece"] = expected_calibration_error(int8_scores, labels, bins)
    fp32["mean_confidence"] = float(confidence_from_risk(fp32_scores).mean())
    int8["mean_confidence"] = float(confidence_from_risk(int8_scores).mean())
    return {
        "condition": condition,
        "severity": severity,
        "sample_count": len(samples),
        "fp32": fp32,
        "int8": int8,
        "delta": {
            "accuracy": float(int8["accuracy"]) - float(fp32["accuracy"]),
            "recall_positive": float(int8["recall_positive"]) - float(fp32["recall_positive"]),
            "f1_positive": float(int8["f1_positive"]) - float(fp32["f1_positive"]),
            "roc_auc": float(int8["roc_auc"]) - float(fp32["roc_auc"]),
            "ece": float(int8["ece"]) - float(fp32["ece"]),
            "mean_risk_score": float(int8["mean_risk_score"]) - float(fp32["mean_risk_score"]),
            "mean_confidence": float(int8["mean_confidence"]) - float(fp32["mean_confidence"]),
        },
    }


def write_outputs(rows: list[dict[str, object]], metadata: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "asn_ptq_robustness_parity.csv"
    json_path = output_dir / "asn_ptq_robustness_parity.json"
    report_path = output_dir / "REPORT.md"

    csv_rows: list[dict[str, object]] = []
    for row in rows:
        fp32 = row["fp32"]
        int8 = row["int8"]
        delta = row["delta"]
        csv_rows.append(
            {
                "condition": row["condition"],
                "severity": row["severity"],
                "sample_count": row["sample_count"],
                "fp32_f1": fp32["f1_positive"],
                "int8_f1": int8["f1_positive"],
                "delta_f1": delta["f1_positive"],
                "fp32_recall": fp32["recall_positive"],
                "int8_recall": int8["recall_positive"],
                "delta_recall": delta["recall_positive"],
                "fp32_roc_auc": fp32["roc_auc"],
                "int8_roc_auc": int8["roc_auc"],
                "delta_roc_auc": delta["roc_auc"],
                "fp32_ece": fp32["ece"],
                "int8_ece": int8["ece"],
                "delta_ece": delta["ece"],
                "fp32_mean_risk": fp32["mean_risk_score"],
                "int8_mean_risk": int8["mean_risk_score"],
                "delta_mean_risk": delta["mean_risk_score"],
            }
        )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    json_path.write_text(json.dumps({"metadata": metadata, "results": rows}, indent=2), encoding="utf-8")

    clean = rows[0]
    worst_delta = min(rows, key=lambda item: float(item["delta"]["f1_positive"]))
    max_abs_delta = max(rows, key=lambda item: abs(float(item["delta"]["f1_positive"])))
    lines = [
        "# ASN FP32/INT8 Robustness Consistency",
        "",
        "## Setup",
        "",
        f"- Run: `{metadata['run_id']}`",
        f"- Split: `{metadata['split']}` ({metadata['sample_count']} samples)",
        f"- PTQ backend: `{metadata['backend']}`",
        f"- Shared temperature: {metadata['temperature']:.6f}",
        f"- Shared calibrated threshold: {metadata['decision_threshold']:.6f}",
        "- FP32 and INT8 use identical samples, corruption parameters, and random seeds.",
        "",
        "## Summary",
        "",
        f"- Clean F1: FP32={clean['fp32']['f1_positive']:.4f}, INT8={clean['int8']['f1_positive']:.4f}, delta={clean['delta']['f1_positive']:+.4f}.",
        f"- Most negative F1 delta: {worst_delta['condition']} severity {worst_delta['severity']} ({worst_delta['delta']['f1_positive']:+.4f}).",
        f"- Largest absolute F1 delta: {max_abs_delta['condition']} severity {max_abs_delta['severity']} ({max_abs_delta['delta']['f1_positive']:+.4f}).",
        "",
        "## Results",
        "",
        "| Condition | Severity | FP32 F1 | INT8 F1 | Delta F1 | FP32 recall | INT8 recall | FP32 ECE | INT8 ECE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {condition} | {severity} | {fp32_f1:.4f} | {int8_f1:.4f} | {delta_f1:+.4f} | "
            "{fp32_recall:.4f} | {int8_recall:.4f} | {fp32_ece:.4f} | {int8_ece:.4f} |".format(
                condition=row["condition"],
                severity=row["severity"],
                fp32_f1=float(row["fp32"]["f1_positive"]),
                int8_f1=float(row["int8"]["f1_positive"]),
                delta_f1=float(row["delta"]["f1_positive"]),
                fp32_recall=float(row["fp32"]["recall_positive"]),
                int8_recall=float(row["int8"]["recall_positive"]),
                fp32_ece=float(row["fp32"]["ece"]),
                int8_ece=float(row["int8"]["ece"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "PTQ is acceptable when it preserves the shape of the corruption response and does not introduce a material additional failure mode. Existing degradation failures must still be treated as sensor-quality/OOD problems; parity alone does not make those conditions safe.",
            "",
            "This is a PyTorch CPU quantization experiment. It is not real MCU latency, SRAM, or energy profiling.",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(csv_path)
    print(json_path)
    print(report_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare grouped ASN FP32 and PTQ INT8 robustness.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--calibration-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "asn_student_ptq_robustness")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--calib-split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backend", default="fbgemm", choices=["fbgemm", "qnnpack"])
    parser.add_argument("--max-calib-batches", type=int, default=None)
    parser.add_argument("--sources", nargs="*", default=None)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    args = parser.parse_args()

    torch.backends.quantized.engine = args.backend
    checkpoint = torch.load(args.run_dir / "best.pt", map_location="cpu")
    train_args = as_namespace(checkpoint["args"])
    index = Path(train_args.index)
    if not index.is_absolute():
        index = PROJECT_ROOT / index
    sources = set(args.sources) if args.sources else source_set(getattr(train_args, "sources", None))
    samples = read_samples(index, args.split, sources)
    if not samples:
        raise RuntimeError(f"No samples found for split={args.split}")

    fp32_model = build_student_from_checkpoint(checkpoint)
    quant_base = build_student_from_checkpoint(checkpoint)
    fuse_student_encoder(quant_base.encoder)
    quant_model = QuantizedEncoderWrapper(quant_base)
    quant_model.eval()
    quant_model.qconfig = get_default_qconfig(args.backend)
    prepared = prepare(quant_model, inplace=False)
    calibrate(prepared, make_loader(args, checkpoint, args.calib_split), args.max_calib_batches)
    int8_model = convert(prepared, inplace=False)

    calibration = json.loads(args.calibration_json.read_text(encoding="utf-8"))
    temperature = float(calibration["temperature"])
    threshold = float(calibration["decision_threshold"])
    conditions = [("clean", 0)]
    for condition in ["snr_noise", "low_volume", "clipping", "dropout"]:
        for severity in [1, 2, 3]:
            conditions.append((condition, severity))

    rows: list[dict[str, object]] = []
    for condition, severity in conditions:
        row = evaluate_pair(
            fp32_model,
            int8_model,
            samples,
            int(train_args.sample_rate),
            int(float(train_args.duration) * int(train_args.sample_rate)),
            temperature,
            threshold,
            condition,
            severity,
            args.batch_size,
            args.seed,
            args.bins,
        )
        rows.append(row)
        print(
            f"{condition}:{severity} FP32 F1={row['fp32']['f1_positive']:.4f} "
            f"INT8 F1={row['int8']['f1_positive']:.4f} delta={row['delta']['f1_positive']:+.4f}",
            flush=True,
        )

    metadata = {
        "run_id": args.run_dir.name,
        "split": args.split,
        "sample_count": len(samples),
        "backend": args.backend,
        "temperature": temperature,
        "decision_threshold": threshold,
        "index": str(index),
    }
    write_outputs(rows, metadata, args.output_dir)


if __name__ == "__main__":
    main()
