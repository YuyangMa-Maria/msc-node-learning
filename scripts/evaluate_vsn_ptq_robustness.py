"""Compare VSN FP32 and PTQ predictions under image degradation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.ao.quantization import convert, get_default_qconfig, prepare
from torch.utils.data import DataLoader

from evaluate_vsn_student_robustness import (
    RobustVsnDataset,
    calibrated_metrics,
    load_temperature,
    plot_robustness,
    write_csv,
)
from export_vsn_student_ptq import (
    QuantizedStudentWrapper,
    build_student,
    calibrate,
    fuse_student,
    get_checkpoint_sources,
    get_index,
    make_loader,
)
from train_vsn_binary import PROJECT_ROOT, build_transforms, read_samples


def as_namespace(data: dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(**data)


def checkpoint_image_size(checkpoint: dict[str, object]) -> int:
    if "image_size" in checkpoint:
        return int(checkpoint["image_size"])
    args = as_namespace(checkpoint["args"])  # type: ignore[arg-type]
    return int(getattr(args, "image_size"))


def evaluate_condition(
    model: torch.nn.Module,
    samples,
    transform,
    corruption: str,
    severity: int,
    temperature: float,
    batch_size: int,
    bins: int,
    seed: int,
) -> dict[str, object]:
    loader = DataLoader(
        RobustVsnDataset(samples, transform, corruption, severity, seed),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    logits_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            logits_list.append(model(images).flatten().detach().cpu())
            labels_list.append(labels.detach().cpu())
    metrics = calibrated_metrics(torch.cat(logits_list), torch.cat(labels_list), temperature, bins)
    metrics["corruption"] = corruption
    metrics["severity"] = severity
    metrics["sample_count"] = len(samples)
    return metrics


def read_fp32_rows(path: Path | None) -> dict[tuple[str, int], dict[str, float]]:
    if path is None or not path.exists():
        return {}
    rows: dict[tuple[str, int], dict[str, float]] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["corruption"], int(row["severity"]))
            rows[key] = {
                "f1_positive": float(row["f1_positive"]),
                "recall_positive": float(row["recall_positive"]),
                "ece": float(row["ece"]),
            }
    return rows


def write_markdown(
    rows: list[dict[str, object]],
    output_path: Path,
    metadata: dict[str, object],
    fp32_rows: dict[tuple[str, int], dict[str, float]],
) -> None:
    clean = next(row for row in rows if row["corruption"] == "clean")
    worst = min([row for row in rows if row["corruption"] != "clean"], key=lambda row: float(row["f1_positive"]))
    lines = [
        "# VSN Student PTQ INT8 Robustness Evaluation",
        "",
        f"- Run: `{metadata['run_id']}`",
        f"- Image size: {metadata['image_size']}",
        f"- Backend: {metadata['backend']}",
        f"- Temperature: {metadata['temperature']:.4f}",
        "- Risk score: calibrated `sigmoid(logit / T)`",
        "",
        "## Summary",
        "",
        f"- INT8 clean F1: {clean['f1_positive']:.4f}, clean ECE: {clean['ece']:.6f}",
        f"- Worst INT8 condition by F1: {worst['corruption']} severity {worst['severity']} with F1={worst['f1_positive']:.4f}, recall={worst['recall_positive']:.4f}, ECE={worst['ece']:.6f}",
    ]
    clean_key = ("clean", 0)
    if clean_key in fp32_rows:
        delta = float(clean["f1_positive"]) - fp32_rows[clean_key]["f1_positive"]
        lines.append(f"- Clean F1 delta versus FP32 robustness run: {delta:+.4f}")
    lines.extend(
        [
            "",
            "## Results",
            "",
            "| Corruption | Severity | INT8 F1 | FP32 F1 | Delta F1 | INT8 recall | INT8 ECE | Confusion matrix (TN/FP/FN/TP) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in rows:
        key = (str(row["corruption"]), int(row["severity"]))
        fp32_f1 = fp32_rows.get(key, {}).get("f1_positive")
        delta_f1 = None if fp32_f1 is None else float(row["f1_positive"]) - fp32_f1
        lines.append(
            "| {corr} | {sev} | {f1:.4f} | {fp32} | {delta} | {rec:.4f} | {ece:.6f} | {tn}/{fp}/{fn}/{tp} |".format(
                corr=row["corruption"],
                sev=row["severity"],
                f1=float(row["f1_positive"]),
                fp32="n/a" if fp32_f1 is None else f"{fp32_f1:.4f}",
                delta="n/a" if delta_f1 is None else f"{delta_f1:+.4f}",
                rec=float(row["recall_positive"]),
                ece=float(row["ece"]),
                tn=row["tn"],
                fp=row["fp"],
                fn=row["fn"],
                tp=row["tp"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This is a deployment-oriented PyTorch INT8 robustness check. It should be used to decide whether PTQ introduces obvious degradation before considering QAT. It is not real embedded hardware validation.",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VSN PTQ INT8 robustness under visual corruptions.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--calibration-json", type=Path, required=True)
    parser.add_argument("--fp32-robustness-csv", type=Path, default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--calib-split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backend", default="fbgemm", choices=["fbgemm", "qnnpack"])
    parser.add_argument("--max-calib-batches", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "vsn_student_ptq_robustness")
    args = parser.parse_args()

    torch.backends.quantized.engine = args.backend
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.run_dir / "best.pt", map_location="cpu")
    image_size = checkpoint_image_size(checkpoint)
    sources = get_checkpoint_sources(checkpoint)
    index = get_index(checkpoint)
    samples = read_samples(index, args.split, sources)
    if not samples:
        raise RuntimeError(f"No samples found for split={args.split} sources={sources}")

    calib_loader = make_loader(index, args.calib_split, sources, image_size, args.batch_size)
    fused_student = fuse_student(build_student(checkpoint))  # type: ignore[arg-type]
    quant_model = QuantizedStudentWrapper(fused_student)
    quant_model.eval()
    quant_model.qconfig = get_default_qconfig(args.backend)
    prepared = prepare(quant_model, inplace=False)
    calibrate(prepared, calib_loader, args.max_calib_batches)
    int8_model = convert(prepared, inplace=False)

    _, eval_tf = build_transforms(image_size)
    temperature = load_temperature(args.calibration_json)
    conditions = [("clean", 0)]
    for corruption in ["low_light", "blur", "gaussian_noise", "occlusion", "jpeg"]:
        for severity in [1, 2, 3]:
            conditions.append((corruption, severity))

    rows = []
    for corruption, severity in conditions:
        result = evaluate_condition(
            int8_model,
            samples,
            eval_tf,
            corruption,
            severity,
            temperature,
            args.batch_size,
            args.bins,
            args.seed,
        )
        rows.append(result)
        print(
            f"{args.run_dir.name} INT8 {corruption}:{severity} "
            f"f1={result['f1_positive']:.4f} recall={result['recall_positive']:.4f} ece={result['ece']:.6f}",
            flush=True,
        )

    stem = f"{args.run_dir.name}_{args.split}_ptq_int8_robustness"
    csv_path = args.output_dir / f"{stem}.csv"
    json_path = args.output_dir / f"{stem}.json"
    md_path = args.output_dir / f"{stem}.md"
    plot_path = args.output_dir / f"{stem}.png"
    metadata = {
        "run_id": args.run_dir.name,
        "image_size": image_size,
        "sources": sorted(sources) if sources else "all",
        "split": args.split,
        "backend": args.backend,
        "temperature": temperature,
    }
    write_csv(rows, csv_path)
    json_path.write_text(json.dumps({"metadata": metadata, "results": rows}, indent=2), encoding="utf-8")
    plot_robustness(rows, plot_path)
    write_markdown(rows, md_path, metadata, read_fp32_rows(args.fp32_robustness_csv))
    print(csv_path)
    print(json_path)
    print(plot_path)
    print(md_path)


if __name__ == "__main__":
    main()
