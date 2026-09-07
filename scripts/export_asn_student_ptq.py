"""Export and validate the ASN student with post-training quantisation."""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.ao.quantization import DeQuantStub, QuantStub, convert, fuse_modules, get_default_qconfig, prepare
from torch.utils.data import DataLoader

from train_asn_audio import PROJECT_ROOT, build_loss, read_samples
from train_asn_student import AsnStudentDataset, AsnStudentModel, StudentAudioCnn, count_parameters, limit_samples


def as_namespace(data: dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(**data)


class QuantizedEncoderWrapper(nn.Module):
    def __init__(self, student: AsnStudentModel) -> None:
        super().__init__()
        self.frontend = student.frontend
        self.quant = QuantStub()
        self.encoder = student.encoder
        self.dequant = DeQuantStub()

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        features = self.frontend(waveform)
        features = self.quant(features)
        logits = self.encoder(features)
        return self.dequant(logits)


def fuse_student_encoder(encoder: StudentAudioCnn) -> StudentAudioCnn:
    encoder.eval()
    fuse_modules(encoder.net, [["0", "1", "2"], ["4", "5", "6"], ["8", "9", "10"], ["12", "13", "14"]], inplace=True)
    return encoder


def build_student_from_checkpoint(checkpoint: dict[str, object]) -> AsnStudentModel:
    args = as_namespace(checkpoint["args"])  # type: ignore[arg-type]
    model = AsnStudentModel(
        sample_rate=int(getattr(args, "sample_rate")),
        n_fft=int(getattr(args, "n_fft")),
        hop_length=int(getattr(args, "hop_length")),
        n_mels=int(getattr(args, "n_mels")),
        width=float(getattr(args, "width")),
        dropout=float(getattr(args, "dropout")),
    )
    model.load_state_dict(checkpoint["model"])  # type: ignore[arg-type]
    model.eval()
    return model


def make_loader(args: argparse.Namespace, checkpoint: dict[str, object], split: str) -> DataLoader:
    train_args = as_namespace(checkpoint["args"])  # type: ignore[arg-type]
    sources = set(args.sources) if args.sources else None
    if sources is None:
        checkpoint_sources = getattr(train_args, "sources", None)
        if isinstance(checkpoint_sources, list):
            sources = set(str(item) for item in checkpoint_sources)
    index = Path(getattr(train_args, "index"))
    if not index.is_absolute():
        index = PROJECT_ROOT / index
    samples = read_samples(index, split, sources)
    limit = {"train": args.max_train, "val": args.max_val, "test": args.max_test}[split]
    samples = limit_samples(samples, limit, int(getattr(train_args, "seed", 42)))
    if not samples:
        raise RuntimeError(f"No ASN samples found for split={split}")
    dataset = AsnStudentDataset(
        samples=samples,
        sample_rate=int(getattr(train_args, "sample_rate")),
        student_duration=float(getattr(train_args, "duration")),
        teacher_duration=float(getattr(train_args, "teacher_duration", 10.0)),
        train=False,
        seed=int(getattr(train_args, "seed", 42)),
    )
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)


def binary_metrics(logits: torch.Tensor, labels: torch.Tensor, threshold: float) -> dict[str, float | int]:
    probs = torch.sigmoid(logits).detach().cpu()
    y = labels.detach().cpu().int()
    pred = (probs >= threshold).int()
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
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "threshold": threshold,
    }


def evaluate(model: nn.Module, loader: DataLoader, threshold: float) -> tuple[dict[str, float | int], float]:
    model.eval()
    logits_all: list[torch.Tensor] = []
    labels_all: list[torch.Tensor] = []
    times: list[float] = []
    with torch.no_grad():
        for student_waveform, _, labels in loader:
            start = time.perf_counter()
            logits = model(student_waveform).flatten()
            elapsed = time.perf_counter() - start
            times.append(elapsed / student_waveform.size(0) * 1000.0)
            logits_all.append(logits.detach().cpu())
            labels_all.append(labels.detach().cpu())
    return binary_metrics(torch.cat(logits_all), torch.cat(labels_all), threshold), float(sum(times) / max(len(times), 1))


def calibrate(model: nn.Module, loader: DataLoader, max_batches: int | None) -> None:
    model.eval()
    with torch.no_grad():
        for idx, (student_waveform, _, _) in enumerate(loader):
            if max_batches is not None and idx >= max_batches:
                break
            _ = model(student_waveform)


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def save_state_dict_size(model: nn.Module, path: Path) -> float:
    torch.save(model.state_dict(), path)
    return file_size_mb(path)


def trace_and_save(model: nn.Module, duration: float, sample_rate: int, path: Path) -> float | None:
    try:
        model.eval()
        example = torch.randn(1, int(duration * sample_rate))
        traced = torch.jit.trace(model, example)
        traced.save(str(path))
        return file_size_mb(path)
    except Exception as exc:  # noqa: BLE001
        (path.with_suffix(".trace_error.txt")).write_text(str(exc), encoding="utf-8")
        return None


def run_one(args: argparse.Namespace, run_dir: Path) -> dict[str, object]:
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu")
    train_args = as_namespace(checkpoint["args"])  # type: ignore[arg-type]
    threshold = float(checkpoint["val_metrics"].get("threshold", 0.5))  # type: ignore[index]
    calib_loader = make_loader(args, checkpoint, args.calib_split)
    test_loader = make_loader(args, checkpoint, args.test_split)

    fp32_model = build_student_from_checkpoint(checkpoint)
    fp32_metrics, fp32_ms = evaluate(fp32_model, test_loader, threshold)
    params = count_parameters(fp32_model)

    quant_base = build_student_from_checkpoint(checkpoint)
    fuse_student_encoder(quant_base.encoder)  # type: ignore[arg-type]
    quant_model = QuantizedEncoderWrapper(quant_base)
    quant_model.eval()
    quant_model.qconfig = get_default_qconfig(args.backend)
    prepared = prepare(quant_model, inplace=False)
    calibrate(prepared, calib_loader, args.max_calib_batches)
    int8_model = convert(prepared, inplace=False)
    int8_metrics, int8_ms = evaluate(int8_model, test_loader, threshold)

    output_dir = args.output_dir / run_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    fp32_state_mb = save_state_dict_size(fp32_model, output_dir / "fp32_state_dict.pt")
    int8_state_mb = save_state_dict_size(int8_model, output_dir / "int8_state_dict.pt")
    int8_traced_mb = trace_and_save(
        int8_model,
        duration=float(getattr(train_args, "duration")),
        sample_rate=int(getattr(train_args, "sample_rate")),
        path=output_dir / "int8_traced.pt",
    )

    result: dict[str, object] = {
        "run_id": run_dir.name,
        "duration": float(getattr(train_args, "duration")),
        "width": float(getattr(train_args, "width")),
        "training_mode": checkpoint.get("training_mode", "unknown"),
        "backend": args.backend,
        "parameters": params,
        "parameters_m": params / 1_000_000,
        "decision_threshold": threshold,
        "fp32_state_dict_mb": fp32_state_mb,
        "int8_state_dict_mb": int8_state_mb,
        "int8_traced_mb": int8_traced_mb,
        "fp32_cpu_ms_per_audio": fp32_ms,
        "int8_cpu_ms_per_audio": int8_ms,
        "fp32_test": fp32_metrics,
        "int8_test": int8_metrics,
        "delta": {
            "accuracy": float(int8_metrics["accuracy"]) - float(fp32_metrics["accuracy"]),
            "recall_positive": float(int8_metrics["recall_positive"]) - float(fp32_metrics["recall_positive"]),
            "f1_positive": float(int8_metrics["f1_positive"]) - float(fp32_metrics["f1_positive"]),
            "roc_auc": float(int8_metrics["roc_auc"]) - float(fp32_metrics["roc_auc"]),
            "cpu_ms_per_audio": int8_ms - fp32_ms,
            "state_dict_mb": int8_state_mb - fp32_state_mb,
        },
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def write_summary(results: list[dict[str, object]], output_dir: Path) -> None:
    rows = []
    for result in results:
        fp32 = result["fp32_test"]  # type: ignore[assignment]
        int8 = result["int8_test"]  # type: ignore[assignment]
        delta = result["delta"]  # type: ignore[assignment]
        rows.append(
            {
                "run_id": result["run_id"],
                "training_mode": result["training_mode"],
                "duration": result["duration"],
                "width": result["width"],
                "fp32_f1": fp32["f1_positive"],
                "int8_f1": int8["f1_positive"],
                "delta_f1": delta["f1_positive"],
                "fp32_recall": fp32["recall_positive"],
                "int8_recall": int8["recall_positive"],
                "delta_recall": delta["recall_positive"],
                "fp32_accuracy": fp32["accuracy"],
                "int8_accuracy": int8["accuracy"],
                "delta_accuracy": delta["accuracy"],
                "fp32_roc_auc": fp32["roc_auc"],
                "int8_roc_auc": int8["roc_auc"],
                "delta_roc_auc": delta["roc_auc"],
                "fp32_state_dict_mb": result["fp32_state_dict_mb"],
                "int8_state_dict_mb": result["int8_state_dict_mb"],
                "int8_traced_mb": "" if result["int8_traced_mb"] is None else result["int8_traced_mb"],
                "fp32_cpu_ms": result["fp32_cpu_ms_per_audio"],
                "int8_cpu_ms": result["int8_cpu_ms_per_audio"],
                "delta_cpu_ms": delta["cpu_ms_per_audio"],
            }
        )
    csv_path = output_dir / "summary_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    lines = [
        "# ASN Student PTQ INT8 Evaluation",
        "",
        "## Scope",
        "",
        "- Static PTQ was applied to the ASN student CNN encoder.",
        "- The log-mel frontend remains FP32 in this PyTorch export; this mirrors a DSP/frontend plus quantized classifier deployment route.",
        "- Validation split was used for calibration; test split was used for FP32 versus INT8 comparison.",
        "",
        "## Results",
        "",
        "| Run | Mode | Duration | FP32 F1 | INT8 F1 | Delta F1 | FP32 recall | INT8 recall | INT8 state MB | FP32 CPU ms | INT8 CPU ms |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {run} | {mode} | {duration:.1f}s | {fp32_f1:.4f} | {int8_f1:.4f} | {df1:+.4f} | {fp32_rec:.4f} | {int8_rec:.4f} | {int8_mb:.4f} | {fp32_ms:.4f} | {int8_ms:.4f} |".format(
                run=row["run_id"],
                mode=row["training_mode"],
                duration=float(row["duration"]),
                fp32_f1=float(row["fp32_f1"]),
                int8_f1=float(row["int8_f1"]),
                df1=float(row["delta_f1"]),
                fp32_rec=float(row["fp32_recall"]),
                int8_rec=float(row["int8_recall"]),
                int8_mb=float(row["int8_state_dict_mb"]),
                fp32_ms=float(row["fp32_cpu_ms"]),
                int8_ms=float(row["int8_cpu_ms"]),
            )
        )
    lines.extend(
        [
            "",
            "## Caution",
            "",
            "- These are PyTorch CPU INT8 estimates, not MCU/TFLite Micro profiling.",
            "- Because the frontend is not quantized here, reported model files should be interpreted as deployment-oriented artifacts rather than full embedded memory measurements.",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(csv_path)
    print(output_dir / "REPORT.md")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export/evaluate ASN student PTQ INT8 checkpoints.")
    parser.add_argument("--run-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--sources", nargs="*", default=None)
    parser.add_argument("--calib-split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--test-split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-calib-batches", type=int, default=None)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--backend", default="fbgemm", choices=["fbgemm", "qnnpack"])
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "asn_student_ptq")
    args = parser.parse_args()

    torch.backends.quantized.engine = args.backend
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = [run_one(args, run_dir) for run_dir in args.run_dirs]
    write_summary(results, args.output_dir)


if __name__ == "__main__":
    main()
