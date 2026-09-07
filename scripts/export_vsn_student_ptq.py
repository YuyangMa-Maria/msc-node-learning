"""Export and validate the VSN student with post-training quantisation."""

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

from train_vsn_binary import PROJECT_ROOT, VsnBinaryDataset, build_transforms, read_samples
from train_vsn_student_baseline import DepthwiseSeparableBlock, VsnStudentDwCnn, count_parameters


def as_namespace(data: dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(**data)


def source_set(value: object) -> set[str] | None:
    if value is None or value == "all":
        return None
    if isinstance(value, list):
        return set(str(item) for item in value)
    return None


def checkpoint_image_size(checkpoint: dict[str, object]) -> int:
    if "image_size" in checkpoint:
        return int(checkpoint["image_size"])
    args = as_namespace(checkpoint["args"])  # type: ignore[arg-type]
    return int(getattr(args, "image_size"))


def build_student(checkpoint: dict[str, object]) -> nn.Module:
    args = as_namespace(checkpoint["args"])  # type: ignore[arg-type]
    model = VsnStudentDwCnn(width=float(getattr(args, "width", 1.0)), dropout=float(getattr(args, "dropout", 0.1)))
    model.load_state_dict(checkpoint["model"])  # type: ignore[arg-type]
    model.eval()
    return model


class QuantizedStudentWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.quant = QuantStub()
        self.model = model
        self.dequant = DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        x = self.model(x)
        return self.dequant(x)


def fuse_student(model: VsnStudentDwCnn) -> VsnStudentDwCnn:
    model.eval()
    fuse_modules(model.features, [["0", "1", "2"]], inplace=True)
    for module in model.features:
        if isinstance(module, DepthwiseSeparableBlock):
            fuse_modules(module.block, [["0", "1", "2"], ["3", "4", "5"]], inplace=True)
    return model


def get_checkpoint_sources(checkpoint: dict[str, object]) -> set[str] | None:
    args = as_namespace(checkpoint["args"])  # type: ignore[arg-type]
    return source_set(getattr(args, "sources", None))


def get_index(checkpoint: dict[str, object]) -> Path:
    args = as_namespace(checkpoint["args"])  # type: ignore[arg-type]
    index = Path(args.index)
    return index if index.is_absolute() else PROJECT_ROOT / index


def make_loader(index: Path, split: str, sources: set[str] | None, image_size: int, batch_size: int) -> DataLoader:
    _, eval_tf = build_transforms(image_size)
    samples = read_samples(index, split, sources)
    if not samples:
        raise RuntimeError(f"No samples found for split={split} sources={sources}")
    return DataLoader(VsnBinaryDataset(samples, eval_tf), batch_size=batch_size, shuffle=False, num_workers=0)


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


def evaluate(model: nn.Module, loader: DataLoader) -> tuple[dict[str, float | int], float]:
    model.eval()
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    times: list[float] = []
    with torch.no_grad():
        for images, labels in loader:
            start = time.perf_counter()
            logits = model(images).flatten()
            elapsed = time.perf_counter() - start
            times.append(elapsed / images.size(0) * 1000.0)
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())
    metrics = binary_metrics(torch.cat(all_logits), torch.cat(all_labels))
    return metrics, float(sum(times) / max(len(times), 1))


def calibrate(model: nn.Module, loader: DataLoader, max_batches: int | None) -> None:
    model.eval()
    with torch.no_grad():
        for idx, (images, _) in enumerate(loader):
            if max_batches is not None and idx >= max_batches:
                break
            _ = model(images)


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def save_state_dict_size(model: nn.Module, path: Path) -> float:
    torch.save(model.state_dict(), path)
    return file_size_mb(path)


def trace_and_save(model: nn.Module, image_size: int, path: Path) -> float:
    model.eval()
    example = torch.randn(1, 3, image_size, image_size)
    traced = torch.jit.trace(model, example)
    traced.save(str(path))
    return file_size_mb(path)


def run_one(args: argparse.Namespace, run_dir: Path) -> dict[str, object]:
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu")
    image_size = checkpoint_image_size(checkpoint)
    sources = set(args.sources) if args.sources else get_checkpoint_sources(checkpoint)
    index = get_index(checkpoint)
    calib_loader = make_loader(index, args.calib_split, sources, image_size, args.batch_size)
    test_loader = make_loader(index, args.test_split, sources, image_size, args.batch_size)

    fp32_model = build_student(checkpoint)
    fp32_metrics, fp32_ms = evaluate(fp32_model, test_loader)
    fp32_params = count_parameters(fp32_model)

    fused_student = fuse_student(build_student(checkpoint))  # type: ignore[arg-type]
    quant_model = QuantizedStudentWrapper(fused_student)
    quant_model.eval()
    quant_model.qconfig = get_default_qconfig(args.backend)
    prepared = prepare(quant_model, inplace=False)
    calibrate(prepared, calib_loader, args.max_calib_batches)
    int8_model = convert(prepared, inplace=False)
    int8_metrics, int8_ms = evaluate(int8_model, test_loader)

    output_dir = args.output_dir / run_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    fp32_state_path = output_dir / "fp32_state_dict.pt"
    int8_state_path = output_dir / "int8_state_dict.pt"
    int8_script_path = output_dir / "int8_traced.pt"
    fp32_state_mb = save_state_dict_size(fp32_model, fp32_state_path)
    int8_state_mb = save_state_dict_size(int8_model, int8_state_path)
    int8_script_mb = trace_and_save(int8_model, image_size, int8_script_path)

    result: dict[str, object] = {
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "image_size": image_size,
        "sources": sorted(sources) if sources else "all",
        "backend": args.backend,
        "parameters": fp32_params,
        "parameters_m": fp32_params / 1_000_000,
        "fp32_state_dict_mb": fp32_state_mb,
        "int8_state_dict_mb": int8_state_mb,
        "int8_traced_mb": int8_script_mb,
        "fp32_cpu_ms_per_image": fp32_ms,
        "int8_cpu_ms_per_image": int8_ms,
        "fp32_test": fp32_metrics,
        "int8_test": int8_metrics,
        "delta": {
            "accuracy": float(int8_metrics["accuracy"]) - float(fp32_metrics["accuracy"]),
            "recall_positive": float(int8_metrics["recall_positive"]) - float(fp32_metrics["recall_positive"]),
            "f1_positive": float(int8_metrics["f1_positive"]) - float(fp32_metrics["f1_positive"]),
            "roc_auc": float(int8_metrics["roc_auc"]) - float(fp32_metrics["roc_auc"]),
            "cpu_ms_per_image": int8_ms - fp32_ms,
            "state_dict_mb": int8_state_mb - fp32_state_mb,
        },
        "files": {
            "fp32_state_dict": str(fp32_state_path),
            "int8_state_dict": str(int8_state_path),
            "int8_traced": str(int8_script_path),
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
                "image_size": result["image_size"],
                "fp32_accuracy": fp32["accuracy"],
                "int8_accuracy": int8["accuracy"],
                "delta_accuracy": delta["accuracy"],
                "fp32_recall": fp32["recall_positive"],
                "int8_recall": int8["recall_positive"],
                "delta_recall": delta["recall_positive"],
                "fp32_f1": fp32["f1_positive"],
                "int8_f1": int8["f1_positive"],
                "delta_f1": delta["f1_positive"],
                "fp32_roc_auc": fp32["roc_auc"],
                "int8_roc_auc": int8["roc_auc"],
                "delta_roc_auc": delta["roc_auc"],
                "fp32_state_dict_mb": result["fp32_state_dict_mb"],
                "int8_state_dict_mb": result["int8_state_dict_mb"],
                "int8_traced_mb": result["int8_traced_mb"],
                "fp32_cpu_ms": result["fp32_cpu_ms_per_image"],
                "int8_cpu_ms": result["int8_cpu_ms_per_image"],
                "delta_cpu_ms": delta["cpu_ms_per_image"],
            }
        )
    csv_path = output_dir / "summary_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    lines = [
        "# VSN Student PTQ INT8 Evaluation",
        "",
        "## Scope",
        "",
        "- Static post-training quantization (PTQ) was applied to VSN student checkpoints.",
        "- Conv-BN-ReLU blocks were fused before calibration.",
        "- Validation split was used for observer calibration.",
        "- Test split was used for FP32 vs INT8 comparison.",
        "",
        "## Results",
        "",
        "| Run | Image | FP32 F1 | INT8 F1 | Delta F1 | FP32 recall | INT8 recall | INT8 state MB | INT8 traced MB | FP32 CPU ms | INT8 CPU ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {run} | {size} | {fp32_f1:.4f} | {int8_f1:.4f} | {df1:+.4f} | {fp32_rec:.4f} | {int8_rec:.4f} | {int8_mb:.4f} | {script_mb:.4f} | {fp32_ms:.4f} | {int8_ms:.4f} |".format(
                run=row["run_id"],
                size=int(row["image_size"]),
                fp32_f1=float(row["fp32_f1"]),
                int8_f1=float(row["int8_f1"]),
                df1=float(row["delta_f1"]),
                fp32_rec=float(row["fp32_recall"]),
                int8_rec=float(row["int8_recall"]),
                int8_mb=float(row["int8_state_dict_mb"]),
                script_mb=float(row["int8_traced_mb"]),
                fp32_ms=float(row["fp32_cpu_ms"]),
                int8_ms=float(row["int8_cpu_ms"]),
            )
        )
    lines.extend(
        [
            "",
            "## Caution",
            "",
            "- PyTorch INT8 model size and CPU timing are deployment-oriented estimates, not MCU/TFLite Micro tensor-arena profiling.",
            "- If INT8 accuracy drops significantly, QAT should be considered only for the selected student candidate.",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(csv_path)
    print(output_dir / "REPORT.md")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export/evaluate VSN student PTQ INT8 checkpoints.")
    parser.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--sources", nargs="*", default=None)
    parser.add_argument("--calib-split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--test-split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-calib-batches", type=int, default=None)
    parser.add_argument("--backend", default="fbgemm", choices=["fbgemm", "qnnpack"])
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "vsn_student_ptq")
    args = parser.parse_args()

    torch.backends.quantized.engine = args.backend
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = [run_one(args, run_dir) for run_dir in args.run_dirs]
    write_summary(results, args.output_dir)


if __name__ == "__main__":
    main()
