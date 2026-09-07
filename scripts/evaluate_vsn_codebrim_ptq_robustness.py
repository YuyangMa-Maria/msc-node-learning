"""Assess multi-task VSN quantisation and controlled image degradation."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch import nn
from torch.ao.quantization import (
    DeQuantStub,
    QuantStub,
    convert,
    fuse_modules,
    get_default_qconfig,
    prepare,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from evaluate_vsn_sdnet_zero_shot import binary_metrics, sigmoid
from train_vsn_codebrim_students import IMAGENET_MEAN, IMAGENET_STD, read_rows
from train_vsn_student_baseline import (
    DepthwiseSeparableBlock,
    VsnStudentDwCnn,
    count_parameters,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = (
    ("clean", 0),
    ("low_light", 1),
    ("low_light", 2),
    ("low_light", 3),
    ("blur", 1),
    ("blur", 2),
    ("blur", 3),
    ("gaussian_noise", 1),
    ("gaussian_noise", 2),
    ("gaussian_noise", 3),
    ("occlusion", 1),
    ("occlusion", 2),
    ("occlusion", 3),
    ("jpeg", 1),
    ("jpeg", 2),
    ("jpeg", 3),
)


def apply_corruption(
    image: Image.Image,
    corruption: str,
    severity: int,
    seed: int,
) -> Image.Image:
    if corruption == "clean":
        return image
    if corruption == "low_light":
        return ImageEnhance.Brightness(image).enhance(
            {1: 0.7, 2: 0.45, 3: 0.25}[severity]
        )
    if corruption == "blur":
        return image.filter(
            ImageFilter.GaussianBlur(radius={1: 1.0, 2: 2.0, 3: 3.5}[severity])
        )
    if corruption == "gaussian_noise":
        rng = np.random.default_rng(seed)
        array = np.asarray(image).astype(np.float32) / 255.0
        sigma = {1: 0.03, 2: 0.07, 3: 0.12}[severity]
        array = np.clip(array + rng.normal(0.0, sigma, size=array.shape), 0.0, 1.0)
        return Image.fromarray((array * 255).astype(np.uint8))
    if corruption == "occlusion":
        rng = random.Random(seed)
        array = np.asarray(image.copy()).copy()
        height, width = array.shape[:2]
        side = int(min(height, width) * {1: 0.12, 2: 0.22, 3: 0.35}[severity])
        x0 = rng.randint(0, max(width - side, 0))
        y0 = rng.randint(0, max(height - side, 0))
        array[y0 : y0 + side, x0 : x0 + side] = 0
        return Image.fromarray(array)
    if corruption == "jpeg":
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality={1: 60, 2: 35, 3: 15}[severity])
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")
    raise ValueError(f"Unsupported corruption: {corruption}")


class CodebrimRobustnessDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        image_size: int,
        corruption: str,
        severity: int,
        seed: int,
    ) -> None:
        self.rows = rows
        self.corruption = corruption
        self.severity = severity
        self.seed = seed
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        row = self.rows[index]
        with Image.open(PROJECT_ROOT / row["path"]) as source:
            image = source.convert("RGB")
            image = apply_corruption(
                image,
                self.corruption,
                self.severity,
                self.seed + index,
            )
            tensor = self.transform(image)
        return tensor, torch.tensor(int(row["damage"])), index


class QuantizedStudentWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.quant = QuantStub()
        self.model = model
        self.dequant = DeQuantStub()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.dequant(self.model(self.quant(inputs)))


def build_student(checkpoint: dict[str, Any]) -> VsnStudentDwCnn:
    model = VsnStudentDwCnn(width=1.0, dropout=0.1)
    state = checkpoint.get("student_model")
    if state is None:
        raise KeyError("CODEBRIM checkpoint is missing student_model")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def fuse_student(model: VsnStudentDwCnn) -> VsnStudentDwCnn:
    model.eval()
    fuse_modules(model.features, [["0", "1", "2"]], inplace=True)
    for module in model.features:
        if isinstance(module, DepthwiseSeparableBlock):
            fuse_modules(
                module.block,
                [["0", "1", "2"], ["3", "4", "5"]],
                inplace=True,
            )
    return model


def make_loader(
    rows: list[dict[str, str]],
    image_size: int,
    corruption: str,
    severity: int,
    seed: int,
    batch_size: int,
) -> DataLoader:
    return DataLoader(
        CodebrimRobustnessDataset(
            rows,
            image_size=image_size,
            corruption=corruption,
            severity=severity,
            seed=seed,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )


def stratified_calibration_rows(
    rows: list[dict[str, str]],
    count: int,
    seed: int,
) -> list[dict[str, str]]:
    rng = random.Random(seed)
    negatives = [row for row in rows if int(row["damage"]) == 0]
    positives = [row for row in rows if int(row["damage"]) == 1]
    rng.shuffle(negatives)
    rng.shuffle(positives)
    negative_count = min(len(negatives), count // 2)
    positive_count = min(len(positives), count - negative_count)
    selected = negatives[:negative_count] + positives[:positive_count]
    if len(selected) < count:
        used = {row["path"] for row in selected}
        remaining = [row for row in rows if row["path"] not in used]
        rng.shuffle(remaining)
        selected.extend(remaining[: count - len(selected)])
    rng.shuffle(selected)
    return selected


def quantize_model(
    checkpoint: dict[str, Any],
    calibration_loader: DataLoader,
    backend: str,
) -> QuantizedStudentWrapper:
    torch.backends.quantized.engine = backend
    wrapper = QuantizedStudentWrapper(fuse_student(build_student(checkpoint)))
    wrapper.eval()
    wrapper.qconfig = get_default_qconfig(backend)
    prepared = prepare(wrapper, inplace=False)
    with torch.inference_mode():
        for images, _, _ in calibration_loader:
            prepared(images)
    quantized = convert(prepared, inplace=False)
    quantized.eval()
    return quantized


def collect_logits(
    model: nn.Module,
    loader: DataLoader,
) -> tuple[np.ndarray, np.ndarray]:
    logits = np.empty(len(loader.dataset), dtype=np.float32)
    labels = np.empty(len(loader.dataset), dtype=np.int64)
    model.eval()
    with torch.inference_mode():
        for images, batch_labels, indices in loader:
            batch_logits = model(images).flatten().detach().cpu().numpy()
            index_values = indices.numpy()
            logits[index_values] = batch_logits
            labels[index_values] = batch_labels.numpy().astype(np.int64)
    return logits, labels


def probability_consistency(
    fp32_probabilities: np.ndarray,
    int8_probabilities: np.ndarray,
) -> dict[str, float]:
    difference = int8_probabilities - fp32_probabilities
    if float(np.std(fp32_probabilities)) > 0.0 and float(
        np.std(int8_probabilities)
    ) > 0.0:
        correlation = float(np.corrcoef(fp32_probabilities, int8_probabilities)[0, 1])
    else:
        correlation = float("nan")
    return {
        "probability_mae": float(np.mean(np.abs(difference))),
        "probability_rmse": float(np.sqrt(np.mean(difference**2))),
        "probability_max_abs": float(np.max(np.abs(difference))),
        "probability_mean_signed_delta": float(np.mean(difference)),
        "probability_pearson": correlation,
        "decision_agreement": float(
            np.mean(
                (fp32_probabilities >= 0.5).astype(np.int64)
                == (int8_probabilities >= 0.5).astype(np.int64)
            )
        ),
    }


def metric_delta(
    fp32_metrics: dict[str, float | int],
    int8_metrics: dict[str, float | int],
) -> dict[str, float]:
    names = (
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "f1_positive",
        "roc_auc",
        "average_precision",
        "brier",
        "ece_15",
    )
    return {
        name: float(int8_metrics[name]) - float(fp32_metrics[name]) for name in names
    }


def state_dict_size_mb(model: nn.Module, path: Path) -> float:
    torch.save(model.state_dict(), path)
    return path.stat().st_size / (1024 * 1024)


def traced_size_mb(model: nn.Module, image_size: int, path: Path) -> float:
    traced = torch.jit.trace(model, torch.randn(1, 3, image_size, image_size))
    traced.save(str(path))
    return path.stat().st_size / (1024 * 1024)


def benchmark(
    model: nn.Module,
    image_size: int,
    warmup: int,
    repetitions: int,
) -> dict[str, float | int]:
    sample = torch.randn(1, 3, image_size, image_size)
    model.eval()
    torch.set_num_threads(1)
    with torch.inference_mode():
        for _ in range(warmup):
            model(sample)
        timings = []
        for _ in range(repetitions):
            start = time.perf_counter()
            model(sample)
            timings.append((time.perf_counter() - start) * 1000.0)
    values = np.asarray(timings)
    return {
        "repetitions": repetitions,
        "threads": 1,
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.quantile(values, 0.95)),
        "mean_ms": float(np.mean(values)),
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "corruption",
        "severity",
        "fp32_macro_f1",
        "int8_macro_f1",
        "delta_macro_f1",
        "fp32_balanced_accuracy",
        "int8_balanced_accuracy",
        "delta_balanced_accuracy",
        "fp32_roc_auc",
        "int8_roc_auc",
        "delta_roc_auc",
        "fp32_ece",
        "int8_ece",
        "delta_ece",
        "probability_mae",
        "decision_agreement",
        "probability_pearson",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            fp32 = row["fp32"]
            int8 = row["int8"]
            delta = row["delta"]
            consistency = row["consistency"]
            writer.writerow(
                {
                    "corruption": row["corruption"],
                    "severity": row["severity"],
                    "fp32_macro_f1": fp32["macro_f1"],
                    "int8_macro_f1": int8["macro_f1"],
                    "delta_macro_f1": delta["macro_f1"],
                    "fp32_balanced_accuracy": fp32["balanced_accuracy"],
                    "int8_balanced_accuracy": int8["balanced_accuracy"],
                    "delta_balanced_accuracy": delta["balanced_accuracy"],
                    "fp32_roc_auc": fp32["roc_auc"],
                    "int8_roc_auc": int8["roc_auc"],
                    "delta_roc_auc": delta["roc_auc"],
                    "fp32_ece": fp32["ece_15"],
                    "int8_ece": int8["ece_15"],
                    "delta_ece": delta["ece_15"],
                    "probability_mae": consistency["probability_mae"],
                    "decision_agreement": consistency["decision_agreement"],
                    "probability_pearson": consistency["probability_pearson"],
                }
            )


def write_markdown(report: dict[str, Any], path: Path) -> None:
    clean = report["conditions"][0]
    worst = min(
        report["conditions"],
        key=lambda item: float(item["fp32"]["macro_f1"]),
    )
    lines = [
        "# CODEBRIM Binary-KD Student PTQ and Robustness",
        "",
        "The primary 15,017-parameter Binary-KD Student is evaluated on the frozen "
        "official CODEBRIM test split. Static eager-mode INT8 PTQ uses only a "
        "stratified subset of the training split for observer calibration.",
        "",
        "## Clean INT8 consistency",
        "",
        "| Metric | FP32 | INT8 | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for metric in (
        "macro_f1",
        "balanced_accuracy",
        "roc_auc",
        "average_precision",
        "brier",
        "ece_15",
    ):
        lines.append(
            f"| {metric} | {clean['fp32'][metric]:.4f} | "
            f"{clean['int8'][metric]:.4f} | {clean['delta'][metric]:+.4f} |"
        )
    lines.extend(
        [
            "",
            f"Clean probability MAE is "
            f"`{clean['consistency']['probability_mae']:.6f}` and decision agreement "
            f"is `{clean['consistency']['decision_agreement']:.4f}`.",
            "",
            "## Controlled degradation",
            "",
            "| Condition | FP32 Macro-F1 | INT8 Macro-F1 | FP32 ECE | INT8 ECE | Agreement |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in report["conditions"]:
        condition = (
            "clean"
            if row["corruption"] == "clean"
            else f"{row['corruption']}-s{row['severity']}"
        )
        lines.append(
            f"| {condition} | {row['fp32']['macro_f1']:.4f} | "
            f"{row['int8']['macro_f1']:.4f} | {row['fp32']['ece_15']:.4f} | "
            f"{row['int8']['ece_15']:.4f} | "
            f"{row['consistency']['decision_agreement']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Diagnostic decision",
            "",
            f"- QAT trigger: **{report['diagnostic_decision']['qat_recommended']}**.",
            f"- Worst FP32 condition by Macro-F1: "
            f"`{worst['corruption']}-s{worst['severity']}` at "
            f"`{worst['fp32']['macro_f1']:.4f}`.",
            f"- FP32 state dict: `{report['artifacts']['fp32_state_dict_mb']:.4f}` MB; "
            f"INT8 state dict: `{report['artifacts']['int8_state_dict_mb']:.4f}` MB.",
            f"- One-thread desktop CPU p95 latency: FP32 "
            f"`{report['latency']['fp32']['p95_ms']:.3f}` ms, INT8 "
            f"`{report['latency']['int8']['p95_ms']:.3f}` ms.",
            "",
            "## Claims boundary",
            "",
            "These are controlled synthetic corruptions and PyTorch CPU eager-mode "
            "quantisation results. They do not constitute real post-disaster field "
            "validation or ESP-DL hardware profiling. Desktop CPU latency must not be "
            "reported as ESP32-S3 latency.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate CODEBRIM Binary-KD Student PTQ and robustness."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--index",
        type=Path,
        default=PROJECT_ROOT
        / "experiments"
        / "vsn_codebrim"
        / "codebrim_multitask_index.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "vsn_codebrim_ptq_robustness",
    )
    parser.add_argument("--backend", choices=["fbgemm", "qnnpack"], default="fbgemm")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--calibration-samples", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--latency-warmup", type=int, default=30)
    parser.add_argument("--latency-repetitions", type=int, default=200)
    args = parser.parse_args()

    checkpoint = torch.load(
        args.run_dir / "best.pt",
        map_location="cpu",
        weights_only=False,
    )
    result = json.loads((args.run_dir / "result.json").read_text(encoding="utf-8"))
    image_size = int(checkpoint["student_image_size"])
    temperature = float(result["calibration"]["temperature"])
    rows = read_rows(args.index)
    train_rows = [row for row in rows if row["split"] == "train"]
    test_rows = [row for row in rows if row["split"] == "test"]
    calibration_rows = stratified_calibration_rows(
        train_rows,
        min(args.calibration_samples, len(train_rows)),
        args.seed,
    )
    calibration_loader = make_loader(
        calibration_rows,
        image_size,
        "clean",
        0,
        args.seed,
        args.batch_size,
    )

    fp32_model = build_student(checkpoint)
    int8_model = quantize_model(checkpoint, calibration_loader, args.backend)
    if count_parameters(fp32_model) != 15017:
        raise RuntimeError("Unexpected deployment parameter count")

    condition_results: list[dict[str, Any]] = []
    for corruption, severity in CONDITIONS:
        print(f"Evaluating {corruption} severity={severity}", flush=True)
        loader = make_loader(
            test_rows,
            image_size,
            corruption,
            severity,
            args.seed,
            args.batch_size,
        )
        fp32_logits, labels = collect_logits(fp32_model, loader)
        int8_logits, int8_labels = collect_logits(int8_model, loader)
        if not np.array_equal(labels, int8_labels):
            raise RuntimeError("FP32 and INT8 label order mismatch")
        fp32_probabilities = sigmoid(fp32_logits / temperature)
        int8_probabilities = sigmoid(int8_logits / temperature)
        fp32_metrics = binary_metrics(labels, fp32_probabilities)
        int8_metrics = binary_metrics(labels, int8_probabilities)
        condition_results.append(
            {
                "corruption": corruption,
                "severity": severity,
                "sample_count": len(labels),
                "fp32": fp32_metrics,
                "int8": int8_metrics,
                "delta": metric_delta(fp32_metrics, int8_metrics),
                "consistency": probability_consistency(
                    fp32_probabilities,
                    int8_probabilities,
                ),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fp32_state_path = args.output_dir / "binary_kd_seed2026_fp32_state_dict.pt"
    int8_state_path = args.output_dir / "binary_kd_seed2026_int8_state_dict.pt"
    int8_traced_path = args.output_dir / "binary_kd_seed2026_int8_traced.pt"
    fp32_size = state_dict_size_mb(fp32_model, fp32_state_path)
    int8_size = state_dict_size_mb(int8_model, int8_state_path)
    int8_traced_size = traced_size_mb(int8_model, image_size, int8_traced_path)
    latency = {
        "fp32": benchmark(
            fp32_model,
            image_size,
            args.latency_warmup,
            args.latency_repetitions,
        ),
        "int8": benchmark(
            int8_model,
            image_size,
            args.latency_warmup,
            args.latency_repetitions,
        ),
    }

    clean = condition_results[0]
    qat_triggers = {
        "clean_macro_f1_drop_gt_0.01": clean["delta"]["macro_f1"] < -0.01,
        "clean_auroc_drop_gt_0.01": clean["delta"]["roc_auc"] < -0.01,
        "clean_decision_agreement_lt_0.99": (
            clean["consistency"]["decision_agreement"] < 0.99
        ),
        "any_condition_macro_f1_extra_drop_gt_0.02": any(
            row["delta"]["macro_f1"] < -0.02 for row in condition_results
        ),
    }
    report = {
        "protocol": {
            "run_dir": str(args.run_dir),
            "index": str(args.index),
            "checkpoint_selection": "best validation binary macro-F1",
            "test_used_for_selection": False,
            "image_size": image_size,
            "deployed_parameters": count_parameters(fp32_model),
            "temperature_from_validation": temperature,
            "decision_threshold": 0.5,
            "ptq_backend": args.backend,
            "ptq_calibration_split": "train",
            "ptq_calibration_samples": len(calibration_rows),
            "ptq_calibration_class_counts": {
                "negative": sum(int(row["damage"]) == 0 for row in calibration_rows),
                "positive": sum(int(row["damage"]) == 1 for row in calibration_rows),
            },
            "robustness_split": "frozen official test",
            "robustness_seed": args.seed,
        },
        "conditions": condition_results,
        "latency": latency,
        "artifacts": {
            "fp32_state_dict": str(fp32_state_path),
            "int8_state_dict": str(int8_state_path),
            "int8_traced": str(int8_traced_path),
            "fp32_state_dict_mb": fp32_size,
            "int8_state_dict_mb": int8_size,
            "int8_traced_mb": int8_traced_size,
            "state_dict_reduction_fraction": 1.0 - int8_size / fp32_size,
        },
        "diagnostic_decision": {
            "qat_recommended": any(qat_triggers.values()),
            "triggers": qat_triggers,
            "note": (
                "These are engineering diagnostic triggers, not safety certification "
                "criteria."
            ),
        },
        "claims_boundary": (
            "Controlled synthetic corruptions and PyTorch CPU eager-mode PTQ do not "
            "constitute real disaster validation or ESP-DL hardware profiling."
        ),
    }
    json_path = args.output_dir / "REPORT.json"
    csv_path = args.output_dir / "condition_metrics.csv"
    md_path = args.output_dir / "REPORT.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(condition_results, csv_path)
    write_markdown(report, md_path)
    print(f"Wrote {md_path}", flush=True)
    print(f"Wrote {json_path}", flush=True)
    print(f"Wrote {csv_path}", flush=True)


if __name__ == "__main__":
    main()
