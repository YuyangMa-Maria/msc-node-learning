"""Evaluate calibrated decision-level fusion of paired VSN and ASN scores.

Both nodes use aligned split membership from the paired dataset. Fusion consumes
only scores and confidence-like weights, matching the compact communication
contract rather than combining raw images, audio or hidden features.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from train_asn_audio import AsnModel
from train_vsn_binary import build_model, build_transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "dataset" / "multimodal-concrete-crack-detection-dataset"


@dataclass
class PairSample:
    image_path: Path
    audio_path: Path
    label: int
    split: str


def as_namespace(data: dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(**data)


def split_pairs(train: float, val: float, seed: int) -> list[PairSample]:
    rows: list[dict[str, object]] = []
    with (DATASET_ROOT / "dataset.csv").open(newline="", encoding="utf-8") as f:
        for item in csv.DictReader(f):
            label_name = item["label"].strip().lower()
            if label_name not in {"normal", "abnormal"}:
                continue
            rows.append(
                {
                    "image_path": DATASET_ROOT / item["image_path"],
                    "audio_path": DATASET_ROOT / item["audio_path"],
                    "label": 1 if label_name == "abnormal" else 0,
                }
            )

    by_label: dict[int, list[dict[str, object]]] = {0: [], 1: []}
    for row in rows:
        by_label[int(row["label"])].append(row)
    rng = random.Random(seed)
    samples: list[PairSample] = []
    for label_rows in by_label.values():
        rng.shuffle(label_rows)
        n = len(label_rows)
        n_train = int(n * train)
        n_val = int(n * val)
        for idx, row in enumerate(label_rows):
            if idx < n_train:
                split = "train"
            elif idx < n_train + n_val:
                split = "val"
            else:
                split = "test"
            samples.append(
                PairSample(
                    image_path=Path(row["image_path"]),
                    audio_path=Path(row["audio_path"]),
                    label=int(row["label"]),
                    split=split,
                )
            )
    return samples


def confidence_from_risk(probs: torch.Tensor) -> torch.Tensor:
    probs = probs.clamp(1e-7, 1 - 1e-7)
    entropy = -(probs * torch.log2(probs) + (1 - probs) * torch.log2(1 - probs))
    return 1.0 - entropy


def best_f1_threshold(scores: torch.Tensor, labels: torch.Tensor) -> float:
    y = labels.detach().cpu().int()
    candidates = torch.unique(scores.detach().cpu()).tolist()
    if not candidates:
        return 0.5
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in candidates:
        pred = (scores >= threshold).int()
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


def metrics(scores: torch.Tensor, labels: torch.Tensor, threshold: float) -> dict[str, float]:
    scores = scores.detach().cpu()
    labels = labels.detach().cpu().int()
    pred = (scores >= threshold).int()
    tp = int(((pred == 1) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    try:
        auc = float(roc_auc_score(labels.numpy(), scores.numpy()))
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": accuracy,
        "precision_positive": precision,
        "recall_positive": recall,
        "specificity": specificity,
        "f1_positive": f1,
        "roc_auc": auc,
        "threshold": threshold,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def apply_image_corruption(image: Image.Image, corruption: str, severity: int, seed: int) -> Image.Image:
    if corruption == "clean":
        return image
    if corruption == "low_light":
        factors = {1: 0.7, 2: 0.45, 3: 0.25}
        return ImageEnhance.Brightness(image).enhance(factors[severity])
    if corruption == "blur":
        radii = {1: 1.0, 2: 2.0, 3: 3.5}
        return image.filter(ImageFilter.GaussianBlur(radius=radii[severity]))
    if corruption == "jpeg":
        qualities = {1: 60, 2: 35, 3: 15}
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=qualities[severity])
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")
    raise ValueError(f"Unsupported image corruption: {corruption}")


def apply_audio_corruption(waveform: torch.Tensor, corruption: str, severity: int, seed: int) -> torch.Tensor:
    if corruption == "clean":
        return waveform
    if corruption == "gaussian_noise":
        stds = {1: 0.001, 2: 0.003, 3: 0.007}
        generator = torch.Generator().manual_seed(seed)
        return waveform + torch.randn(waveform.shape, generator=generator) * stds[severity]
    if corruption == "low_volume":
        factors = {1: 0.7, 2: 0.4, 3: 0.2}
        return waveform * factors[severity]
    raise ValueError(f"Unsupported audio corruption: {corruption}")


class PairedDataset(Dataset):
    def __init__(
        self,
        samples: list[PairSample],
        image_transform,
        target_audio_samples: int,
        image_corruption: str,
        image_severity: int,
        audio_corruption: str,
        audio_severity: int,
        seed: int,
    ) -> None:
        self.samples = samples
        self.image_transform = image_transform
        self.target_audio_samples = target_audio_samples
        self.image_corruption = image_corruption
        self.image_severity = image_severity
        self.audio_corruption = audio_corruption
        self.audio_severity = audio_severity
        self.seed = seed

    def __len__(self) -> int:
        return len(self.samples)

    def _fix_audio_length(self, waveform: torch.Tensor) -> torch.Tensor:
        waveform = waveform.mean(dim=0)
        n = waveform.numel()
        if n == self.target_audio_samples:
            return waveform
        if n > self.target_audio_samples:
            start = (n - self.target_audio_samples) // 2
            return waveform[start : start + self.target_audio_samples]
        return torch.nn.functional.pad(waveform, (0, self.target_audio_samples - n))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        image = apply_image_corruption(image, self.image_corruption, self.image_severity, self.seed + index)
        image_tensor = self.image_transform(image)

        waveform, sr = torchaudio.load(sample.audio_path)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        waveform = self._fix_audio_length(waveform)
        waveform = apply_audio_corruption(waveform, self.audio_corruption, self.audio_severity, self.seed + index)
        return image_tensor, waveform.float(), torch.tensor(sample.label, dtype=torch.float32)


def load_vsn(run_dir: Path, calibration_json: Path, device: torch.device):
    checkpoint = torch.load(run_dir / "best.pt", map_location=device)
    args = as_namespace(checkpoint["args"])
    model = build_model(args.model, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    temperature = float(json.loads(calibration_json.read_text(encoding="utf-8"))["temperature"])
    return model, temperature, int(getattr(args, "image_size", 224))


def load_asn(run_dir: Path, calibration_json: Path, device: torch.device):
    checkpoint = torch.load(run_dir / "best.pt", map_location=device)
    args = as_namespace(checkpoint["args"])
    model = AsnModel(args.sample_rate, args.n_fft, args.hop_length, args.n_mels).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    calibration = json.loads(calibration_json.read_text(encoding="utf-8"))
    return model, float(calibration["temperature"]), int(args.sample_rate * args.duration)


def collect_scores(
    samples: list[PairSample],
    vsn_model,
    vsn_temperature: float,
    asn_model,
    asn_temperature: float,
    image_size: int,
    target_audio_samples: int,
    condition: dict[str, object],
    batch_size: int,
    device: torch.device,
    seed: int,
) -> dict[str, torch.Tensor]:
    _, image_tf = build_transforms(image_size)
    dataset = PairedDataset(
        samples,
        image_tf,
        target_audio_samples,
        image_corruption=str(condition["image_corruption"]),
        image_severity=int(condition["image_severity"]),
        audio_corruption=str(condition["audio_corruption"]),
        audio_severity=int(condition["audio_severity"]),
        seed=seed,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    vsn_scores: list[torch.Tensor] = []
    asn_scores: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    with torch.no_grad():
        for images, waveforms, y in loader:
            images = images.to(device, non_blocking=True)
            waveforms = waveforms.to(device, non_blocking=True)
            v_logits = vsn_model(images).flatten().detach().cpu()
            a_logits = asn_model(waveforms).flatten().detach().cpu()
            vsn_scores.append(torch.sigmoid(v_logits / vsn_temperature))
            asn_scores.append(torch.sigmoid(a_logits / asn_temperature))
            labels.append(y.detach().cpu())
    return {
        "vsn": torch.cat(vsn_scores),
        "asn": torch.cat(asn_scores),
        "labels": torch.cat(labels).float(),
    }


def fused_scores(scores: dict[str, torch.Tensor], condition: dict[str, object]) -> dict[str, torch.Tensor]:
    v = scores["vsn"]
    a = scores["asn"]
    cv = confidence_from_risk(v)
    ca = confidence_from_risk(a)
    v_quality = float(condition["visual_quality"])
    a_quality = float(condition["audio_quality"])
    quality_aware = (cv * v_quality * v + ca * a_quality * a) / (cv * v_quality + ca * a_quality).clamp_min(1e-7)
    if v_quality < 0.5 <= a_quality:
        quality_gate = a
    elif a_quality < 0.5 <= v_quality:
        quality_gate = v
    else:
        quality_gate = quality_aware
    return {
        "vsn_only": v,
        "asn_only": a,
        "fusion_mean": (v + a) / 2.0,
        "fusion_max": torch.maximum(v, a),
        "fusion_conf_weighted": (cv * v + ca * a) / (cv + ca).clamp_min(1e-7),
        "fusion_quality_aware": quality_aware,
        "fusion_quality_gate": quality_gate,
    }


def threshold_for(method: str, condition: dict[str, object], thresholds: dict[str, float]) -> float:
    if method != "fusion_quality_gate":
        return thresholds[method]
    v_quality = float(condition["visual_quality"])
    a_quality = float(condition["audio_quality"])
    if v_quality < 0.5 <= a_quality:
        return thresholds["asn_only"]
    if a_quality < 0.5 <= v_quality:
        return thresholds["vsn_only"]
    return thresholds["fusion_quality_aware"]


def run_experiment(args: argparse.Namespace) -> tuple[list[dict[str, object]], dict[str, float]]:
    device = torch.device(args.device)
    vsn_model, vsn_temperature, image_size = load_vsn(args.vsn_run_dir, args.vsn_calibration_json, device)
    asn_model, asn_temperature, target_audio_samples = load_asn(args.asn_run_dir, args.asn_calibration_json, device)
    all_samples = split_pairs(args.train, args.val, args.seed)
    val_samples = [sample for sample in all_samples if sample.split == "val"]
    test_samples = [sample for sample in all_samples if sample.split == "test"]

    clean_condition = {
        "name": "clean",
        "image_corruption": "clean",
        "image_severity": 0,
        "audio_corruption": "clean",
        "audio_severity": 0,
        "visual_quality": 1.0,
        "audio_quality": 1.0,
    }
    conditions = [
        clean_condition,
        {
            "name": "vsn_low_light_3",
            "image_corruption": "low_light",
            "image_severity": 3,
            "audio_corruption": "clean",
            "audio_severity": 0,
            "visual_quality": 0.25,
            "audio_quality": 1.0,
        },
        {
            "name": "vsn_blur_3",
            "image_corruption": "blur",
            "image_severity": 3,
            "audio_corruption": "clean",
            "audio_severity": 0,
            "visual_quality": 0.20,
            "audio_quality": 1.0,
        },
        {
            "name": "asn_noise_3",
            "image_corruption": "clean",
            "image_severity": 0,
            "audio_corruption": "gaussian_noise",
            "audio_severity": 3,
            "visual_quality": 1.0,
            "audio_quality": 0.35,
        },
        {
            "name": "both_blur_noise_3",
            "image_corruption": "blur",
            "image_severity": 3,
            "audio_corruption": "gaussian_noise",
            "audio_severity": 3,
            "visual_quality": 0.20,
            "audio_quality": 0.35,
        },
    ]

    val_scores = collect_scores(
        val_samples,
        vsn_model,
        vsn_temperature,
        asn_model,
        asn_temperature,
        image_size,
        target_audio_samples,
        clean_condition,
        args.batch_size,
        device,
        args.seed,
    )
    val_methods = fused_scores(val_scores, clean_condition)
    thresholds = {method: best_f1_threshold(score, val_scores["labels"]) for method, score in val_methods.items()}

    rows: list[dict[str, object]] = []
    for condition in conditions:
        condition_scores = collect_scores(
            test_samples,
            vsn_model,
            vsn_temperature,
            asn_model,
            asn_temperature,
            image_size,
            target_audio_samples,
            condition,
            args.batch_size,
            device,
            args.seed,
        )
        method_scores = fused_scores(condition_scores, condition)
        for method, score in method_scores.items():
            result = metrics(score, condition_scores["labels"], threshold_for(method, condition, thresholds))
            rows.append(
                {
                    "condition": condition["name"],
                    "method": method,
                    "sample_count": len(test_samples),
                    "visual_quality": condition["visual_quality"],
                    "audio_quality": condition["audio_quality"],
                    **result,
                }
            )
            print(
                f"{condition['name']} {method}: accuracy={result['accuracy']:.4f} "
                f"f1={result['f1_positive']:.4f} recall={result['recall_positive']:.4f}",
                flush=True,
            )
    return rows, thresholds


def write_outputs(rows: list[dict[str, object]], thresholds: dict[str, float], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "fusion_metrics.csv"
    json_path = output_dir / "fusion_metrics.json"
    md_path = output_dir / "REPORT.md"
    plot_path = output_dir / "fusion_f1_recall.png"

    fieldnames = [
        "condition",
        "method",
        "sample_count",
        "visual_quality",
        "audio_quality",
        "accuracy",
        "precision_positive",
        "recall_positive",
        "specificity",
        "f1_positive",
        "roc_auc",
        "threshold",
        "tn",
        "fp",
        "fn",
        "tp",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})
    json_path.write_text(json.dumps({"thresholds": thresholds, "results": rows}, indent=2), encoding="utf-8")

    conditions = list(dict.fromkeys(str(row["condition"]) for row in rows))
    methods = list(dict.fromkeys(str(row["method"]) for row in rows))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=140)
    x = np.arange(len(conditions))
    for metric_name, ax in [("f1_positive", axes[0]), ("recall_positive", axes[1])]:
        width = 0.8 / len(methods)
        for idx, method in enumerate(methods):
            values = [next(row[metric_name] for row in rows if row["condition"] == condition and row["method"] == method) for condition in conditions]
            ax.bar(x + idx * width, values, width=width, label=method)
        ax.set_xticks(x + width * (len(methods) - 1) / 2, conditions, rotation=25, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(metric_name.replace("_positive", "").upper())
        ax.grid(True, axis="y", alpha=0.25)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)

    lines = [
        "# VSN + ASN Fusion Experiment Report",
        "",
        "## Setup",
        "",
        "- Dataset: paired image/audio samples from `multimodal-concrete-crack-detection-dataset`",
        "- Split: same 70/15/15 stratified split logic used by VSN/ASN",
        "- VSN: calibrated MobileNetV3-Small visual risk score",
        "- ASN: calibrated tiny log-mel CNN acoustic risk score",
        "- Fusion threshold: selected on clean validation split for each method, then fixed for all test conditions",
        "",
        "## Fusion Methods",
        "",
        "- `vsn_only`: visual risk score only",
        "- `asn_only`: acoustic risk score only",
        "- `fusion_mean`: arithmetic mean of VSN and ASN risk scores",
        "- `fusion_max`: conservative risk fusion that uses the maximum node risk score",
        "- `fusion_conf_weighted`: weighted by entropy-based confidence",
        "- `fusion_quality_aware`: confidence-weighted fusion with simulated input-quality downweighting under degraded conditions",
        "- `fusion_quality_gate`: routes to the reliable modality when one node reports poor input quality",
        "",
        "## Validation Thresholds",
        "",
        "| Method | Threshold |",
        "| --- | ---: |",
    ]
    for method, threshold in thresholds.items():
        lines.append(f"| {method} | {threshold:.4f} |")
    lines.extend(
        [
            "",
            "## Test Results",
            "",
            "| Condition | Method | Accuracy | Precision | Recall | F1 | ROC-AUC | Confusion matrix (TN/FP/FN/TP) |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in rows:
        lines.append(
            "| {condition} | {method} | {acc:.4f} | {prec:.4f} | {rec:.4f} | {f1:.4f} | {auc:.4f} | {tn}/{fp}/{fn}/{tp} |".format(
                condition=row["condition"],
                method=row["method"],
                acc=row["accuracy"],
                prec=row["precision_positive"],
                rec=row["recall_positive"],
                f1=row["f1_positive"],
                auc=row["roc_auc"],
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
            "Fusion is most useful when one modality is degraded and the other remains reliable. The quality-aware fusion condition represents the Node Learning idea that each node should report not only `risk_score`, but also usable `confidence/status` information so unreliable modalities can be downweighted.",
            "",
            "## Files",
            "",
            "- Metrics CSV: `fusion_metrics.csv`",
            "- Metrics JSON: `fusion_metrics.json`",
            "- Plot: `fusion_f1_recall.png`",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {plot_path}")
    print(f"Wrote {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VSN+ASN paired risk-score fusion.")
    parser.add_argument("--vsn-run-dir", type=Path, default=PROJECT_ROOT / "outputs" / "vsn_binary_mixed_full" / "mobilenet_v3_small_pretrained")
    parser.add_argument("--vsn-calibration-json", type=Path, default=PROJECT_ROOT / "outputs" / "vsn_calibration" / "vsn_binary_mixed_full_mobilenet_v3_small_pretrained_calibration.json")
    parser.add_argument("--asn-run-dir", type=Path, default=PROJECT_ROOT / "outputs" / "asn_audio_formal" / "tiny_logmel_cnn")
    parser.add_argument("--asn-calibration-json", type=Path, default=PROJECT_ROOT / "outputs" / "asn_calibration" / "asn_audio_formal_tiny_logmel_cnn_calibration.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "fusion_vsn_asn")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    rows, thresholds = run_experiment(args)
    write_outputs(rows, thresholds, args.output_dir)


if __name__ == "__main__":
    main()
