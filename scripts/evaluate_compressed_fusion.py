"""Evaluate calibrated INT8 node outputs under clean and degraded fusion.

All fusion thresholds are selected on the grouped validation split and then
held fixed on test conditions. Optional VBN scores are label-aligned proxy
evidence; they are not presented as synchronous physical measurements.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import pickle
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
from torch import nn
from torch.ao.quantization import convert, get_default_qconfig, prepare
from torch.utils.data import DataLoader, Dataset

from evaluate_asn_robustness import add_noise_at_snr
from export_asn_student_ptq import QuantizedEncoderWrapper, fuse_student_encoder
from export_vsn_student_ptq import QuantizedStudentWrapper, fuse_student
from train_asn_student import AsnStudentModel
from train_vbn_orion import DEFAULT_INDEX, build_feature_table, read_rows
from train_vsn_binary import PROJECT_ROOT, build_transforms
from train_vsn_student_baseline import VsnStudentDwCnn

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
            split = "train" if idx < n_train else "val" if idx < n_train + n_val else "test"
            samples.append(PairSample(Path(row["image_path"]), Path(row["audio_path"]), int(row["label"]), split))
    return samples


def read_grouped_pairs(index: Path) -> list[PairSample]:
    samples: list[PairSample] = []
    with index.open(newline="", encoding="utf-8") as handle:
        for item in csv.DictReader(handle):
            image_path = Path(item["paired_image_path"])
            audio_path = Path(item["path"])
            if not image_path.is_absolute():
                image_path = PROJECT_ROOT / image_path
            if not audio_path.is_absolute():
                audio_path = PROJECT_ROOT / audio_path
            split = item["split"].strip().lower()
            if split not in {"train", "val", "test"}:
                raise ValueError(f"Unsupported split '{split}' in {index}")
            samples.append(PairSample(image_path, audio_path, int(item["label"]), split))
    if not samples:
        raise RuntimeError(f"No paired samples found in {index}")
    return samples


def load_temperature(path: Path | None) -> float:
    if path is None:
        return 1.0
    data = json.loads(path.read_text(encoding="utf-8"))
    return float(data.get("temperature", 1.0))


def apply_image_corruption(image: Image.Image, corruption: str, severity: int, seed: int) -> Image.Image:
    if corruption == "clean":
        return image
    if corruption == "low_light":
        return ImageEnhance.Brightness(image).enhance({1: 0.7, 2: 0.45, 3: 0.25}[severity])
    if corruption == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius={1: 1.0, 2: 2.0, 3: 3.5}[severity]))
    if corruption == "jpeg":
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality={1: 60, 2: 35, 3: 15}[severity])
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")
    raise ValueError(f"Unsupported image corruption: {corruption}")


def apply_audio_corruption(waveform: torch.Tensor, corruption: str, severity: int, seed: int) -> torch.Tensor:
    if corruption == "clean":
        return waveform
    if corruption == "snr_noise":
        return add_noise_at_snr(waveform, {1: 20.0, 2: 10.0, 3: 0.0}[severity], seed)
    if corruption == "low_volume":
        return waveform * {1: 0.7, 2: 0.4, 3: 0.2}[severity]
    raise ValueError(f"Unsupported audio corruption: {corruption}")


class CompressedPairDataset(Dataset):
    def __init__(
        self,
        samples: list[PairSample],
        image_transform,
        audio_samples: int,
        image_corruption: str,
        image_severity: int,
        audio_corruption: str,
        audio_severity: int,
        seed: int,
    ) -> None:
        self.samples = samples
        self.image_transform = image_transform
        self.audio_samples = audio_samples
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
        if n == self.audio_samples:
            return waveform
        if n > self.audio_samples:
            start = (n - self.audio_samples) // 2
            return waveform[start : start + self.audio_samples]
        return torch.nn.functional.pad(waveform, (0, self.audio_samples - n))

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


def build_vsn_student(checkpoint: dict[str, object]) -> nn.Module:
    args = as_namespace(checkpoint["args"])  # type: ignore[arg-type]
    model = VsnStudentDwCnn(width=float(getattr(args, "width", 1.0)), dropout=float(getattr(args, "dropout", 0.1)))
    model.load_state_dict(checkpoint["model"])  # type: ignore[arg-type]
    model.eval()
    return model


def build_asn_student(checkpoint: dict[str, object]) -> AsnStudentModel:
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


def calibrate_vsn(model: nn.Module, loader: DataLoader, max_batches: int | None) -> None:
    model.eval()
    with torch.no_grad():
        for idx, (images, _, _) in enumerate(loader):
            if max_batches is not None and idx >= max_batches:
                break
            _ = model(images)


def calibrate_asn(model: nn.Module, loader: DataLoader, max_batches: int | None) -> None:
    model.eval()
    with torch.no_grad():
        for idx, (_, waveforms, _) in enumerate(loader):
            if max_batches is not None and idx >= max_batches:
                break
            _ = model(waveforms)


def load_vsn_int8(run_dir: Path, calib_loader: DataLoader, backend: str, max_batches: int | None) -> nn.Module:
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu")
    fused_student = fuse_student(build_vsn_student(checkpoint))  # type: ignore[arg-type]
    model = QuantizedStudentWrapper(fused_student)
    model.eval()
    model.qconfig = get_default_qconfig(backend)
    prepared = prepare(model, inplace=False)
    calibrate_vsn(prepared, calib_loader, max_batches)
    return convert(prepared, inplace=False)


def load_asn_int8(run_dir: Path, calib_loader: DataLoader, backend: str, max_batches: int | None) -> tuple[nn.Module, int]:
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu")
    args = as_namespace(checkpoint["args"])  # type: ignore[arg-type]
    student = build_asn_student(checkpoint)
    fuse_student_encoder(student.encoder)  # type: ignore[arg-type]
    model = QuantizedEncoderWrapper(student)
    model.eval()
    model.qconfig = get_default_qconfig(backend)
    prepared = prepare(model, inplace=False)
    calibrate_asn(prepared, calib_loader, max_batches)
    converted = convert(prepared, inplace=False)
    return converted, int(float(getattr(args, "duration")) * int(getattr(args, "sample_rate")))


def confidence_from_risk(probs: torch.Tensor) -> torch.Tensor:
    probs = probs.clamp(1e-7, 1 - 1e-7)
    entropy = -(probs * torch.log2(probs) + (1 - probs) * torch.log2(1 - probs))
    return 1.0 - entropy


def best_f1_threshold(scores: torch.Tensor, labels: torch.Tensor) -> float:
    scores_cpu = scores.detach().cpu()
    y = labels.detach().cpu().int()
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in torch.unique(scores_cpu).tolist():
        pred = (scores_cpu >= threshold).int()
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


def metrics(scores: torch.Tensor, labels: torch.Tensor, threshold: float) -> dict[str, float | int]:
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


def collect_scores(
    samples: list[PairSample],
    vsn_model: nn.Module,
    vsn_temperature: float,
    asn_model: nn.Module,
    asn_temperature: float,
    image_size: int,
    audio_samples: int,
    condition: dict[str, object],
    batch_size: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    _, image_tf = build_transforms(image_size)
    dataset = CompressedPairDataset(
        samples,
        image_tf,
        audio_samples,
        str(condition["image_corruption"]),
        int(condition["image_severity"]),
        str(condition["audio_corruption"]),
        int(condition["audio_severity"]),
        seed,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    vsn_scores: list[torch.Tensor] = []
    asn_scores: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    with torch.no_grad():
        for images, waveforms, y in loader:
            v_logits = vsn_model(images).flatten().detach().cpu()
            a_logits = asn_model(waveforms).flatten().detach().cpu()
            vsn_scores.append(torch.sigmoid(v_logits / vsn_temperature))
            asn_scores.append(torch.sigmoid(a_logits / asn_temperature))
            labels.append(y.detach().cpu())
    return {"vsn": torch.cat(vsn_scores), "asn": torch.cat(asn_scores), "labels": torch.cat(labels).float()}


def load_vbn_split_scores(vbn_run_dir: Path, index: Path, max_points: int, sample_rate: float) -> dict[str, dict[int, list[float]]]:
    with (vbn_run_dir / "best_model.pkl").open("rb") as f:
        saved = pickle.load(f)
    model = saved["model"]
    rows = read_rows(index, binary_only=True)
    X, y, _, cached_rows = build_feature_table(rows, vbn_run_dir / f"features_max{max_points}.json", max_points, sample_rate)
    probs = model.predict_proba(X)[:, 1]
    split_scores: dict[str, dict[int, list[float]]] = {"train": {0: [], 1: []}, "val": {0: [], 1: []}, "test": {0: [], 1: []}}
    for row, label, prob in zip(cached_rows, y, probs):
        split_scores[row["split"]][int(label)].append(float(prob))
    return split_scores


def align_vbn_scores(labels: torch.Tensor, pool: dict[int, list[float]], seed: int) -> torch.Tensor:
    rng = random.Random(seed)
    counters = {0: 0, 1: 0}
    shuffled = {label: list(values) for label, values in pool.items()}
    for values in shuffled.values():
        rng.shuffle(values)
    scores = []
    for raw_label in labels.detach().cpu().int().tolist():
        values = shuffled[int(raw_label)]
        idx = counters[int(raw_label)] % len(values)
        scores.append(values[idx])
        counters[int(raw_label)] += 1
    return torch.tensor(scores, dtype=torch.float32)


def degrade_vbn(scores: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "clean":
        return scores
    if mode == "uncertain":
        return 0.5 + (scores - 0.5) * 0.30
    if mode == "offline":
        return torch.full_like(scores, 0.5)
    raise ValueError(f"Unsupported VBN mode: {mode}")


def fused_scores(scores: dict[str, torch.Tensor], condition: dict[str, object], include_vbn: bool) -> dict[str, torch.Tensor]:
    """Apply the ablation fusion rules to one common set of node scores."""
    v = scores["vsn"]
    a = scores["asn"]
    cv = confidence_from_risk(v)
    ca = confidence_from_risk(a)
    qv = float(condition["visual_quality"])
    qa = float(condition["audio_quality"])
    if not include_vbn:
        quality_aware = (cv * qv * v + ca * qa * a) / (cv * qv + ca * qa).clamp_min(1e-7)
        quality_gate = a if qv < 0.5 <= qa else v if qa < 0.5 <= qv else quality_aware
        return {
            "vsn_only": v,
            "asn_only": a,
            "fusion_mean": (v + a) / 2.0,
            "fusion_max": torch.maximum(v, a),
            "fusion_conf_weighted": (cv * v + ca * a) / (cv + ca).clamp_min(1e-7),
            "fusion_quality_aware": quality_aware,
            "fusion_quality_gate": quality_gate,
        }

    b = scores["vbn"]
    cb = confidence_from_risk(b)
    qb = float(condition["vbn_quality"])
    # Quality is an externally defined sensor-state term. It is intentionally
    # separate from confidence, which is derived from the model output.
    qualities = torch.tensor([qv, qa, qb], dtype=torch.float32)
    stacked = torch.stack([v, a, b], dim=0)
    confs = torch.stack([cv, ca, cb], dim=0)
    quality_weights = confs * qualities[:, None]
    quality_aware = (quality_weights * stacked).sum(dim=0) / quality_weights.sum(dim=0).clamp_min(1e-7)
    active = qualities >= 0.5
    gated_scores = stacked[active] if int(active.sum()) else stacked
    gated_confs = confs[active] if int(active.sum()) else confs
    quality_gate = (gated_scores * gated_confs).sum(dim=0) / gated_confs.sum(dim=0).clamp_min(1e-7)
    quality_filtered_max = torch.max(gated_scores, dim=0).values
    return {
        "vsn_only": v,
        "asn_only": a,
        "vbn_only": b,
        "fusion_mean_3": stacked.mean(dim=0),
        "fusion_max_3": torch.max(stacked, dim=0).values,
        "fusion_quality_filtered_max_3": quality_filtered_max,
        "fusion_conf_weighted_3": (confs * stacked).sum(dim=0) / confs.sum(dim=0).clamp_min(1e-7),
        "fusion_quality_aware_3": quality_aware,
        "fusion_quality_gate_3": quality_gate,
    }


def threshold_for(method: str, condition: dict[str, object], thresholds: dict[str, float]) -> float:
    if method == "fusion_quality_gate":
        if float(condition["visual_quality"]) < 0.5 <= float(condition["audio_quality"]):
            return thresholds["asn_only"]
        if float(condition["audio_quality"]) < 0.5 <= float(condition["visual_quality"]):
            return thresholds["vsn_only"]
        return thresholds["fusion_quality_aware"]
    if method in {"fusion_quality_gate_3", "fusion_quality_filtered_max_3"} and "fusion_quality_aware_3" in thresholds:
        active = [
            ("vsn_only", float(condition["visual_quality"]) >= 0.5),
            ("asn_only", float(condition["audio_quality"]) >= 0.5),
            ("vbn_only", float(condition["vbn_quality"]) >= 0.5),
        ]
        active_methods = [name for name, is_active in active if is_active]
        if len(active_methods) == 1:
            return thresholds[active_methods[0]]
        if method == "fusion_quality_filtered_max_3":
            return thresholds["fusion_quality_filtered_max_3"]
        return thresholds["fusion_quality_aware_3"]
    return thresholds[method]


def evaluate_conditions(
    val_samples: list[PairSample],
    test_samples: list[PairSample],
    vsn_model: nn.Module,
    vsn_temperature: float,
    asn_model: nn.Module,
    asn_temperature: float,
    image_size: int,
    audio_samples: int,
    batch_size: int,
    seed: int,
    include_vbn: bool,
    vbn_pool: dict[str, dict[int, list[float]]] | None,
) -> tuple[list[dict[str, object]], dict[str, float]]:
    """Tune on clean validation data and evaluate every fixed test condition."""
    clean_condition = {
        "name": "clean",
        "image_corruption": "clean",
        "image_severity": 0,
        "audio_corruption": "clean",
        "audio_severity": 0,
        "vbn_mode": "clean",
        "visual_quality": 1.0,
        "audio_quality": 1.0,
        "vbn_quality": 1.0,
    }
    conditions = [
        clean_condition,
        {
            "name": "vsn_blur_asn_clean_vbn_clean" if include_vbn else "vsn_blur_3",
            "image_corruption": "blur",
            "image_severity": 3,
            "audio_corruption": "clean",
            "audio_severity": 0,
            "vbn_mode": "clean",
            "visual_quality": 0.20,
            "audio_quality": 1.0,
            "vbn_quality": 1.0,
        },
        {
            "name": "asn_noise_vsn_clean_vbn_clean" if include_vbn else "asn_noise_3",
            "image_corruption": "clean",
            "image_severity": 0,
            "audio_corruption": "snr_noise",
            "audio_severity": 3,
            "vbn_mode": "clean",
            "visual_quality": 1.0,
            "audio_quality": 0.35,
            "vbn_quality": 1.0,
        },
        {
            "name": "vsn_asn_degraded_vbn_clean" if include_vbn else "both_blur_noise_3",
            "image_corruption": "blur",
            "image_severity": 3,
            "audio_corruption": "snr_noise",
            "audio_severity": 3,
            "vbn_mode": "clean",
            "visual_quality": 0.20,
            "audio_quality": 0.35,
            "vbn_quality": 1.0,
        },
    ]
    if include_vbn:
        conditions.extend(
            [
                {
                    "name": "vbn_uncertain_vsn_asn_clean",
                    "image_corruption": "clean",
                    "image_severity": 0,
                    "audio_corruption": "clean",
                    "audio_severity": 0,
                    "vbn_mode": "uncertain",
                    "visual_quality": 1.0,
                    "audio_quality": 1.0,
                    "vbn_quality": 0.35,
                },
                {
                    "name": "all_three_degraded",
                    "image_corruption": "blur",
                    "image_severity": 3,
                    "audio_corruption": "snr_noise",
                    "audio_severity": 3,
                    "vbn_mode": "uncertain",
                    "visual_quality": 0.20,
                    "audio_quality": 0.35,
                    "vbn_quality": 0.35,
                },
            ]
        )

    val_scores = collect_scores(
        val_samples, vsn_model, vsn_temperature, asn_model, asn_temperature, image_size, audio_samples, clean_condition, batch_size, seed
    )
    if include_vbn and vbn_pool is not None:
        val_scores["vbn"] = align_vbn_scores(val_scores["labels"], vbn_pool["val"], seed)
    val_methods = fused_scores(val_scores, clean_condition, include_vbn)
    thresholds = {method: best_f1_threshold(score, val_scores["labels"]) for method, score in val_methods.items()}

    rows: list[dict[str, object]] = []
    for condition in conditions:
        scores = collect_scores(
            test_samples, vsn_model, vsn_temperature, asn_model, asn_temperature, image_size, audio_samples, condition, batch_size, seed
        )
        if include_vbn and vbn_pool is not None:
            scores["vbn"] = degrade_vbn(align_vbn_scores(scores["labels"], vbn_pool["test"], seed), str(condition["vbn_mode"]))
        methods = fused_scores(scores, condition, include_vbn)
        for method, score in methods.items():
            result = metrics(score, scores["labels"], threshold_for(method, condition, thresholds))
            rows.append(
                {
                    "condition": condition["name"],
                    "method": method,
                    "sample_count": len(test_samples),
                    "visual_quality": condition["visual_quality"],
                    "audio_quality": condition["audio_quality"],
                    "vbn_quality": condition["vbn_quality"] if include_vbn else "",
                    **result,
                }
            )
            print(f"{condition['name']} {method}: f1={result['f1_positive']:.4f} recall={result['recall_positive']:.4f}", flush=True)
    return rows, thresholds


def write_report(
    rows: list[dict[str, object]],
    thresholds: dict[str, float],
    output_dir: Path,
    title: str,
    include_vbn: bool,
    metadata: dict[str, object],
) -> None:
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
        "vbn_quality",
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
    json_path.write_text(json.dumps({"metadata": metadata, "thresholds": thresholds, "results": rows}, indent=2), encoding="utf-8")

    conditions = list(dict.fromkeys(str(row["condition"]) for row in rows))
    methods = list(dict.fromkeys(str(row["method"]) for row in rows))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), dpi=140)
    x = np.arange(len(conditions))
    for metric_name, ax in [("f1_positive", axes[0]), ("recall_positive", axes[1])]:
        width = 0.8 / len(methods)
        for idx, method in enumerate(methods):
            values = [next(float(row[metric_name]) for row in rows if row["condition"] == condition and row["method"] == method) for condition in conditions]
            ax.bar(x + idx * width, values, width=width, label=method)
        ax.set_xticks(x + width * (len(methods) - 1) / 2, conditions, rotation=25, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(metric_name.replace("_positive", "").upper())
        ax.grid(True, axis="y", alpha=0.25)
    axes[0].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)

    lines = [
        f"# {title}",
        "",
        "## Setup",
        "",
        f"- VSN: `{metadata['vsn_run_dir']}`",
        f"- ASN: `{metadata['asn_run_dir']}`",
        "- VSN/ASN are rebuilt as PyTorch static PTQ INT8 models for this evaluation.",
        "- Thresholds are selected on the clean validation split and fixed for test conditions.",
    ]
    if metadata.get("pair_index"):
        lines.append(f"- Paired samples follow the leakage-resistant grouped split in `{metadata['pair_index']}`.")
    if include_vbn:
        lines.append("- VBN is label-aligned proxy evidence from ORION AE, not synchronized with VSN/ASN.")
    asn_temperature = float(metadata.get("asn_temperature", 1.0))
    if abs(asn_temperature - 1.0) < 1e-9:
        lines.append("- ASN student risk scores use raw sigmoid in this experiment; ASN student temperature calibration is a recommended next check.")
    else:
        lines.append(f"- ASN student risk scores use temperature-scaled sigmoid with T={asn_temperature:.6f}.")
    lines.extend(
        [
            "",
            "## Validation Thresholds",
            "",
            "| Method | Threshold |",
            "| --- | ---: |",
        ]
    )
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
            "This experiment checks whether the system-level fusion logic still works after replacing VSN and ASN with compressed INT8 student nodes. Results should be described as compressed-node fusion validation, not final hardware deployment.",
            "",
            "## Files",
            "",
            "- Metrics CSV: `fusion_metrics.csv`",
            "- Metrics JSON: `fusion_metrics.json`",
            "- Plot: `fusion_f1_recall.png`",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(csv_path)
    print(json_path)
    print(plot_path)
    print(md_path)


def main() -> None:
    """Load calibrated PTQ nodes and run two- and three-node stress tests."""
    parser = argparse.ArgumentParser(description="Evaluate compressed INT8 VSN/ASN fusion and optional VBN proxy fusion.")
    parser.add_argument("--vsn-run-dir", type=Path, default=PROJECT_ROOT / "outputs" / "vsn_student_kd" / "vsn_student_dwcnn_s128_w1_kd_t4_a0.5")
    parser.add_argument(
        "--vsn-calibration-json",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "vsn_student_calibration" / "vsn_student_kd_vsn_student_dwcnn_s128_w1_kd_t4_a0.5_calibration.json",
    )
    parser.add_argument("--asn-run-dir", type=Path, default=PROJECT_ROOT / "outputs" / "asn_student_baselines" / "asn_student_w0p5_d5p0s_scratch")
    parser.add_argument("--asn-calibration-json", type=Path, default=None)
    parser.add_argument("--asn-temperature", type=float, default=1.0)
    parser.add_argument("--pair-index", type=Path, default=None)
    parser.add_argument("--vbn-run-dir", type=Path, default=PROJECT_ROOT / "outputs" / "vbn_orion")
    parser.add_argument("--vbn-index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--vbn-max-points", type=int, default=200_000)
    parser.add_argument("--vbn-sample-rate", type=float, default=5_000_000.0)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "compressed_fusion")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backend", default="fbgemm", choices=["fbgemm", "qnnpack"])
    parser.add_argument("--max-calib-batches", type=int, default=None)
    args = parser.parse_args()

    torch.backends.quantized.engine = args.backend
    all_samples = read_grouped_pairs(args.pair_index) if args.pair_index is not None else split_pairs(args.train, args.val, args.seed)
    val_samples = [sample for sample in all_samples if sample.split == "val"]
    test_samples = [sample for sample in all_samples if sample.split == "test"]
    vsn_checkpoint = torch.load(args.vsn_run_dir / "best.pt", map_location="cpu")
    image_size = int(vsn_checkpoint.get("image_size", 128))
    asn_checkpoint = torch.load(args.asn_run_dir / "best.pt", map_location="cpu")
    asn_args = as_namespace(asn_checkpoint["args"])  # type: ignore[arg-type]
    audio_samples = int(float(getattr(asn_args, "duration")) * int(getattr(asn_args, "sample_rate")))
    _, image_tf = build_transforms(image_size)
    clean_calib_loader = DataLoader(
        CompressedPairDataset(val_samples, image_tf, audio_samples, "clean", 0, "clean", 0, args.seed),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    vsn_model = load_vsn_int8(args.vsn_run_dir, clean_calib_loader, args.backend, args.max_calib_batches)
    asn_model, audio_samples = load_asn_int8(args.asn_run_dir, clean_calib_loader, args.backend, args.max_calib_batches)
    vsn_temperature = load_temperature(args.vsn_calibration_json)
    asn_temperature = load_temperature(args.asn_calibration_json) if args.asn_calibration_json is not None else args.asn_temperature

    metadata = {
        "vsn_run_dir": str(args.vsn_run_dir),
        "asn_run_dir": str(args.asn_run_dir),
        "vsn_temperature": vsn_temperature,
        "asn_temperature": asn_temperature,
        "pair_index": None if args.pair_index is None else str(args.pair_index),
        "backend": args.backend,
        "image_size": image_size,
        "audio_samples": audio_samples,
    }
    two_rows, two_thresholds = evaluate_conditions(
        val_samples,
        test_samples,
        vsn_model,
        vsn_temperature,
        asn_model,
        asn_temperature,
        image_size,
        audio_samples,
        args.batch_size,
        args.seed,
        include_vbn=False,
        vbn_pool=None,
    )
    write_report(two_rows, two_thresholds, args.output_dir / "vsn_asn_int8", "Compressed INT8 VSN + ASN Fusion Report", False, metadata)

    vbn_pool = load_vbn_split_scores(args.vbn_run_dir, args.vbn_index, args.vbn_max_points, args.vbn_sample_rate)
    three_rows, three_thresholds = evaluate_conditions(
        val_samples,
        test_samples,
        vsn_model,
        vsn_temperature,
        asn_model,
        asn_temperature,
        image_size,
        audio_samples,
        args.batch_size,
        args.seed,
        include_vbn=True,
        vbn_pool=vbn_pool,
    )
    write_report(
        three_rows,
        three_thresholds,
        args.output_dir / "vsn_asn_vbn_int8",
        "Compressed INT8 VSN + ASN + VBN Proxy Fusion Report",
        True,
        metadata,
    )


if __name__ == "__main__":
    main()
