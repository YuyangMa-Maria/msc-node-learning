"""Test shared-layer candidates under modality-specific degradation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from PIL import Image, ImageEnhance, ImageFilter
from torch import nn
from torch.utils.data import DataLoader, Dataset

from evaluate_asn_robustness import corrupt_audio
from evaluate_parameter_sharing_cnn_embeddings import (
    DEFAULT_ASN_CHECKPOINT,
    DEFAULT_ASN_INDEX,
    DEFAULT_VBN_FEATURES,
    DEFAULT_VSN_CHECKPOINT,
    DEFAULT_VSN_INDEX,
    PROJECT_ROOT,
    SeparateRiskHeadsModel,
    SharedRiskHeadModel,
    SharedTrunkNodeCalibratedModel,
    extract_asn_embeddings,
    extract_vsn_embeddings,
    json_safe,
    load_asn_model,
    load_vbn_feature_embeddings,
    load_vsn_model,
    make_modality_data,
    metric_row,
    predict_probs,
    save_json,
    write_csv,
)
from train_asn_audio import AudioSample, read_samples as read_asn_samples
from train_asn_student import limit_samples as limit_asn_samples
from train_vsn_binary import Sample, build_transforms, limit_samples as limit_vsn_samples, read_samples as read_vsn_samples


def apply_visual_degradation(image: Image.Image, condition: str, severity: int, seed: int) -> Image.Image:
    if condition == "clean":
        return image
    if condition == "low_light":
        return ImageEnhance.Brightness(image).enhance({1: 0.7, 2: 0.45, 3: 0.25}[severity])
    if condition == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius={1: 1.0, 2: 2.0, 3: 3.5}[severity]))
    if condition == "gaussian_noise":
        rng = np.random.default_rng(seed)
        arr = np.asarray(image).astype(np.float32) / 255.0
        arr = np.clip(arr + rng.normal(0.0, {1: 0.03, 2: 0.07, 3: 0.12}[severity], size=arr.shape), 0.0, 1.0)
        return Image.fromarray((arr * 255).astype(np.uint8))
    raise ValueError(f"Unsupported visual condition: {condition}")


class DegradedVsnEmbeddingDataset(Dataset):
    def __init__(self, samples: list[Sample], transform, condition: str, severity: int, seed: int) -> None:
        self.samples = samples
        self.transform = transform
        self.condition = condition
        self.severity = severity
        self.seed = seed

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        image = Image.open(sample.path).convert("RGB")
        image = apply_visual_degradation(image, self.condition, self.severity, self.seed + index)
        return self.transform(image), torch.tensor(sample.label, dtype=torch.float32)


def fix_audio_length(waveform: torch.Tensor, target_samples: int) -> torch.Tensor:
    waveform = waveform.mean(dim=0)
    n = waveform.numel()
    if n == target_samples:
        return waveform
    if n > target_samples:
        start = (n - target_samples) // 2
        return waveform[start : start + target_samples]
    return torch.nn.functional.pad(waveform, (0, target_samples - n))


class DegradedAsnEmbeddingDataset(Dataset):
    def __init__(
        self,
        samples: list[AudioSample],
        sample_rate: int,
        target_samples: int,
        condition: str,
        severity: int,
        seed: int,
    ) -> None:
        self.samples = samples
        self.sample_rate = sample_rate
        self.target_samples = target_samples
        self.condition = condition
        self.severity = severity
        self.seed = seed

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        waveform, sr = torchaudio.load(sample.path)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        waveform = fix_audio_length(waveform, self.target_samples)
        waveform = corrupt_audio(waveform, self.condition, self.severity, self.seed + index)
        return waveform.float(), torch.tensor(sample.label, dtype=torch.float32)


def extract_degraded_vsn_test_embeddings(
    index_path: Path,
    checkpoint_path: Path,
    cache_path: Path,
    eval_limit: int,
    condition: str,
    severity: int,
    batch_size: int,
    seed: int,
    device: torch.device,
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray]:
    if condition == "clean":
        raise ValueError("Use the clean embedding cache for clean VSN.")
    if cache_path.exists():
        cached = np.load(cache_path)
        return cached["test_x"].astype(np.float32), cached["test_y"].astype(np.int64)
    model, image_size, _ckpt_args = load_vsn_model(checkpoint_path, device)
    _, eval_tf = build_transforms(image_size)
    samples = limit_vsn_samples(read_vsn_samples(index_path, "test", None), eval_limit, seed + 2)
    loader = DataLoader(
        DegradedVsnEmbeddingDataset(samples, eval_tf, condition, severity, seed),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    embeddings: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.no_grad():
        for images, y in loader:
            images = images.to(device, non_blocking=True)
            feats = model.features(images)
            emb = F.adaptive_avg_pool2d(feats, 1).flatten(1)
            embeddings.append(emb.detach().cpu().numpy().astype(np.float32))
            labels.append(y.numpy().astype(np.int64))
    x = np.concatenate(embeddings, axis=0)
    y = np.concatenate(labels, axis=0)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, test_x=x, test_y=y)
    print(f"[VSN degraded] {condition}_s{severity}: {x.shape}")
    return x, y


def extract_degraded_asn_test_embeddings(
    index_path: Path,
    checkpoint_path: Path,
    cache_path: Path,
    eval_limit: int,
    condition: str,
    severity: int,
    batch_size: int,
    seed: int,
    device: torch.device,
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray]:
    if condition == "clean":
        raise ValueError("Use the clean embedding cache for clean ASN.")
    if cache_path.exists():
        cached = np.load(cache_path)
        return cached["test_x"].astype(np.float32), cached["test_y"].astype(np.int64)
    model, ckpt_args = load_asn_model(checkpoint_path, device)
    sample_rate = int(ckpt_args.get("sample_rate", 16_000))
    duration = float(ckpt_args.get("duration", 5.0))
    target_samples = int(sample_rate * duration)
    samples = limit_asn_samples(read_asn_samples(index_path, "test", None), eval_limit, seed + 5)
    loader = DataLoader(
        DegradedAsnEmbeddingDataset(samples, sample_rate, target_samples, condition, severity, seed),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    embeddings: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.no_grad():
        for waveform, y in loader:
            waveform = waveform.to(device, non_blocking=True)
            spec = model.frontend(waveform)
            emb = model.encoder.net(spec).flatten(1)
            embeddings.append(emb.detach().cpu().numpy().astype(np.float32))
            labels.append(y.numpy().astype(np.int64))
    x = np.concatenate(embeddings, axis=0)
    y = np.concatenate(labels, axis=0)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, test_x=x, test_y=y)
    print(f"[ASN degraded] {condition}_s{severity}: {x.shape}")
    return x, y


def load_clean_modality_data(args: argparse.Namespace, device: torch.device):
    clean_cache = args.base_output_dir / "embedding_cache"
    vsn_x, vsn_y, vsn_meta = extract_vsn_embeddings(
        args.vsn_index,
        args.vsn_checkpoint,
        clean_cache / f"vsn_cnn_embeddings_t{args.vsn_train_limit}_e{args.vsn_eval_limit}_s{args.seed}.npz",
        args.vsn_train_limit,
        args.vsn_eval_limit,
        args.extract_batch_size,
        args.seed,
        device,
        args.num_workers,
    )
    asn_x, asn_y, asn_meta = extract_asn_embeddings(
        args.asn_index,
        args.asn_checkpoint,
        clean_cache / f"asn_cnn_embeddings_t{args.asn_train_limit}_e{args.asn_eval_limit}_s{args.seed}.npz",
        args.asn_train_limit,
        args.asn_eval_limit,
        args.extract_batch_size,
        args.seed,
        device,
        args.num_workers,
    )
    vbn_x, vbn_y, vbn_meta = load_vbn_feature_embeddings(args.vbn_features)
    data = {
        "vsn": make_modality_data("vsn", str(vsn_meta["source_representation"]), vsn_x, vsn_y),
        "asn": make_modality_data("asn", str(asn_meta["source_representation"]), asn_x, asn_y),
        "vbn": make_modality_data("vbn", str(vbn_meta["source_representation"]), vbn_x, vbn_y),
    }
    return data, {"vsn": vsn_meta, "asn": asn_meta, "vbn": vbn_meta}


def load_sharing_model(method: str, checkpoint_path: Path, input_dims: dict[str, int], device: torch.device) -> tuple[nn.Module, dict[str, float], dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    saved_args = checkpoint.get("args", {})
    if not isinstance(saved_args, dict):
        saved_args = vars(saved_args)
    embedding_dim = int(saved_args.get("embedding_dim", 64))
    projection_hidden = int(saved_args.get("projection_hidden", 96))
    head_hidden = int(saved_args.get("head_hidden", 32))
    dropout = float(saved_args.get("dropout", 0.1))
    if method == "separate_heads":
        model = SeparateRiskHeadsModel(input_dims, embedding_dim, projection_hidden, head_hidden, dropout)
    elif method == "shared_risk_head":
        model = SharedRiskHeadModel(input_dims, embedding_dim, projection_hidden, head_hidden, dropout)
    elif method == "shared_trunk_node_calibrated":
        model = SharedTrunkNodeCalibratedModel(input_dims, embedding_dim, projection_hidden, head_hidden, dropout)
    else:
        raise ValueError(f"Unknown method: {method}")
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    thresholds = {str(key): float(value) for key, value in checkpoint["thresholds"].items()}
    return model, thresholds, saved_args


def confidence_from_probs(probs: np.ndarray) -> np.ndarray:
    p = np.clip(probs.astype(np.float64), 1e-7, 1.0 - 1e-7)
    entropy = -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))
    return (1.0 - entropy).astype(np.float32)


def best_threshold(probs: np.ndarray, y: np.ndarray) -> float:
    best_t = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.05, 0.95, 181):
        pred = (probs >= threshold).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(threshold)
    return best_t


def embedding_quality_stats(clean_data) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for node, data in clean_data.items():
        norms = np.linalg.norm(data.train_x, axis=1)
        stats[node] = {
            "p50": float(np.percentile(norms, 50)),
            "p95": float(np.percentile(norms, 95)),
            "p99": float(np.percentile(norms, 99)),
        }
    return stats


def quality_weights_from_scaled_embeddings(x: np.ndarray, stats: dict[str, float]) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1)
    scale = max(stats["p95"] - stats["p50"], 1e-6)
    excess = np.maximum(0.0, norms - stats["p95"])
    weights = 1.0 / (1.0 + excess / scale)
    return np.clip(weights, 0.05, 1.0).astype(np.float32)


def label_aligned_fusion(
    probs_by_node: dict[str, np.ndarray],
    labels_by_node: dict[str, np.ndarray],
    nodes: list[str],
    seed: int,
    quality_by_node: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    fused_scores: list[float] = []
    fused_labels: list[int] = []
    for label in [0, 1]:
        candidate_indices = {node: np.where(labels_by_node[node] == label)[0] for node in nodes}
        count = min(len(indices) for indices in candidate_indices.values())
        sampled = {node: rng.permutation(indices)[:count] for node, indices in candidate_indices.items()}
        for idx in range(count):
            scores = np.asarray([probs_by_node[node][sampled[node][idx]] for node in nodes], dtype=np.float32)
            conf = confidence_from_probs(scores).clip(0.05, 1.0)
            if quality_by_node is None:
                quality = np.ones_like(conf)
            else:
                quality = np.asarray([quality_by_node[node][sampled[node][idx]] for node in nodes], dtype=np.float32).clip(0.05, 1.0)
            weights = conf * quality
            fused_scores.append(float(np.sum(scores * weights) / np.sum(weights)))
            fused_labels.append(label)
    return np.asarray(fused_scores, dtype=np.float32), np.asarray(fused_labels, dtype=np.int64)


def scenario_name(vsn_condition: str, vsn_severity: int, asn_condition: str, asn_severity: int) -> str:
    parts = []
    if vsn_condition != "clean":
        parts.append(f"vsn_{vsn_condition}_s{vsn_severity}")
    if asn_condition != "clean":
        parts.append(f"asn_{asn_condition}_s{asn_severity}")
    return "clean" if not parts else "__".join(parts)


def build_scenarios() -> list[dict[str, object]]:
    scenarios: list[dict[str, object]] = [{"scenario": "clean", "vsn_condition": "clean", "vsn_severity": 0, "asn_condition": "clean", "asn_severity": 0}]
    for condition in ["low_light", "blur"]:
        for severity in [2, 3]:
            scenarios.append(
                {
                    "scenario": scenario_name(condition, severity, "clean", 0),
                    "vsn_condition": condition,
                    "vsn_severity": severity,
                    "asn_condition": "clean",
                    "asn_severity": 0,
                }
            )
    for condition in ["snr_noise", "dropout"]:
        for severity in [2, 3]:
            scenarios.append(
                {
                    "scenario": scenario_name("clean", 0, condition, severity),
                    "vsn_condition": "clean",
                    "vsn_severity": 0,
                    "asn_condition": condition,
                    "asn_severity": severity,
                }
            )
    scenarios.extend(
        [
            {
                "scenario": scenario_name("blur", 3, "snr_noise", 3),
                "vsn_condition": "blur",
                "vsn_severity": 3,
                "asn_condition": "snr_noise",
                "asn_severity": 3,
            },
            {
                "scenario": scenario_name("low_light", 3, "dropout", 3),
                "vsn_condition": "low_light",
                "vsn_severity": 3,
                "asn_condition": "dropout",
                "asn_severity": 3,
            },
        ]
    )
    return scenarios


def plot_fusion_metrics(rows: list[dict[str, object]], output_path: Path) -> None:
    fusion_rows = [row for row in rows if row["fusion_set"] == "vsn_asn" and row["fusion_rule"] == "status_aware_embedding_ood"]
    methods = ["separate_heads", "shared_risk_head", "shared_trunk_node_calibrated"]
    scenarios = list(dict.fromkeys([str(row["scenario"]) for row in fusion_rows]))
    fig, ax = plt.subplots(figsize=(13.5, 5.2), dpi=140)
    x = np.arange(len(scenarios))
    width = 0.24
    for offset, method in zip([-width, 0.0, width], methods):
        y = []
        for scenario in scenarios:
            match = [row for row in fusion_rows if row["scenario"] == scenario and row["method"] == method]
            y.append(float(match[0]["f1_positive"]) if match else float("nan"))
        ax.bar(x + offset, y, width=width, label=method)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("VSN+ASN fused F1")
    ax.set_xticks(x, scenarios, rotation=28, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def markdown_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        values = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, sep, *body])


def write_report(
    output_path: Path,
    args: argparse.Namespace,
    per_node_rows: list[dict[str, object]],
    fusion_rows: list[dict[str, object]],
    metadata: dict[str, dict[str, object]],
    elapsed: float,
) -> None:
    vsn_asn_rows = [row for row in fusion_rows if row["fusion_set"] == "vsn_asn"]
    avg_rows = []
    for rule in sorted({str(row["fusion_rule"]) for row in vsn_asn_rows}):
        for method in sorted({str(row["method"]) for row in vsn_asn_rows}):
            rows = [row for row in vsn_asn_rows if row["method"] == method and row["fusion_rule"] == rule]
            avg_rows.append(
                {
                    "fusion_rule": rule,
                    "method": method,
                    "mean_f1": float(np.mean([float(row["f1_positive"]) for row in rows])),
                    "worst_f1": float(np.min([float(row["f1_positive"]) for row in rows])),
                    "mean_recall": float(np.mean([float(row["recall_positive"]) for row in rows])),
                }
            )
    shared_focus = [
        {
            "scenario": row["scenario"],
            "fusion_rule": row["fusion_rule"],
            "fusion_set": row["fusion_set"],
            "f1": row["f1_positive"],
            "recall": row["recall_positive"],
            "accuracy": row["accuracy"],
            "threshold": row["threshold"],
        }
        for row in fusion_rows
        if row["method"] == "shared_trunk_node_calibrated" and row["fusion_rule"] == "status_aware_embedding_ood"
    ]
    lines = [
        "# Shared Higher-Layer Robustness Stress Test",
        "",
        "## Purpose",
        "",
        "This experiment combines the literature-driven robustness direction with the supervisor's parameter-sharing suggestion. It tests whether the shared higher risk layer remains useful when VSN or ASN inputs are synthetically degraded.",
        "",
        "## Setup",
        "",
        "- VSN representation: frozen VSN KD student CNN embedding.",
        "- ASN representation: frozen ASN student CNN embedding.",
        "- VBN representation: ORION AE proxy literature-guided vibration features.",
        "- Shared-layer models: the three models from the clean CNN-embedding parameter-sharing experiment.",
        "- Degradation conditions: VSN low light / blur, ASN SNR noise / dropout, and two combined stress cases.",
        "",
        "## Representation Counts",
        "",
        markdown_table(
            [
                {
                    "node": name,
                    "representation": meta["source_representation"],
                    "dim": meta["embedding_dim"],
                    "train/val/test": f"{meta['sample_counts']['train']}/{meta['sample_counts']['val']}/{meta['sample_counts']['test']}",
                }
                for name, meta in metadata.items()
            ],
            ["node", "representation", "dim", "train/val/test"],
        ),
        "",
        "## VSN+ASN Fusion Robustness Summary",
        "",
        markdown_table(avg_rows, ["fusion_rule", "method", "mean_f1", "worst_f1", "mean_recall"]),
        "",
        "## Shared-Trunk Node-Calibrated Scenario Results with Status-Aware Fusion",
        "",
        markdown_table(shared_focus, ["scenario", "fusion_rule", "fusion_set", "f1", "recall", "accuracy", "threshold"]),
        "",
        "## Interpretation",
        "",
        "- The test checks whether shared higher-layer risk reasoning is robust under controlled sensor degradation.",
        "- The status-aware rule adds a lightweight embedding-OOD gate, reducing the influence of nodes whose embedding distribution shifts away from clean training data.",
        "- VSN+ASN fusion is the more reliable robustness indicator because it uses larger held-out image/audio test sets.",
        "- VSN+ASN+VBN fusion is included as a label-aligned stress test, but it is limited by the very small VBN proxy test split.",
        "",
        "## Claims to Avoid / Cautious Wording",
        "",
        "- Synthetic degradation is controlled robustness stress testing, not real disaster validation.",
        "- VBN is still a proxy structural time-series dataset, not true MPU6050 building vibration validation.",
        "- Three-node fusion remains label-aligned stress testing, not synchronous tri-modal sensing.",
        "- These results evaluate software-level shared higher layers, not physical split inference across hardware.",
        "",
        "## Next Useful Literature-Inspired Experiments",
        "",
        "- A small shared-head federated/continual update experiment, where nodes update only higher risk-layer parameters.",
        "- Optional VSN input-size or attention ablation if we need a stronger model-design comparison.",
        "- Optional ASN handcrafted feature baseline if the dissertation needs an interpretable acoustic baseline.",
        f"",
        f"Runtime: {elapsed:.1f} seconds.",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate shared higher-layer robustness under VSN/ASN degradation.")
    parser.add_argument("--vsn-index", type=Path, default=DEFAULT_VSN_INDEX)
    parser.add_argument("--asn-index", type=Path, default=DEFAULT_ASN_INDEX)
    parser.add_argument("--vbn-features", type=Path, default=DEFAULT_VBN_FEATURES)
    parser.add_argument("--vsn-checkpoint", type=Path, default=DEFAULT_VSN_CHECKPOINT)
    parser.add_argument("--asn-checkpoint", type=Path, default=DEFAULT_ASN_CHECKPOINT)
    parser.add_argument("--base-output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "parameter_sharing_cnn_embeddings")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "parameter_sharing_cnn_embedding_robustness")
    parser.add_argument("--vsn-train-limit", type=int, default=3000)
    parser.add_argument("--vsn-eval-limit", type=int, default=1000)
    parser.add_argument("--asn-train-limit", type=int, default=1600)
    parser.add_argument("--asn-eval-limit", type=int, default=500)
    parser.add_argument("--extract-batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    start = time.perf_counter()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    degraded_cache = args.output_dir / "embedding_cache"
    device = torch.device(args.device)

    clean_data, metadata = load_clean_modality_data(args, device)
    quality_stats = embedding_quality_stats(clean_data)
    input_dims = {name: data.train_x.shape[1] for name, data in clean_data.items()}
    method_names = ["separate_heads", "shared_risk_head", "shared_trunk_node_calibrated"]
    models: dict[str, nn.Module] = {}
    thresholds_by_method: dict[str, dict[str, float]] = {}
    for method in method_names:
        model, thresholds, _saved_args = load_sharing_model(method, args.base_output_dir / f"{method}.pt", input_dims, device)
        models[method] = model
        thresholds_by_method[method] = thresholds

    scenarios = build_scenarios()
    per_node_rows: list[dict[str, object]] = []
    fusion_rows: list[dict[str, object]] = []

    # Tune fusion thresholds on clean validation and keep them fixed under degraded test scenarios.
    fusion_thresholds: dict[str, dict[str, dict[str, float]]] = {method: {} for method in method_names}
    for method, model in models.items():
        val_probs = {node: predict_probs(model, node, clean_data[node].val_x, device) for node in ["vsn", "asn", "vbn"]}
        val_labels = {node: clean_data[node].val_y for node in ["vsn", "asn", "vbn"]}
        val_quality = {node: quality_weights_from_scaled_embeddings(clean_data[node].val_x, quality_stats[node]) for node in ["vsn", "asn", "vbn"]}
        for fusion_set, nodes in {"vsn_asn": ["vsn", "asn"], "vsn_asn_vbn": ["vsn", "asn", "vbn"]}.items():
            fusion_thresholds[method][fusion_set] = {}
            for fusion_rule in ["confidence_only", "status_aware_embedding_ood"]:
                quality = None if fusion_rule == "confidence_only" else val_quality
                fused, labels = label_aligned_fusion(val_probs, val_labels, nodes, args.seed + 100, quality)
                fusion_thresholds[method][fusion_set][fusion_rule] = best_threshold(fused, labels)

    for scenario in scenarios:
        scenario_id = str(scenario["scenario"])
        vsn_condition = str(scenario["vsn_condition"])
        asn_condition = str(scenario["asn_condition"])
        vsn_severity = int(scenario["vsn_severity"])
        asn_severity = int(scenario["asn_severity"])

        if vsn_condition == "clean":
            vsn_x_raw = clean_data["vsn"].test_x
            vsn_y = clean_data["vsn"].test_y
        else:
            raw, vsn_y = extract_degraded_vsn_test_embeddings(
                args.vsn_index,
                args.vsn_checkpoint,
                degraded_cache / f"vsn_{vsn_condition}_s{vsn_severity}_e{args.vsn_eval_limit}_seed{args.seed}.npz",
                args.vsn_eval_limit,
                vsn_condition,
                vsn_severity,
                args.extract_batch_size,
                args.seed,
                device,
                args.num_workers,
            )
            vsn_x_raw = clean_data["vsn"].scaler.transform(raw).astype(np.float32)

        if asn_condition == "clean":
            asn_x_raw = clean_data["asn"].test_x
            asn_y = clean_data["asn"].test_y
        else:
            raw, asn_y = extract_degraded_asn_test_embeddings(
                args.asn_index,
                args.asn_checkpoint,
                degraded_cache / f"asn_{asn_condition}_s{asn_severity}_e{args.asn_eval_limit}_seed{args.seed}.npz",
                args.asn_eval_limit,
                asn_condition,
                asn_severity,
                args.extract_batch_size,
                args.seed,
                device,
                args.num_workers,
            )
            asn_x_raw = clean_data["asn"].scaler.transform(raw).astype(np.float32)

        scenario_x = {"vsn": vsn_x_raw, "asn": asn_x_raw, "vbn": clean_data["vbn"].test_x}
        scenario_y = {"vsn": vsn_y, "asn": asn_y, "vbn": clean_data["vbn"].test_y}
        scenario_quality = {node: quality_weights_from_scaled_embeddings(scenario_x[node], quality_stats[node]) for node in ["vsn", "asn", "vbn"]}

        for method, model in models.items():
            probs_by_node = {node: predict_probs(model, node, scenario_x[node], device) for node in ["vsn", "asn", "vbn"]}
            for node in ["vsn", "asn", "vbn"]:
                row = metric_row(scenario_y[node], probs_by_node[node], thresholds_by_method[method][node])
                per_node_rows.append(
                    {
                        "scenario": scenario_id,
                        "method": method,
                        "node": node,
                        "vsn_condition": vsn_condition,
                        "vsn_severity": vsn_severity,
                        "asn_condition": asn_condition,
                        "asn_severity": asn_severity,
                        **row,
                    }
                )
            for fusion_set, nodes in {"vsn_asn": ["vsn", "asn"], "vsn_asn_vbn": ["vsn", "asn", "vbn"]}.items():
                for fusion_rule in ["confidence_only", "status_aware_embedding_ood"]:
                    quality = None if fusion_rule == "confidence_only" else scenario_quality
                    fused, labels = label_aligned_fusion(probs_by_node, scenario_y, nodes, args.seed + 200, quality)
                    threshold = fusion_thresholds[method][fusion_set][fusion_rule]
                    row = metric_row(labels, fused, threshold)
                    fusion_rows.append(
                        {
                            "scenario": scenario_id,
                            "method": method,
                            "fusion_set": fusion_set,
                            "fusion_rule": fusion_rule,
                            "nodes": "+".join(nodes),
                            "aligned_sample_count": int(len(labels)),
                            "vsn_condition": vsn_condition,
                            "vsn_severity": vsn_severity,
                            "asn_condition": asn_condition,
                            "asn_severity": asn_severity,
                            **row,
                        }
                    )

    elapsed = time.perf_counter() - start
    write_csv(args.output_dir / "per_node_metrics.csv", per_node_rows)
    write_csv(args.output_dir / "fusion_metrics.csv", fusion_rows)
    plot_fusion_metrics(fusion_rows, args.output_dir / "vsn_asn_fusion_robustness.png")
    summary = {
        "task": "shared_higher_layer_robustness_stress_test",
        "metadata": metadata,
        "embedding_quality_stats": quality_stats,
        "fusion_thresholds": fusion_thresholds,
        "scenarios": scenarios,
        "per_node_metrics": per_node_rows,
        "fusion_metrics": fusion_rows,
        "elapsed_seconds": elapsed,
        "cautions": [
            "Synthetic degradation is not real disaster validation.",
            "VBN remains an ORION AE proxy rather than MPU6050 building vibration validation.",
            "Three-node fusion is label-aligned stress testing, not synchronous tri-modal sensing.",
        ],
    }
    save_json(args.output_dir / "summary.json", summary)
    report_path = args.output_dir / "SHARED_HIGHER_LAYER_ROBUSTNESS_REPORT.md"
    write_report(report_path, args, per_node_rows, fusion_rows, metadata, elapsed)

    final_package = PROJECT_ROOT / "outputs" / "reports" / "final_results_package"
    if final_package.exists():
        shutil.copy2(report_path, final_package / report_path.name)
        shutil.copy2(args.output_dir / "fusion_metrics.csv", final_package / "shared_layer_robustness_fusion_metrics.csv")
        shutil.copy2(args.output_dir / "per_node_metrics.csv", final_package / "shared_layer_robustness_per_node_metrics.csv")
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
