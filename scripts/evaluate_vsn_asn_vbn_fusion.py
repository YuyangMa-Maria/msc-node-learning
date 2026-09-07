"""Run a label-aligned three-node fusion stress test.

VBN proxy records are sampled to match VSN/ASN labels, not timestamps or physical
events. The experiment checks fusion behaviour when a third score is available;
it must not be described as synchronous tri-modal sensing.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from evaluate_vsn_asn_fusion import (
    PROJECT_ROOT,
    collect_scores,
    confidence_from_risk,
    load_asn,
    load_vsn,
    split_pairs,
)
from train_vbn_orion import DEFAULT_INDEX, build_feature_table, read_rows


def best_f1_threshold(scores: torch.Tensor, labels: torch.Tensor) -> float:
    y = labels.detach().cpu().int()
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in torch.unique(scores.detach().cpu()).tolist():
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
        if not values:
            raise RuntimeError(f"No VBN scores available for label={raw_label}")
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


def fused_scores(scores: dict[str, torch.Tensor], condition: dict[str, object]) -> dict[str, torch.Tensor]:
    v = scores["vsn"]
    a = scores["asn"]
    b = scores["vbn"]
    cv = confidence_from_risk(v)
    ca = confidence_from_risk(a)
    cb = confidence_from_risk(b)
    qv = float(condition["visual_quality"])
    qa = float(condition["audio_quality"])
    qb = float(condition["vbn_quality"])
    qualities = torch.tensor([qv, qa, qb], dtype=torch.float32)
    stacked = torch.stack([v, a, b], dim=0)
    confs = torch.stack([cv, ca, cb], dim=0)
    quality_weights = confs * qualities[:, None]
    quality_sum = quality_weights.sum(dim=0).clamp_min(1e-7)
    quality_aware = (quality_weights * stacked).sum(dim=0) / quality_sum
    active = qualities >= 0.5
    if int(active.sum()) > 0:
        gated_scores = stacked[active]
        gated_confs = confs[active]
        quality_gate = (gated_scores * gated_confs).sum(dim=0) / gated_confs.sum(dim=0).clamp_min(1e-7)
        quality_filtered_max = torch.max(gated_scores, dim=0).values
    else:
        quality_gate = quality_aware
        quality_filtered_max = quality_aware
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
    if method not in {"fusion_quality_gate_3", "fusion_quality_filtered_max_3"}:
        return thresholds[method]
    if float(condition["visual_quality"]) >= 0.5 and float(condition["audio_quality"]) < 0.5 and float(condition["vbn_quality"]) < 0.5:
        return thresholds["vsn_only"]
    if float(condition["audio_quality"]) >= 0.5 and float(condition["visual_quality"]) < 0.5 and float(condition["vbn_quality"]) < 0.5:
        return thresholds["asn_only"]
    if float(condition["vbn_quality"]) >= 0.5 and float(condition["visual_quality"]) < 0.5 and float(condition["audio_quality"]) < 0.5:
        return thresholds["vbn_only"]
    if method == "fusion_quality_filtered_max_3":
        return thresholds["fusion_quality_filtered_max_3"]
    return thresholds["fusion_quality_aware_3"]


def run_experiment(args: argparse.Namespace) -> tuple[list[dict[str, object]], dict[str, float]]:
    device = torch.device(args.device)
    vsn_model, vsn_temperature, image_size = load_vsn(args.vsn_run_dir, args.vsn_calibration_json, device)
    asn_model, asn_temperature, target_audio_samples = load_asn(args.asn_run_dir, args.asn_calibration_json, device)
    all_samples = split_pairs(args.train, args.val, args.seed)
    val_samples = [sample for sample in all_samples if sample.split == "val"]
    test_samples = [sample for sample in all_samples if sample.split == "test"]
    vbn_pool = load_vbn_split_scores(args.vbn_run_dir, args.vbn_index, args.vbn_max_points, args.vbn_sample_rate)
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
            "name": "vsn_blur_asn_clean_vbn_clean",
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
            "name": "asn_noise_vsn_clean_vbn_clean",
            "image_corruption": "clean",
            "image_severity": 0,
            "audio_corruption": "gaussian_noise",
            "audio_severity": 3,
            "vbn_mode": "clean",
            "visual_quality": 1.0,
            "audio_quality": 0.35,
            "vbn_quality": 1.0,
        },
        {
            "name": "vsn_asn_degraded_vbn_clean",
            "image_corruption": "blur",
            "image_severity": 3,
            "audio_corruption": "gaussian_noise",
            "audio_severity": 3,
            "vbn_mode": "clean",
            "visual_quality": 0.20,
            "audio_quality": 0.35,
            "vbn_quality": 1.0,
        },
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
            "audio_corruption": "gaussian_noise",
            "audio_severity": 3,
            "vbn_mode": "uncertain",
            "visual_quality": 0.20,
            "audio_quality": 0.35,
            "vbn_quality": 0.35,
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
    val_scores["vbn"] = align_vbn_scores(val_scores["labels"], vbn_pool["val"], args.seed)
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
        base_vbn = align_vbn_scores(condition_scores["labels"], vbn_pool["test"], args.seed)
        condition_scores["vbn"] = degrade_vbn(base_vbn, str(condition["vbn_mode"]))
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
                    "vbn_quality": condition["vbn_quality"],
                    **result,
                }
            )
            print(f"{condition['name']} {method}: f1={result['f1_positive']:.4f} recall={result['recall_positive']:.4f}")
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
    json_path.write_text(json.dumps({"thresholds": thresholds, "results": rows}, indent=2), encoding="utf-8")

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
        "# VSN + ASN + VBN Fusion Experiment Report",
        "",
        "## Setup",
        "",
        "- VSN/ASN source: paired image/audio samples from the multimodal concrete crack dataset",
        "- VBN source: ORION AE proxy scores sampled by matching binary label",
        "- This is a label-aligned fusion stress test, not a real synchronized three-sensor dataset.",
        "- Thresholds are selected on the clean validation split and then fixed for all test conditions.",
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
            "`fusion_quality_filtered_max_3` first removes low-quality modalities and then applies conservative max-risk fusion to the remaining reliable nodes. This keeps the safety-oriented behaviour of max-risk fusion while avoiding degraded nodes that can distort the fused decision.",
            "",
            "The three-node fusion test shows whether VBN can act as an additional local evidence source when VSN or ASN is degraded. The key limitation is that VBN is label-aligned from a separate proxy dataset, so this result should be presented as system-level fusion validation rather than real-world synchronized multimodal performance.",
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
    parser = argparse.ArgumentParser(description="Evaluate label-aligned VSN+ASN+VBN three-node fusion.")
    parser.add_argument("--vsn-run-dir", type=Path, default=PROJECT_ROOT / "outputs" / "vsn_binary_mixed_full" / "mobilenet_v3_small_pretrained")
    parser.add_argument("--vsn-calibration-json", type=Path, default=PROJECT_ROOT / "outputs" / "vsn_calibration" / "vsn_binary_mixed_full_mobilenet_v3_small_pretrained_calibration.json")
    parser.add_argument("--asn-run-dir", type=Path, default=PROJECT_ROOT / "outputs" / "asn_audio_formal" / "tiny_logmel_cnn")
    parser.add_argument("--asn-calibration-json", type=Path, default=PROJECT_ROOT / "outputs" / "asn_calibration" / "asn_audio_formal_tiny_logmel_cnn_calibration.json")
    parser.add_argument("--vbn-run-dir", type=Path, default=PROJECT_ROOT / "outputs" / "vbn_orion")
    parser.add_argument("--vbn-index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--vbn-max-points", type=int, default=200_000)
    parser.add_argument("--vbn-sample-rate", type=float, default=5_000_000.0)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "fusion_vsn_asn_vbn")
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
