"""Run the formal multi-seed comparison of retained sharing strategies.

Results are paired by random seed so that each shared candidate is compared with
the separate-head baseline under the same initialisation and data. Confidence
intervals and Wilcoxon tests prevent a favourable single run from being
reported as a reliable improvement.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import t as student_t
from scipy.stats import wilcoxon
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn

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
    best_threshold,
    predict_probs,
    set_seed,
    train_model,
)
from evaluate_sharing_candidate_layers import SharedProjectionRiskHeadModel, load_data


METHOD_ORDER = [
    "separate_heads",
    "shared_risk_head",
    "shared_trunk_node_calibrated",
    "shared_projection_risk_head",
]
MODALITIES = ["vsn", "asn", "vbn"]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, value: object) -> None:
    def convert(item: object) -> object:
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, dict):
            return {str(key): convert(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [convert(val) for val in item]
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, float) and not math.isfinite(item):
            return None
        return item

    with path.open("w", encoding="utf-8") as handle:
        json.dump(convert(value), handle, indent=2, ensure_ascii=False)


def expected_calibration_error(labels: np.ndarray, probs: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = max(len(labels), 1)
    ece = 0.0
    for idx in range(bins):
        lower = edges[idx]
        upper = edges[idx + 1]
        if idx == bins - 1:
            mask = (probs >= lower) & (probs <= upper)
        else:
            mask = (probs >= lower) & (probs < upper)
        count = int(mask.sum())
        if count == 0:
            continue
        confidence = float(probs[mask].mean())
        prevalence = float(labels[mask].mean())
        ece += (count / total) * abs(confidence - prevalence)
    return float(ece)


def metric_dict(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (probs >= threshold).astype(np.int64)
    clipped = np.clip(probs, 1e-7, 1.0 - 1e-7)
    return {
        "accuracy": float(accuracy_score(labels, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, pred)),
        "precision": float(precision_score(labels, pred, zero_division=0)),
        "recall": float(recall_score(labels, pred, zero_division=0)),
        "f1": float(f1_score(labels, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probs)),
        "average_precision": float(average_precision_score(labels, probs)),
        "brier": float(brier_score_loss(labels, probs)),
        "nll": float(log_loss(labels, clipped, labels=[0, 1])),
        "ece_15": expected_calibration_error(labels, probs, bins=15),
        "threshold": float(threshold),
    }


def confidence_interval(values: list[float], confidence: float = 0.95) -> tuple[float, float, float, float]:
    """Return mean, sample SD and a Student-t confidence interval."""
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    if len(array) < 2:
        return mean, 0.0, mean, mean
    std = float(array.std(ddof=1))
    critical = float(student_t.ppf((1.0 + confidence) / 2.0, df=len(array) - 1))
    half_width = critical * std / math.sqrt(len(array))
    return mean, std, mean - half_width, mean + half_width


def instantiate_models(input_dims: dict[str, int], args: argparse.Namespace) -> dict[str, nn.Module]:
    return {
        "separate_heads": SeparateRiskHeadsModel(
            input_dims, args.embedding_dim, args.projection_hidden, args.head_hidden, args.dropout
        ),
        "shared_risk_head": SharedRiskHeadModel(
            input_dims, args.embedding_dim, args.projection_hidden, args.head_hidden, args.dropout
        ),
        "shared_trunk_node_calibrated": SharedTrunkNodeCalibratedModel(
            input_dims, args.embedding_dim, args.projection_hidden, args.head_hidden, args.dropout
        ),
        "shared_projection_risk_head": SharedProjectionRiskHeadModel(
            input_dims, args.embedding_dim, args.projection_hidden, args.head_hidden, args.dropout
        ),
    }


def aggregate_seed_metrics(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Summarise each method and modality across matched random seeds."""
    metrics = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
        "brier",
        "nll",
        "ece_15",
    ]
    aggregate: list[dict[str, object]] = []
    for method in METHOD_ORDER:
        for modality in [*MODALITIES, "macro"]:
            selected = [row for row in rows if row["method"] == method and row["modality"] == modality]
            for metric in metrics:
                values = [float(row[metric]) for row in selected if metric in row]
                if not values:
                    continue
                mean, std, ci_low, ci_high = confidence_interval(values)
                aggregate.append(
                    {
                        "method": method,
                        "modality": modality,
                        "metric": metric,
                        "n_seeds": len(values),
                        "mean": mean,
                        "std": std,
                        "ci95_low": ci_low,
                        "ci95_high": ci_high,
                    }
                )
    return aggregate


def paired_comparisons(rows: list[dict[str, object]], baseline: str = "separate_heads") -> list[dict[str, object]]:
    """Compare candidates with the baseline using per-seed metric deltas."""
    comparisons: list[dict[str, object]] = []
    metrics = ["f1", "recall", "balanced_accuracy", "roc_auc", "brier", "ece_15"]
    seeds = sorted({int(row["seed"]) for row in rows})
    lookup = {
        (str(row["method"]), str(row["modality"]), int(row["seed"])): row
        for row in rows
    }
    for method in METHOD_ORDER:
        if method == baseline:
            continue
        for modality in [*MODALITIES, "macro"]:
            for metric in metrics:
                deltas: list[float] = []
                for seed in seeds:
                    candidate = lookup.get((method, modality, seed))
                    reference = lookup.get((baseline, modality, seed))
                    if candidate is None or reference is None or metric not in candidate or metric not in reference:
                        continue
                    candidate_value = float(candidate[metric])
                    reference_value = float(reference[metric])
                    delta = candidate_value - reference_value
                    if metric in {"brier", "ece_15"}:
                        delta = -delta
                    deltas.append(delta)
                if not deltas:
                    continue
                mean, std, ci_low, ci_high = confidence_interval(deltas)
                try:
                    test = wilcoxon(deltas, zero_method="wilcox", alternative="two-sided")
                    p_value = float(test.pvalue)
                except ValueError:
                    p_value = 1.0
                comparisons.append(
                    {
                        "candidate": method,
                        "baseline": baseline,
                        "modality": modality,
                        "metric": metric,
                        "direction": "positive_favors_candidate",
                        "n_pairs": len(deltas),
                        "mean_delta": mean,
                        "std_delta": std,
                        "ci95_low": ci_low,
                        "ci95_high": ci_high,
                        "wilcoxon_p": p_value,
                    }
                )
    return comparisons


def plot_macro_f1(aggregate: list[dict[str, object]], output_path: Path) -> None:
    rows = [
        row
        for row in aggregate
        if row["modality"] == "macro" and row["metric"] in {"f1", "recall"}
    ]
    fig, ax = plt.subplots(figsize=(10.5, 5.2), dpi=160)
    x = np.arange(len(METHOD_ORDER))
    width = 0.36
    for offset, metric in [(-width / 2, "f1"), (width / 2, "recall")]:
        selected = [
            next(row for row in rows if row["method"] == method and row["metric"] == metric)
            for method in METHOD_ORDER
        ]
        means = np.asarray([float(row["mean"]) for row in selected])
        lower = np.asarray([float(row["ci95_low"]) for row in selected])
        upper = np.asarray([float(row["ci95_high"]) for row in selected])
        errors = np.vstack([means - lower, upper - means])
        ax.bar(x + offset, means, width=width, label=metric.upper(), yerr=errors, capsize=4)
    ax.set_xticks(x, METHOD_ORDER, rotation=22, ha="right")
    ax.set_ylim(0.80, 1.01)
    ax.set_ylabel("Held-out macro score")
    ax.set_title("Parameter-sharing stability across training seeds (95% CI)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def format_value(mean: float, std: float) -> str:
    return f"{mean:.4f} +/- {std:.4f}"


def write_report(
    path: Path,
    args: argparse.Namespace,
    aggregate: list[dict[str, object]],
    comparisons: list[dict[str, object]],
    elapsed_seconds: float,
) -> None:
    def find(method: str, modality: str, metric: str) -> dict[str, object]:
        return next(
            row
            for row in aggregate
            if row["method"] == method and row["modality"] == modality and row["metric"] == metric
        )

    lines = [
        "# Conference-Standard Statistical Validation",
        "",
        "## Protocol",
        "",
        f"- Training seeds: `{args.seeds}`.",
        f"- Data/subsample seed fixed at `{args.data_seed}` so seed variation measures optimisation stability.",
        "- Frozen VSN/ASN student embeddings and VBN literature-guided proxy features were used.",
        "- Model selection and decision thresholds used validation data only; the held-out test set was evaluated once per seed.",
        "- Reported intervals are two-sided 95% Student-t confidence intervals across independent training seeds.",
        "- Wilcoxon signed-rank tests are exploratory because five seeds provide limited statistical power.",
        "",
        "## Held-Out Macro Results",
        "",
        "| Method | F1 mean +/- SD | F1 95% CI | Recall mean +/- SD | AUROC mean +/- SD | ECE mean +/- SD |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in METHOD_ORDER:
        f1 = find(method, "macro", "f1")
        recall = find(method, "macro", "recall")
        auc = find(method, "macro", "roc_auc")
        ece = find(method, "macro", "ece_15")
        lines.append(
            f"| `{method}` | {format_value(float(f1['mean']), float(f1['std']))} "
            f"| [{float(f1['ci95_low']):.4f}, {float(f1['ci95_high']):.4f}] "
            f"| {format_value(float(recall['mean']), float(recall['std']))} "
            f"| {format_value(float(auc['mean']), float(auc['std']))} "
            f"| {format_value(float(ece['mean']), float(ece['std']))} |"
        )

    lines.extend(
        [
            "",
            "## Paired Macro F1 Versus Separate Heads",
            "",
            "| Candidate | Mean delta | 95% CI | Wilcoxon p |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    paired_f1 = [
        row for row in comparisons if row["modality"] == "macro" and row["metric"] == "f1"
    ]
    for row in paired_f1:
        lines.append(
            f"| `{row['candidate']}` | {float(row['mean_delta']):+.4f} "
            f"| [{float(row['ci95_low']):+.4f}, {float(row['ci95_high']):+.4f}] "
            f"| {float(row['wilcoxon_p']):.4f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation Rules",
            "",
            "- A higher single-seed score is not treated as sufficient evidence of superiority.",
            "- If a paired 95% CI crosses zero, the result should be described as comparable rather than definitively better.",
            "- Calibration metrics are reported because the output is used as a damage-evidence/risk-proxy score.",
            "- VBN remains a proxy dataset, so the macro result is a software-level heterogeneous-node experiment.",
            "- The tri-modal samples are not synchronous observations of the same physical event.",
            "",
            "## Remaining Publication Gaps",
            "",
            "- Perceptual near-duplicate auditing and source/session-grouped split verification.",
            "- Actual microcontroller Flash, peak SRAM, latency, and energy profiling.",
            "- Real or controlled synchronous VSN/ASN/VBN acquisition.",
            "- A personalised federated baseline beyond FedAvg for shared-module updating.",
            "",
            f"Runtime: {elapsed_seconds:.1f} seconds.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_seed_list(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if len(seeds) < 2:
        raise argparse.ArgumentTypeError("Provide at least two comma-separated seeds.")
    return seeds


def main() -> None:
    """Train each method across seeds and produce report-ready comparisons."""
    parser = argparse.ArgumentParser(
        description="Run multi-seed statistical validation for the key heterogeneous parameter-sharing baselines."
    )
    parser.add_argument("--vsn-index", type=Path, default=DEFAULT_VSN_INDEX)
    parser.add_argument("--asn-index", type=Path, default=DEFAULT_ASN_INDEX)
    parser.add_argument("--vbn-features", type=Path, default=DEFAULT_VBN_FEATURES)
    parser.add_argument("--vsn-checkpoint", type=Path, default=DEFAULT_VSN_CHECKPOINT)
    parser.add_argument("--asn-checkpoint", type=Path, default=DEFAULT_ASN_CHECKPOINT)
    parser.add_argument(
        "--base-output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "parameter_sharing_cnn_embeddings",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "conference_statistical_validation",
    )
    parser.add_argument("--vsn-train-limit", type=int, default=3000)
    parser.add_argument("--vsn-eval-limit", type=int, default=1000)
    parser.add_argument("--asn-train-limit", type=int, default=1600)
    parser.add_argument("--asn-eval-limit", type=int, default=500)
    parser.add_argument("--extract-batch-size", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--projection-hidden", type=int, default=96)
    parser.add_argument("--head-hidden", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--seeds", type=parse_seed_list, default=parse_seed_list("11,23,42,67,101"))
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    start = time.perf_counter()

    original_seed = getattr(args, "seed", None)
    args.seed = args.data_seed
    data, metadata = load_data(args, device)
    if original_seed is None:
        delattr(args, "seed")
    else:
        args.seed = original_seed
    input_dims = {name: node_data.train_x.shape[1] for name, node_data in data.items()}

    per_seed_rows: list[dict[str, object]] = []
    prediction_payload: dict[str, np.ndarray] = {}
    for seed in args.seeds:
        print(f"[seed {seed}]")
        set_seed(seed)
        random.seed(seed)
        models = instantiate_models(input_dims, args)
        for method in METHOD_ORDER:
            print(f"  [train] {method}")
            set_seed(seed)
            model, _history, thresholds = train_model(
                models[method],
                data,
                device,
                args.epochs,
                args.batch_size,
                args.lr,
                args.weight_decay,
            )
            modality_metrics: list[dict[str, float]] = []
            for modality in MODALITIES:
                node_data = data[modality]
                probs = predict_probs(model, modality, node_data.test_x, device)
                metrics = metric_dict(node_data.test_y, probs, thresholds[modality])
                per_seed_rows.append(
                    {
                        "seed": seed,
                        "method": method,
                        "modality": modality,
                        "n_test": len(node_data.test_y),
                        **metrics,
                    }
                )
                modality_metrics.append(metrics)
                prediction_payload[f"{method}__{seed}__{modality}__probs"] = probs.astype(np.float32)
                prediction_payload[f"{method}__{seed}__{modality}__labels"] = node_data.test_y.astype(np.int8)
            macro_row = {
                metric: float(np.mean([row[metric] for row in modality_metrics]))
                for metric in modality_metrics[0]
                if metric != "threshold"
            }
            per_seed_rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "modality": "macro",
                    "n_test": sum(len(data[name].test_y) for name in MODALITIES),
                    **macro_row,
                }
            )

    aggregate_rows = aggregate_seed_metrics(per_seed_rows)
    comparison_rows = paired_comparisons(per_seed_rows)
    elapsed = time.perf_counter() - start

    write_csv(args.output_dir / "per_seed_metrics.csv", per_seed_rows)
    write_csv(args.output_dir / "aggregate_metrics.csv", aggregate_rows)
    write_csv(args.output_dir / "paired_comparisons.csv", comparison_rows)
    np.savez_compressed(args.output_dir / "heldout_predictions.npz", **prediction_payload)
    plot_macro_f1(aggregate_rows, args.output_dir / "macro_seed_stability.png")
    write_report(
        args.output_dir / "CONFERENCE_STATISTICAL_VALIDATION_REPORT.md",
        args,
        aggregate_rows,
        comparison_rows,
        elapsed,
    )
    save_json(
        args.output_dir / "summary.json",
        {
            "task": "conference_statistical_validation",
            "protocol": {
                "data_seed": args.data_seed,
                "training_seeds": args.seeds,
                "epochs": args.epochs,
                "validation_selected_thresholds": True,
                "heldout_test": True,
            },
            "metadata": metadata,
            "aggregate_metrics": aggregate_rows,
            "paired_comparisons": comparison_rows,
            "elapsed_seconds": elapsed,
            "cautions": [
                "VBN uses a proxy structural time-series dataset.",
                "Modalities are not synchronous observations of the same physical event.",
                "Five-seed Wilcoxon tests have limited statistical power.",
                "Frozen embeddings isolate higher-layer sharing; this is not end-to-end representation training.",
            ],
        },
    )
    print(f"Saved conference validation to {args.output_dir}")


if __name__ == "__main__":
    main()
