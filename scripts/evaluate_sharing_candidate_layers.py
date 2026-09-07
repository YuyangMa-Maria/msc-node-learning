"""Sweep compatible higher-layer sharing boundaries across node modalities.

Every candidate keeps the raw-input encoder private. The sweep moves only the
projection, bottleneck, risk trunk and optional calibrator across the
private/shared boundary, allowing performance to be compared with transferable
parameter count and payload size.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
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
    evaluate_model,
    extract_asn_embeddings,
    extract_vsn_embeddings,
    load_vbn_feature_embeddings,
    make_modality_data,
    macro_summary,
    save_json,
    set_seed,
    train_model,
)


class SharedProjectionRiskHeadModel(nn.Module):
    """Private first adapters with a shared projection and shared risk head."""

    def __init__(self, input_dims: dict[str, int], embedding_dim: int, projection_hidden: int, head_hidden: int, dropout: float) -> None:
        super().__init__()
        self.local_adapters = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(dim, projection_hidden),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                )
                for name, dim in input_dims.items()
            }
        )
        self.shared_projection = nn.Sequential(
            nn.Linear(projection_hidden, embedding_dim),
            nn.ReLU(inplace=True),
        )
        self.shared_head = nn.Sequential(
            nn.Linear(embedding_dim, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, modality: str, x: torch.Tensor) -> torch.Tensor:
        return self.shared_head(self.shared_projection(self.local_adapters[modality](x))).flatten()


class SharedProjectionNodeCalibratedModel(nn.Module):
    def __init__(self, input_dims: dict[str, int], embedding_dim: int, projection_hidden: int, head_hidden: int, dropout: float) -> None:
        super().__init__()
        self.local_adapters = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(dim, projection_hidden),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                )
                for name, dim in input_dims.items()
            }
        )
        self.shared_projection = nn.Sequential(
            nn.Linear(projection_hidden, embedding_dim),
            nn.ReLU(inplace=True),
        )
        self.shared_trunk = nn.Sequential(
            nn.Linear(embedding_dim, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.node_calibrators = nn.ModuleDict({name: nn.Linear(head_hidden, 1) for name in input_dims})

    def forward(self, modality: str, x: torch.Tensor) -> torch.Tensor:
        shared = self.shared_trunk(self.shared_projection(self.local_adapters[modality](x)))
        return self.node_calibrators[modality](shared).flatten()


class SharedBottleneckRiskHeadModel(nn.Module):
    def __init__(
        self,
        input_dims: dict[str, int],
        embedding_dim: int,
        projection_hidden: int,
        head_hidden: int,
        bottleneck_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.projections = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(dim, projection_hidden),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(projection_hidden, embedding_dim),
                    nn.ReLU(inplace=True),
                )
                for name, dim in input_dims.items()
            }
        )
        self.shared_bottleneck = nn.Sequential(
            nn.Linear(embedding_dim, bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.shared_head = nn.Sequential(
            nn.Linear(bottleneck_dim, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, modality: str, x: torch.Tensor) -> torch.Tensor:
        return self.shared_head(self.shared_bottleneck(self.projections[modality](x))).flatten()


class SharedBottleneckNodeCalibratedModel(nn.Module):
    def __init__(
        self,
        input_dims: dict[str, int],
        embedding_dim: int,
        projection_hidden: int,
        bottleneck_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.projections = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(dim, projection_hidden),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(projection_hidden, embedding_dim),
                    nn.ReLU(inplace=True),
                )
                for name, dim in input_dims.items()
            }
        )
        self.shared_bottleneck = nn.Sequential(
            nn.Linear(embedding_dim, bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.node_calibrators = nn.ModuleDict({name: nn.Linear(bottleneck_dim, 1) for name in input_dims})

    def forward(self, modality: str, x: torch.Tensor) -> torch.Tensor:
        shared = self.shared_bottleneck(self.projections[modality](x))
        return self.node_calibrators[modality](shared).flatten()


class SharedProjectionBottleneckCalibratedModel(nn.Module):
    def __init__(
        self,
        input_dims: dict[str, int],
        embedding_dim: int,
        projection_hidden: int,
        bottleneck_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.local_adapters = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(dim, projection_hidden),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                )
                for name, dim in input_dims.items()
            }
        )
        self.shared_projection = nn.Sequential(
            nn.Linear(projection_hidden, embedding_dim),
            nn.ReLU(inplace=True),
        )
        self.shared_bottleneck = nn.Sequential(
            nn.Linear(embedding_dim, bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.node_calibrators = nn.ModuleDict({name: nn.Linear(bottleneck_dim, 1) for name in input_dims})

    def forward(self, modality: str, x: torch.Tensor) -> torch.Tensor:
        shared = self.shared_bottleneck(self.shared_projection(self.local_adapters[modality](x)))
        return self.node_calibrators[modality](shared).flatten()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def shared_parameter_count(model: nn.Module) -> int:
    """Count only tensors that would cross the node communication interface."""
    total = 0
    for name in ["shared_projection", "shared_bottleneck", "shared_head", "shared_trunk"]:
        module = getattr(model, name, None)
        if module is not None:
            total += sum(parameter.numel() for parameter in module.parameters())
    return total


def method_description(method: str) -> str:
    descriptions = {
        "separate_heads": "No shared high layer; each node has its own projection and risk head.",
        "shared_risk_head": "Node-specific projections followed by one shared risk head.",
        "shared_trunk_node_calibrated": "Node-specific projections, shared trunk, then tiny node-specific calibrators.",
        "shared_projection_risk_head": "Node-specific first adapter, then shared projection and shared risk head.",
        "shared_projection_node_calibrated": "Node-specific first adapter, shared projection/trunk, then node-specific calibrators.",
        "shared_bottleneck_risk_head": "Node-specific projections, shared compact bottleneck, then shared risk head.",
        "shared_bottleneck_node_calibrated": "Node-specific projections, shared compact bottleneck, then node-specific calibrators.",
        "shared_projection_bottleneck_calibrated": "Node-specific first adapter, shared projection and bottleneck, then node-specific calibrators.",
    }
    return descriptions[method]


def load_data(args: argparse.Namespace, device: torch.device):
    cache_dir = args.base_output_dir / "embedding_cache"
    vsn_x, vsn_y, vsn_meta = extract_vsn_embeddings(
        args.vsn_index,
        args.vsn_checkpoint,
        cache_dir / f"vsn_cnn_embeddings_t{args.vsn_train_limit}_e{args.vsn_eval_limit}_s{args.seed}.npz",
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
        cache_dir / f"asn_cnn_embeddings_t{args.asn_train_limit}_e{args.asn_eval_limit}_s{args.seed}.npz",
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


def plot_macro_results(rows: list[dict[str, object]], model_rows: list[dict[str, object]], path: Path) -> None:
    test_rows = [row for row in rows if row["split"] == "test" and row["modality"] == "macro"]
    test_rows = sorted(test_rows, key=lambda row: float(row["macro_f1_positive"]), reverse=True)
    param_lookup = {str(row["method"]): float(row["total_parameters"]) for row in model_rows}
    labels = [str(row["method"]) for row in test_rows]
    f1 = [float(row["macro_f1_positive"]) for row in test_rows]
    recall = [float(row["macro_recall_positive"]) for row in test_rows]
    params = [param_lookup[str(row["method"])] / 1000.0 for row in test_rows]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), dpi=140)
    axes[0].bar(x - 0.18, f1, width=0.36, label="Macro F1")
    axes[0].bar(x + 0.18, recall, width=0.36, label="Macro recall")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_xticks(x, labels, rotation=30, ha="right")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()
    axes[1].bar(x, params)
    axes[1].set_ylabel("Trainable high-layer params (K)")
    axes[1].set_xticks(x, labels, rotation=30, ha="right")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
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
    path: Path,
    args: argparse.Namespace,
    metadata: dict[str, dict[str, object]],
    result_rows: list[dict[str, object]],
    model_rows: list[dict[str, object]],
    elapsed: float,
) -> None:
    macro_rows = [row for row in result_rows if row["split"] == "test" and row["modality"] == "macro"]
    macro_rows = sorted(macro_rows, key=lambda row: float(row["macro_f1_positive"]), reverse=True)
    best = macro_rows[0]
    joined_rows = []
    for row in macro_rows:
        model_row = next(model for model in model_rows if model["method"] == row["method"])
        joined_rows.append(
            {
                "method": row["method"],
                "macro_f1": row["macro_f1_positive"],
                "macro_recall": row["macro_recall_positive"],
                "params": model_row["total_parameters"],
                "shared_params": model_row["shared_parameters"],
                "int8_mb": model_row["int8_weight_est_mb"],
            }
        )
    representation_rows = [
        {
            "node": name,
            "representation": meta["source_representation"],
            "dim": meta["embedding_dim"],
            "train/val/test": f"{meta['sample_counts']['train']}/{meta['sample_counts']['val']}/{meta['sample_counts']['test']}",
        }
        for name, meta in metadata.items()
    ]
    lines = [
        "# Sharing Candidate Layer Sweep",
        "",
        "## Purpose",
        "",
        "This experiment explores which higher layers are worth sharing across VSN, ASN, and VBN. It keeps low-level encoders modality-specific and tests several candidate shared layers after frozen local representations.",
        "",
        "## Node Representations",
        "",
        markdown_table(representation_rows, ["node", "representation", "dim", "train/val/test"]),
        "",
        "## Candidate Layers",
        "",
        markdown_table(
            [{"method": method, "description": method_description(method)} for method in [str(row["method"]) for row in macro_rows]],
            ["method", "description"],
        ),
        "",
        "## Held-Out Test Macro Results",
        "",
        markdown_table(joined_rows, ["method", "macro_f1", "macro_recall", "params", "shared_params", "int8_mb"]),
        "",
        "## Main Finding",
        "",
        f"- Best candidate: `{best['method']}` with macro F1 {float(best['macro_f1_positive']):.4f} and macro recall {float(best['macro_recall_positive']):.4f}.",
        "- Sharing too low in the stack is still avoided; all candidates share only higher risk-representation layers after modality-specific encoders/adapters.",
        "",
        "## How to Read This",
        "",
        "- If a shared candidate matches or beats separate heads with fewer parameters, it is worth keeping.",
        "- If a candidate performs well only because VBN is tiny, treat it as feasibility evidence rather than final validation.",
        "- Node-specific calibrators are useful when shared representations need small per-modality output adjustment.",
        "",
        "## Recommended Claim",
        "",
        "The defensible design is modality-specific lower encoders followed by shared higher risk-representation layers, optionally with tiny node-specific calibrators.",
        "",
        "## Claims to Avoid",
        "",
        "- Do not claim that raw visual/audio/vibration encoders are shareable.",
        "- Do not claim hardware split-layer inference yet.",
        "- Do not claim VBN is fully validated; it remains proxy-based.",
        f"",
        f"Runtime: {elapsed:.1f} seconds.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Train all sharing boundaries under a common data and optimisation setup."""
    parser = argparse.ArgumentParser(description="Explore additional high-layer sharing candidates across VSN/ASN/VBN.")
    parser.add_argument("--vsn-index", type=Path, default=DEFAULT_VSN_INDEX)
    parser.add_argument("--asn-index", type=Path, default=DEFAULT_ASN_INDEX)
    parser.add_argument("--vbn-features", type=Path, default=DEFAULT_VBN_FEATURES)
    parser.add_argument("--vsn-checkpoint", type=Path, default=DEFAULT_VSN_CHECKPOINT)
    parser.add_argument("--asn-checkpoint", type=Path, default=DEFAULT_ASN_CHECKPOINT)
    parser.add_argument("--base-output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "parameter_sharing_cnn_embeddings")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "sharing_candidate_layers")
    parser.add_argument("--vsn-train-limit", type=int, default=3000)
    parser.add_argument("--vsn-eval-limit", type=int, default=1000)
    parser.add_argument("--asn-train-limit", type=int, default=1600)
    parser.add_argument("--asn-eval-limit", type=int, default=500)
    parser.add_argument("--extract-batch-size", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--projection-hidden", type=int, default=96)
    parser.add_argument("--head-hidden", type=int, default=32)
    parser.add_argument("--bottleneck-dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    start = time.perf_counter()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    data, metadata = load_data(args, device)
    input_dims = {name: modality_data.train_x.shape[1] for name, modality_data in data.items()}
    methods: dict[str, nn.Module] = {
        "separate_heads": SeparateRiskHeadsModel(input_dims, args.embedding_dim, args.projection_hidden, args.head_hidden, args.dropout),
        "shared_risk_head": SharedRiskHeadModel(input_dims, args.embedding_dim, args.projection_hidden, args.head_hidden, args.dropout),
        "shared_trunk_node_calibrated": SharedTrunkNodeCalibratedModel(input_dims, args.embedding_dim, args.projection_hidden, args.head_hidden, args.dropout),
        "shared_projection_risk_head": SharedProjectionRiskHeadModel(input_dims, args.embedding_dim, args.projection_hidden, args.head_hidden, args.dropout),
        "shared_projection_node_calibrated": SharedProjectionNodeCalibratedModel(input_dims, args.embedding_dim, args.projection_hidden, args.head_hidden, args.dropout),
        "shared_bottleneck_risk_head": SharedBottleneckRiskHeadModel(input_dims, args.embedding_dim, args.projection_hidden, args.head_hidden, args.bottleneck_dim, args.dropout),
        "shared_bottleneck_node_calibrated": SharedBottleneckNodeCalibratedModel(input_dims, args.embedding_dim, args.projection_hidden, args.bottleneck_dim, args.dropout),
        "shared_projection_bottleneck_calibrated": SharedProjectionBottleneckCalibratedModel(input_dims, args.embedding_dim, args.projection_hidden, args.bottleneck_dim, args.dropout),
    }

    result_rows: list[dict[str, object]] = []
    model_rows: list[dict[str, object]] = []
    for method, model in methods.items():
        print(f"[train] {method}")
        trained, history, thresholds = train_model(model, data, device, args.epochs, args.batch_size, args.lr, args.weight_decay)
        for split in ["val", "test"]:
            metrics = evaluate_model(trained, data, split, device, thresholds)
            macro = macro_summary(metrics)
            for modality, row in metrics.items():
                result_rows.append({"method": method, "split": split, "modality": modality, **row})
            result_rows.append(
                {
                    "method": method,
                    "split": split,
                    "modality": "macro",
                    "macro_accuracy": macro["macro_accuracy"],
                    "macro_f1_positive": macro["macro_f1_positive"],
                    "macro_recall_positive": macro["macro_recall_positive"],
                    "macro_precision_positive": macro["macro_precision_positive"],
                }
            )
        params = parameter_count(trained)
        shared_params = shared_parameter_count(trained)
        model_rows.append(
            {
                "method": method,
                "total_parameters": params,
                "shared_parameters": shared_params,
                "local_or_node_specific_parameters": params - shared_params,
                "fp32_mb": params * 4 / (1024 * 1024),
                "int8_weight_est_mb": params / (1024 * 1024),
            }
        )
        torch.save({"model_state": trained.state_dict(), "thresholds": thresholds, "args": vars(args), "history": history}, args.output_dir / f"{method}.pt")

    elapsed = time.perf_counter() - start
    write_csv(args.output_dir / "metrics.csv", result_rows)
    write_csv(args.output_dir / "model_sizes.csv", model_rows)
    plot_macro_results(result_rows, model_rows, args.output_dir / "sharing_candidate_macro_results.png")
    summary = {
        "task": "sharing_candidate_layer_sweep",
        "metadata": metadata,
        "metrics": result_rows,
        "model_sizes": model_rows,
        "elapsed_seconds": elapsed,
        "cautions": [
            "All candidates share higher layers only; raw modality encoders remain modality-specific.",
            "VBN remains proxy-based.",
            "Memory values are parameter-size estimates, not hardware SRAM profiling.",
        ],
    }
    save_json(args.output_dir / "summary.json", summary)
    report_path = args.output_dir / "SHARING_CANDIDATE_LAYERS_REPORT.md"
    write_report(report_path, args, metadata, result_rows, model_rows, elapsed)

    final_package = PROJECT_ROOT / "outputs" / "reports" / "final_results_package"
    if final_package.exists():
        shutil.copy2(report_path, final_package / report_path.name)
        shutil.copy2(args.output_dir / "metrics.csv", final_package / "sharing_candidate_layers_metrics.csv")
        shutil.copy2(args.output_dir / "model_sizes.csv", final_package / "sharing_candidate_layers_model_sizes.csv")
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
