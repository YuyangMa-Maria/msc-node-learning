"""Compare the two retained sharing boundaries under harder system conditions.

The candidates were selected by the earlier boundary sweep. This stage keeps
their trained checkpoints fixed while testing degradation, label-aligned fusion
and partial participation, rather than selecting a new architecture on test
performance.
"""

from __future__ import annotations

import argparse
import copy
import csv
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
    SharedTrunkNodeCalibratedModel,
    best_threshold,
    make_loader,
    metric_row,
    predict_probs,
    save_json,
)
from evaluate_shared_layer_robustness import (
    build_scenarios,
    embedding_quality_stats,
    extract_degraded_asn_test_embeddings,
    extract_degraded_vsn_test_embeddings,
    label_aligned_fusion,
    quality_weights_from_scaled_embeddings,
)
from evaluate_sharing_candidate_layers import SharedProjectionRiskHeadModel, load_data


NODES = ["vsn", "asn", "vbn"]
CANDIDATES = {
    "best_performance_shared_trunk_calibrator": "shared_trunk_node_calibrated",
    "best_lightweight_shared_projection_head": "shared_projection_risk_head",
}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def instantiate_candidate(candidate: str, input_dims: dict[str, int], args: argparse.Namespace) -> nn.Module:
    if candidate == "best_performance_shared_trunk_calibrator":
        return SharedTrunkNodeCalibratedModel(input_dims, args.embedding_dim, args.projection_hidden, args.head_hidden, args.dropout)
    if candidate == "best_lightweight_shared_projection_head":
        return SharedProjectionRiskHeadModel(input_dims, args.embedding_dim, args.projection_hidden, args.head_hidden, args.dropout)
    raise ValueError(f"Unknown candidate: {candidate}")


def load_candidate_models(args: argparse.Namespace, input_dims: dict[str, int], device: torch.device) -> dict[str, nn.Module]:
    models: dict[str, nn.Module] = {}
    for candidate, method in CANDIDATES.items():
        model = instantiate_candidate(candidate, input_dims, args).to(device)
        checkpoint = torch.load(args.candidate_dir / f"{method}.pt", map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        models[candidate] = model
    return models


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def shared_module_names(candidate: str) -> list[str]:
    if candidate == "best_performance_shared_trunk_calibrator":
        return ["shared_trunk"]
    if candidate == "best_lightweight_shared_projection_head":
        return ["shared_projection", "shared_head"]
    raise ValueError(f"Unknown candidate: {candidate}")


def shared_parameter_count(model: nn.Module, candidate: str) -> int:
    return sum(sum(parameter.numel() for parameter in getattr(model, name).parameters()) for name in shared_module_names(candidate))


def shared_state(model: nn.Module, candidate: str) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for module_name in shared_module_names(candidate):
        module = getattr(model, module_name)
        for key, value in module.state_dict().items():
            state[f"{module_name}.{key}"] = value.detach().cpu().clone()
    return state


def load_shared_state(model: nn.Module, candidate: str, state: dict[str, torch.Tensor], device: torch.device) -> None:
    for module_name in shared_module_names(candidate):
        module = getattr(model, module_name)
        prefix = f"{module_name}."
        module_state = {key[len(prefix) :]: value.to(device) for key, value in state.items() if key.startswith(prefix)}
        module.load_state_dict(module_state)


def reset_shared_modules(model: nn.Module, candidate: str, seed: int) -> None:
    torch.manual_seed(seed)
    for module_name in shared_module_names(candidate):
        module = getattr(model, module_name)
        for child in module.modules():
            if hasattr(child, "reset_parameters"):
                child.reset_parameters()


def freeze_all_but_shared_modules(model: nn.Module, candidate: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for module_name in shared_module_names(candidate):
        for parameter in getattr(model, module_name).parameters():
            parameter.requires_grad = True


def average_states(states: list[dict[str, torch.Tensor]], weights: list[float]) -> dict[str, torch.Tensor]:
    total = float(sum(weights))
    norm = [weight / total for weight in weights]
    averaged: dict[str, torch.Tensor] = {}
    for key in states[0]:
        value = states[0][key].float() * norm[0]
        for state, weight in zip(states[1:], norm[1:]):
            value = value + state[key].float() * weight
        averaged[key] = value
    return averaged


def build_loss(labels: np.ndarray, device: torch.device) -> nn.Module:
    positives = float(np.sum(labels == 1))
    negatives = float(np.sum(labels == 0))
    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negatives / max(positives, 1.0), dtype=torch.float32, device=device))


def local_train_shared(
    global_model: nn.Module,
    global_state: dict[str, torch.Tensor],
    candidate: str,
    node: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    device: torch.device,
    local_epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
) -> dict[str, torch.Tensor]:
    model = copy.deepcopy(global_model).to(device)
    load_shared_state(model, candidate, global_state, device)
    freeze_all_but_shared_modules(model, candidate)
    model.train()
    loader = make_loader(train_x, train_y, batch_size, shuffle=True)
    criterion = build_loss(train_y, device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    for _ in range(local_epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(node, xb), yb)
            loss.backward()
            optimizer.step()
    return shared_state(model, candidate)


def evaluate_candidate(model: nn.Module, data, device: torch.device, run_id: str, round_idx: int) -> tuple[list[dict[str, object]], dict[str, float], dict[str, float]]:
    thresholds: dict[str, float] = {}
    rows: list[dict[str, object]] = []
    for node, node_data in data.items():
        val_probs = predict_probs(model, node, node_data.val_x, device)
        thresholds[node] = best_threshold(val_probs, node_data.val_y)
    for split in ["val", "test"]:
        for node, node_data in data.items():
            x = getattr(node_data, f"{split}_x")
            y = getattr(node_data, f"{split}_y")
            probs = predict_probs(model, node, x, device)
            metric = metric_row(y, probs, thresholds[node])
            rows.append({"run_id": run_id, "round": round_idx, "split": split, "modality": node, **metric})
    test_rows = [row for row in rows if row["split"] == "test"]
    macro = {
        "macro_f1": float(np.mean([float(row["f1_positive"]) for row in test_rows])),
        "macro_recall": float(np.mean([float(row["recall_positive"]) for row in test_rows])),
        "macro_accuracy": float(np.mean([float(row["accuracy"]) for row in test_rows])),
    }
    return rows, macro, thresholds


def participants_for_schedule(schedule: str, round_idx: int) -> list[str]:
    if schedule == "full_participation":
        return NODES.copy()
    if schedule == "rotating_missing_one":
        missing = ["vbn", "asn", "vsn"][(round_idx - 1) % 3]
        return [node for node in NODES if node != missing]
    if schedule == "sparse_one_node":
        return [NODES[(round_idx - 1) % 3]]
    raise ValueError(f"Unknown schedule: {schedule}")


def run_partial_update(
    candidate: str,
    base_model: nn.Module,
    data,
    schedule: str,
    reset_shared: bool,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    model = copy.deepcopy(base_model).to(device)
    if reset_shared:
        reset_shared_modules(model, candidate, args.seed + 1700)
    freeze_all_but_shared_modules(model, candidate)
    global_state = shared_state(model, candidate)
    client_sizes = {node: len(data[node].train_y) for node in NODES}
    run_id = f"{candidate}__{'scratch' if reset_shared else 'continual'}__{schedule}"
    metric_rows: list[dict[str, object]] = []
    round_rows: list[dict[str, object]] = []
    rows, macro, _thresholds = evaluate_candidate(model, data, device, run_id, 0)
    metric_rows.extend(rows)
    round_rows.append(
        {
            "run_id": run_id,
            "candidate": candidate,
            "schedule": schedule,
            "reset_shared": reset_shared,
            "round": 0,
            "participants": "initial",
            "participant_count": 0,
            **macro,
        }
    )
    for round_idx in range(1, args.rounds + 1):
        participants = participants_for_schedule(schedule, round_idx)
        states: list[dict[str, torch.Tensor]] = []
        weights: list[float] = []
        for node in participants:
            states.append(
                local_train_shared(
                    model,
                    global_state,
                    candidate,
                    node,
                    data[node].train_x,
                    data[node].train_y,
                    device,
                    args.local_epochs,
                    args.batch_size,
                    args.lr,
                    args.weight_decay,
                )
            )
            weights.append(float(client_sizes[node]))
        global_state = average_states(states, weights)
        load_shared_state(model, candidate, global_state, device)
        rows, macro, _thresholds = evaluate_candidate(model, data, device, run_id, round_idx)
        metric_rows.extend(rows)
        round_rows.append(
            {
                "run_id": run_id,
                "candidate": candidate,
                "schedule": schedule,
                "reset_shared": reset_shared,
                "round": round_idx,
                "participants": "+".join(participants),
                "participant_count": len(participants),
                **macro,
            }
        )
    return metric_rows, round_rows


def tune_fusion_thresholds(models: dict[str, nn.Module], data, quality_stats, device: torch.device, seed: int) -> dict[str, dict[str, float]]:
    thresholds: dict[str, dict[str, float]] = {}
    for candidate, model in models.items():
        probs = {node: predict_probs(model, node, data[node].val_x, device) for node in NODES}
        labels = {node: data[node].val_y for node in NODES}
        quality = {node: quality_weights_from_scaled_embeddings(data[node].val_x, quality_stats[node]) for node in NODES}
        thresholds[candidate] = {}
        for fusion_set, nodes in {"vsn_asn": ["vsn", "asn"], "vsn_asn_vbn": NODES}.items():
            fused, y = label_aligned_fusion(probs, labels, nodes, seed, quality)
            thresholds[candidate][fusion_set] = best_threshold(fused, y)
    return thresholds


def evaluate_robustness(
    args: argparse.Namespace,
    models: dict[str, nn.Module],
    data,
    metadata,
    device: torch.device,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Evaluate frozen candidates under matched node-degradation scenarios."""
    quality_stats = embedding_quality_stats(data)
    fusion_thresholds = tune_fusion_thresholds(models, data, quality_stats, device, args.seed + 300)
    scenarios = build_scenarios()
    per_node_rows: list[dict[str, object]] = []
    fusion_rows: list[dict[str, object]] = []
    degraded_cache = args.output_dir / "embedding_cache"

    for scenario in scenarios:
        scenario_id = str(scenario["scenario"])
        vsn_condition = str(scenario["vsn_condition"])
        asn_condition = str(scenario["asn_condition"])
        vsn_severity = int(scenario["vsn_severity"])
        asn_severity = int(scenario["asn_severity"])
        if vsn_condition == "clean":
            vsn_x = data["vsn"].test_x
            vsn_y = data["vsn"].test_y
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
            vsn_x = data["vsn"].scaler.transform(raw).astype(np.float32)
        if asn_condition == "clean":
            asn_x = data["asn"].test_x
            asn_y = data["asn"].test_y
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
            asn_x = data["asn"].scaler.transform(raw).astype(np.float32)

        scenario_x = {"vsn": vsn_x, "asn": asn_x, "vbn": data["vbn"].test_x}
        scenario_y = {"vsn": vsn_y, "asn": asn_y, "vbn": data["vbn"].test_y}
        # Embedding-distance quality is an experimental proxy used consistently
        # across candidates; it is not a replacement for physical node status.
        scenario_quality = {node: quality_weights_from_scaled_embeddings(scenario_x[node], quality_stats[node]) for node in NODES}

        for candidate, model in models.items():
            probs = {node: predict_probs(model, node, scenario_x[node], device) for node in NODES}
            node_thresholds = {
                node: best_threshold(predict_probs(model, node, data[node].val_x, device), data[node].val_y)
                for node in NODES
            }
            for node in NODES:
                row = metric_row(scenario_y[node], probs[node], node_thresholds[node])
                per_node_rows.append(
                    {
                        "scenario": scenario_id,
                        "candidate": candidate,
                        "node": node,
                        "vsn_condition": vsn_condition,
                        "vsn_severity": vsn_severity,
                        "asn_condition": asn_condition,
                        "asn_severity": asn_severity,
                        **row,
                    }
                )
            for fusion_set, nodes in {"vsn_asn": ["vsn", "asn"], "vsn_asn_vbn": NODES}.items():
                fused, y = label_aligned_fusion(probs, scenario_y, nodes, args.seed + 400, scenario_quality)
                row = metric_row(y, fused, fusion_thresholds[candidate][fusion_set])
                fusion_rows.append(
                    {
                        "scenario": scenario_id,
                        "candidate": candidate,
                        "fusion_set": fusion_set,
                        "fusion_rule": "status_aware_embedding_ood",
                        "nodes": "+".join(nodes),
                        "aligned_sample_count": int(len(y)),
                        "vsn_condition": vsn_condition,
                        "vsn_severity": vsn_severity,
                        "asn_condition": asn_condition,
                        "asn_severity": asn_severity,
                        **row,
                    }
                )
    return per_node_rows, fusion_rows


def plot_robustness(rows: list[dict[str, object]], path: Path) -> None:
    rows = [row for row in rows if row["fusion_set"] == "vsn_asn"]
    scenarios = list(dict.fromkeys([str(row["scenario"]) for row in rows]))
    candidates = list(CANDIDATES.keys())
    x = np.arange(len(scenarios))
    width = 0.36
    fig, ax = plt.subplots(figsize=(13.5, 5.2), dpi=140)
    for offset, candidate in zip([-width / 2, width / 2], candidates):
        values = []
        for scenario in scenarios:
            match = [row for row in rows if row["scenario"] == scenario and row["candidate"] == candidate]
            values.append(float(match[0]["f1_positive"]) if match else float("nan"))
        ax.bar(x + offset, values, width=width, label=candidate)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("VSN+ASN fused F1")
    ax.set_xticks(x, scenarios, rotation=28, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_partial(round_rows: list[dict[str, object]], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5), dpi=140)
    for run_id in sorted({str(row["run_id"]) for row in round_rows}):
        rows = sorted([row for row in round_rows if row["run_id"] == run_id], key=lambda row: int(row["round"]))
        axes[0].plot([int(row["round"]) for row in rows], [float(row["macro_f1"]) for row in rows], marker="o", label=run_id)
        axes[1].plot([int(row["round"]) for row in rows], [float(row["macro_recall"]) for row in rows], marker="o", label=run_id)
    axes[0].set_title("Partial Participation Macro F1")
    axes[1].set_title("Partial Participation Macro Recall")
    for ax in axes:
        ax.set_xlabel("Federated round")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.25)
    axes[0].legend(fontsize=6)
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


def write_comparison_report(
    path: Path,
    model_rows: list[dict[str, object]],
    robustness_rows: list[dict[str, object]],
    partial_rows: list[dict[str, object]],
    rounds: int,
    elapsed: float,
) -> None:
    robustness_summary = []
    for candidate in CANDIDATES:
        rows = [row for row in robustness_rows if row["candidate"] == candidate and row["fusion_set"] == "vsn_asn"]
        robustness_summary.append(
            {
                "candidate": candidate,
                "mean_vsn_asn_f1": float(np.mean([float(row["f1_positive"]) for row in rows])),
                "worst_vsn_asn_f1": float(np.min([float(row["f1_positive"]) for row in rows])),
                "mean_vsn_asn_recall": float(np.mean([float(row["recall_positive"]) for row in rows])),
            }
        )
    final_partial = [row for row in partial_rows if int(row["round"]) == rounds]
    partial_summary = [
        {
            "run_id": row["run_id"],
            "candidate": row["candidate"],
            "update_mode": "scratch" if str(row["reset_shared"]) == "True" or row["reset_shared"] is True else "continual",
            "schedule": row["schedule"],
            "macro_f1": row["macro_f1"],
            "macro_recall": row["macro_recall"],
        }
        for row in final_partial
    ]
    lines = [
        "# Final Sharing Candidate Comparison",
        "",
        "## Purpose",
        "",
        "This report compares the two final model-sharing candidates: the best-performance shared trunk with node-specific calibrators, and the best-lightweight shared projection with shared risk head.",
        "",
        "## Candidate Definitions",
        "",
        markdown_table(model_rows, ["candidate", "method", "role", "total_parameters", "shared_parameters", "fp32_mb", "int8_est_mb", "shared_payload_fp32_kb", "shared_payload_int8_kb_est"]),
        "",
        "## Robustness Summary",
        "",
        markdown_table(robustness_summary, ["candidate", "mean_vsn_asn_f1", "worst_vsn_asn_f1", "mean_vsn_asn_recall"]),
        "",
        "## Partial-Participation Update Summary",
        "",
        markdown_table(partial_summary, ["run_id", "candidate", "update_mode", "schedule", "macro_f1", "macro_recall"]),
        "",
        "## Final Recommendation",
        "",
        "- Use `best_performance_shared_trunk_calibrator` as the main system candidate when accuracy/robustness is prioritized.",
        "- Keep `best_lightweight_shared_projection_head` as the TinyML-oriented fallback because it has fewer total high-layer parameters.",
        "- Note the trade-off: the lightweight candidate has fewer total parameters, but the performance candidate has a much smaller communicated shared module.",
        "- For federated update, interpret scratch and continual modes separately: the shared-trunk candidate is strongest as a trained model maintained through continual shared-trunk updates, while the lightweight shared-projection candidate is more suitable if the shared layers must be relearned from scratch.",
        "",
        "## Cautious Wording",
        "",
        "- These are software simulations using cached embeddings, not physical distributed inference on hardware.",
        "- Low-level VSN/ASN/VBN encoders remain modality-specific and are not shared.",
        "- VBN remains proxy-based, so three-node conclusions should be framed as stress-test evidence.",
        f"",
        f"Runtime: {elapsed:.1f} seconds.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_consolidation_report(
    path: Path,
    clean_rows: list[dict[str, object]],
    model_rows: list[dict[str, object]],
    robustness_rows: list[dict[str, object]],
    partial_rows: list[dict[str, object]],
    rounds: int,
) -> None:
    clean_summary = []
    for candidate, method in CANDIDATES.items():
        row = next(item for item in clean_rows if item["method"] == method)
        size = next(item for item in model_rows if item["candidate"] == candidate)
        clean_summary.append(
            {
                "candidate": candidate,
                "clean_macro_f1": row["macro_f1_positive"],
                "clean_macro_recall": row["macro_recall_positive"],
                "total_params": size["total_parameters"],
                "shared_params": size["shared_parameters"],
            }
        )
    robustness_summary = []
    for candidate in CANDIDATES:
        rows = [row for row in robustness_rows if row["candidate"] == candidate and row["fusion_set"] == "vsn_asn"]
        robustness_summary.append(
            {
                "candidate": candidate,
                "mean_robust_f1": float(np.mean([float(row["f1_positive"]) for row in rows])),
                "worst_robust_f1": float(np.min([float(row["f1_positive"]) for row in rows])),
            }
        )
    partial_summary = [
        {
            "candidate": row["candidate"],
            "update_mode": "scratch" if str(row["reset_shared"]) == "True" or row["reset_shared"] is True else "continual",
            "schedule": row["schedule"],
            "macro_f1": row["macro_f1"],
            "macro_recall": row["macro_recall"],
        }
        for row in partial_rows
        if int(row["round"]) == rounds and row["schedule"] in {"full_participation", "rotating_missing_one", "sparse_one_node"}
    ]
    lines = [
        "# Model Sharing Consolidation and Final Candidate Selection",
        "",
        "## Current Model-Sharing Evidence",
        "",
        "The project has moved from independent node classifiers and score-level fusion to a model-sharing Node Learning design. VSN, ASN, and VBN keep modality-specific lower representations, while higher risk-representation layers can be shared and updated by node-local data.",
        "",
        "## Final Candidate Shortlist",
        "",
        markdown_table(clean_summary, ["candidate", "clean_macro_f1", "clean_macro_recall", "total_params", "shared_params"]),
        "",
        "## Robustness Evidence",
        "",
        markdown_table(robustness_summary, ["candidate", "mean_robust_f1", "worst_robust_f1"]),
        "",
        "## Partial-Participation Evidence",
        "",
        markdown_table(partial_summary, ["candidate", "update_mode", "schedule", "macro_f1", "macro_recall"]),
        "",
        "## Selected Main Design",
        "",
        "The recommended main design is `best_performance_shared_trunk_calibrator`: modality-specific encoders/projections, a shared higher risk trunk, and tiny node-specific calibrators. This gives the best clean macro F1 in the layer sweep, keeps the communicated shared module very small, and remains stable when maintained as a continual shared-trunk update.",
        "",
        "## Lightweight Fallback",
        "",
        "`best_lightweight_shared_projection_head` should be retained as a TinyML fallback. It has fewer total high-layer parameters and still performs close to the main candidate. It also learns better than the shared-trunk candidate when shared layers are reset and trained from scratch, but its shared communication payload is larger than the shared-trunk candidate.",
        "",
        "## Dissertation Logic Chain",
        "",
        "1. Independent local node models establish local risk scoring.",
        "2. Score-level fusion establishes multimodal risk aggregation.",
        "3. Parameter-sharing experiments show that higher risk layers can be shared across heterogeneous nodes.",
        "4. Candidate-layer sweep identifies which layers are worth sharing.",
        "5. Robustness testing shows why confidence alone is insufficient and why status-aware fusion is needed.",
        "6. Federated/partial-participation tests show that shared higher layers can be updated without raw data exchange.",
        "",
        "## Claims to Avoid",
        "",
        "- Do not claim raw modality encoders are shared.",
        "- Do not claim real wireless hardware communication yet.",
        "- Do not claim real collapse probability.",
        "- Do not overstate VBN beyond proxy structural time-series validation.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def read_csv_rows(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    """Run robustness and partial-participation checks for retained candidates."""
    parser = argparse.ArgumentParser(description="Compare final model-sharing candidates and generate consolidation reports.")
    parser.add_argument("--vsn-index", type=Path, default=DEFAULT_VSN_INDEX)
    parser.add_argument("--asn-index", type=Path, default=DEFAULT_ASN_INDEX)
    parser.add_argument("--vbn-features", type=Path, default=DEFAULT_VBN_FEATURES)
    parser.add_argument("--vsn-checkpoint", type=Path, default=DEFAULT_VSN_CHECKPOINT)
    parser.add_argument("--asn-checkpoint", type=Path, default=DEFAULT_ASN_CHECKPOINT)
    parser.add_argument("--candidate-dir", type=Path, default=PROJECT_ROOT / "outputs" / "sharing_candidate_layers")
    parser.add_argument("--base-output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "parameter_sharing_cnn_embeddings")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "final_sharing_candidate_selection")
    parser.add_argument("--vsn-train-limit", type=int, default=3000)
    parser.add_argument("--vsn-eval-limit", type=int, default=1000)
    parser.add_argument("--asn-train-limit", type=int, default=1600)
    parser.add_argument("--asn-eval-limit", type=int, default=500)
    parser.add_argument("--extract-batch-size", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--projection-hidden", type=int, default=96)
    parser.add_argument("--head-hidden", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=7e-4)
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
    input_dims = {name: node_data.train_x.shape[1] for name, node_data in data.items()}
    models = load_candidate_models(args, input_dims, device)

    model_rows: list[dict[str, object]] = []
    for candidate, method in CANDIDATES.items():
        model = models[candidate]
        total = parameter_count(model)
        shared = shared_parameter_count(model, candidate)
        model_rows.append(
            {
                "candidate": candidate,
                "method": method,
                "role": "best-performance" if "performance" in candidate else "best-lightweight",
                "total_parameters": total,
                "shared_parameters": shared,
                "fp32_mb": total * 4 / (1024 * 1024),
                "int8_est_mb": total / (1024 * 1024),
                "shared_payload_fp32_kb": shared * 4 / 1024,
                "shared_payload_int8_kb_est": shared / 1024,
            }
        )

    print("[stage] robustness comparison")
    robustness_per_node, robustness_fusion = evaluate_robustness(args, models, data, metadata, device)

    print("[stage] partial participation comparison")
    partial_metric_rows: list[dict[str, object]] = []
    partial_round_rows: list[dict[str, object]] = []
    for candidate, model in models.items():
        for schedule, reset_shared in [
            ("full_participation", True),
            ("rotating_missing_one", True),
            ("sparse_one_node", True),
            ("rotating_missing_one", False),
        ]:
            metric_rows, round_rows = run_partial_update(candidate, model, data, schedule, reset_shared, device, args)
            partial_metric_rows.extend(metric_rows)
            partial_round_rows.extend(round_rows)

    elapsed = time.perf_counter() - start
    write_csv(args.output_dir / "candidate_model_sizes.csv", model_rows)
    write_csv(args.output_dir / "robustness_per_node_metrics.csv", robustness_per_node)
    write_csv(args.output_dir / "robustness_fusion_metrics.csv", robustness_fusion)
    write_csv(args.output_dir / "partial_participation_per_node_metrics.csv", partial_metric_rows)
    write_csv(args.output_dir / "partial_participation_round_metrics.csv", partial_round_rows)
    plot_robustness(robustness_fusion, args.output_dir / "final_candidate_robustness_vsn_asn.png")
    plot_partial(partial_round_rows, args.output_dir / "final_candidate_partial_participation.png")

    clean_rows = [
        row
        for row in read_csv_rows(args.candidate_dir / "metrics.csv")
        if row["split"] == "test" and row["modality"] == "macro"
    ]
    comparison_report = args.output_dir / "FINAL_SHARING_CANDIDATE_COMPARISON.md"
    consolidation_report = args.output_dir / "MODEL_SHARING_CONSOLIDATION_AND_SELECTION.md"
    write_comparison_report(comparison_report, model_rows, robustness_fusion, partial_round_rows, args.rounds, elapsed)
    write_consolidation_report(consolidation_report, clean_rows, model_rows, robustness_fusion, partial_round_rows, args.rounds)

    save_json(
        args.output_dir / "summary.json",
        {
            "task": "final_sharing_candidate_selection",
            "metadata": metadata,
            "model_sizes": model_rows,
            "robustness_fusion_metrics": robustness_fusion,
            "partial_participation_round_metrics": partial_round_rows,
            "elapsed_seconds": elapsed,
        },
    )

    final_package = PROJECT_ROOT / "outputs" / "reports" / "final_results_package"
    if final_package.exists():
        for path in [
            comparison_report,
            consolidation_report,
            args.output_dir / "candidate_model_sizes.csv",
            args.output_dir / "robustness_fusion_metrics.csv",
            args.output_dir / "partial_participation_round_metrics.csv",
        ]:
            shutil.copy2(path, final_package / path.name)
    print(f"Saved reports to {args.output_dir}")


if __name__ == "__main__":
    main()
