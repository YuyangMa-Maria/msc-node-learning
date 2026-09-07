"""Compare global and lightly personalised shared risk heads."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import log_loss
from scipy.stats import t as student_t
from torch import nn

from evaluate_parameter_sharing_cnn_embeddings import (
    DEFAULT_ASN_CHECKPOINT,
    DEFAULT_ASN_INDEX,
    DEFAULT_VBN_FEATURES,
    DEFAULT_VSN_CHECKPOINT,
    DEFAULT_VSN_INDEX,
    PROJECT_ROOT,
    SharedRiskHeadModel,
    SharedTrunkNodeCalibratedModel,
    best_threshold,
    make_loader,
    metric_row,
    predict_probs,
)
from evaluate_shared_head_federated_dropout import participants_for_schedule
from evaluate_shared_head_federated_update import (
    average_states,
    build_loss,
    load_clean_modality_data,
    load_reference_model,
    set_seed,
)


NODES = ["vsn", "asn", "vbn"]
PRIMARY_NODES = ["vsn", "asn"]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def calibration_metrics(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> dict[str, float]:
    labels = labels.astype(np.float64)
    probabilities = np.clip(probabilities.astype(np.float64), 1e-7, 1 - 1e-7)
    brier = float(np.mean((probabilities - labels) ** 2))
    nll = float(log_loss(labels, probabilities, labels=[0, 1]))
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for idx in range(bins):
        lower, upper = edges[idx], edges[idx + 1]
        mask = (probabilities > lower) & (probabilities <= upper)
        if np.any(mask):
            ece += float(np.mean(mask)) * abs(float(np.mean(probabilities[mask])) - float(np.mean(labels[mask])))
    return {"brier": brier, "nll": nll, "ece": ece}


def summarize_metrics(rows: list[dict[str, object]]) -> dict[str, float]:
    all_rows = [row for row in rows if row["split"] == "test"]
    primary_rows = [row for row in all_rows if row["modality"] in PRIMARY_NODES]

    def mean(key: str, selected: list[dict[str, object]]) -> float:
        return float(np.mean([float(row[key]) for row in selected]))

    return {
        "macro_f1_all": mean("f1_positive", all_rows),
        "macro_recall_all": mean("recall_positive", all_rows),
        "macro_auc_all": mean("roc_auc", all_rows),
        "macro_brier_all": mean("brier", all_rows),
        "macro_nll_all": mean("nll", all_rows),
        "macro_ece_all": mean("ece", all_rows),
        "macro_f1_primary": mean("f1_positive", primary_rows),
        "macro_recall_primary": mean("recall_positive", primary_rows),
        "macro_auc_primary": mean("roc_auc", primary_rows),
        "macro_brier_primary": mean("brier", primary_rows),
        "macro_nll_primary": mean("nll", primary_rows),
        "macro_ece_primary": mean("ece", primary_rows),
    }


def evaluate(
    model: nn.Module,
    data,
    device: torch.device,
    strategy: str,
    mode: str,
    schedule: str,
    algorithm_seed: int,
    round_idx: int,
) -> tuple[list[dict[str, object]], dict[str, float]]:
    thresholds: dict[str, float] = {}
    for node, node_data in data.items():
        val_probs = predict_probs(model, node, node_data.val_x, device)
        thresholds[node] = best_threshold(val_probs, node_data.val_y)

    rows: list[dict[str, object]] = []
    for split in ["val", "test"]:
        for node, node_data in data.items():
            features = getattr(node_data, f"{split}_x")
            labels = getattr(node_data, f"{split}_y")
            probabilities = predict_probs(model, node, features, device)
            classification = metric_row(labels, probabilities, thresholds[node])
            calibration = calibration_metrics(labels, probabilities)
            rows.append(
                {
                    "strategy": strategy,
                    "mode": mode,
                    "schedule": schedule,
                    "algorithm_seed": algorithm_seed,
                    "round": round_idx,
                    "split": split,
                    "modality": node,
                    **classification,
                    **calibration,
                }
            )
    return rows, summarize_metrics(rows)


def freeze_global_head_model(model: SharedRiskHeadModel) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.shared_head.parameters():
        parameter.requires_grad = True


def freeze_personalized_model(model: SharedTrunkNodeCalibratedModel, node: str | None = None) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.shared_trunk.parameters():
        parameter.requires_grad = True
    if node is not None:
        for parameter in model.node_calibrators[node].parameters():
            parameter.requires_grad = True


def module_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def load_module_state(module: nn.Module, state: dict[str, torch.Tensor], device: torch.device) -> None:
    module.load_state_dict({name: value.to(device) for name, value in state.items()})


def reset_global_head(model: SharedRiskHeadModel, seed: int) -> None:
    torch.manual_seed(seed)
    for module in model.shared_head.modules():
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()


def reset_personalized_modules(model: SharedTrunkNodeCalibratedModel, seed: int) -> None:
    torch.manual_seed(seed)
    for module in model.shared_trunk.modules():
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()
    torch.manual_seed(seed + 1)
    model.node_calibrators[NODES[0]].reset_parameters()
    template = module_state(model.node_calibrators[NODES[0]])
    for node in NODES[1:]:
        model.node_calibrators[node].load_state_dict(template)


def build_personalized_template(
    reference_model: SharedRiskHeadModel,
    input_dims: dict[str, int],
    saved_args: dict[str, object],
    device: torch.device,
) -> SharedTrunkNodeCalibratedModel:
    model = SharedTrunkNodeCalibratedModel(
        input_dims=input_dims,
        embedding_dim=int(saved_args.get("embedding_dim", 64)),
        projection_hidden=int(saved_args.get("projection_hidden", 96)),
        head_hidden=int(saved_args.get("head_hidden", 32)),
        dropout=float(saved_args.get("dropout", 0.1)),
    )
    model.projections.load_state_dict(reference_model.projections.state_dict())
    model.shared_trunk[0].load_state_dict(reference_model.shared_head[0].state_dict())
    final_state = reference_model.shared_head[3].state_dict()
    for node in NODES:
        model.node_calibrators[node].load_state_dict(final_state)
    return model.to(device)


def train_global_client(
    base_model: SharedRiskHeadModel,
    global_state: dict[str, torch.Tensor],
    node: str,
    node_data,
    device: torch.device,
    local_epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    local_seed: int,
) -> dict[str, torch.Tensor]:
    set_seed(local_seed)
    model = copy.deepcopy(base_model).to(device)
    load_module_state(model.shared_head, global_state, device)
    freeze_global_head_model(model)
    model.train()
    loader = make_loader(node_data.train_x, node_data.train_y, batch_size, shuffle=True)
    criterion = build_loss(node_data.train_y, device)
    optimizer = torch.optim.AdamW(model.shared_head.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(local_epochs):
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(node, features), labels)
            loss.backward()
            optimizer.step()
    return module_state(model.shared_head)


def train_personalized_client(
    base_model: SharedTrunkNodeCalibratedModel,
    trunk_state: dict[str, torch.Tensor],
    local_head_state: dict[str, torch.Tensor],
    node: str,
    node_data,
    device: torch.device,
    local_epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    local_seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    set_seed(local_seed)
    model = copy.deepcopy(base_model).to(device)
    load_module_state(model.shared_trunk, trunk_state, device)
    load_module_state(model.node_calibrators[node], local_head_state, device)
    freeze_personalized_model(model, node)
    model.train()
    loader = make_loader(node_data.train_x, node_data.train_y, batch_size, shuffle=True)
    criterion = build_loss(node_data.train_y, device)
    parameters = list(model.shared_trunk.parameters()) + list(model.node_calibrators[node].parameters())
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    for _ in range(local_epochs):
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(node, features), labels)
            loss.backward()
            optimizer.step()
    return module_state(model.shared_trunk), module_state(model.node_calibrators[node])


def run_global_shared(
    reference_model: SharedRiskHeadModel,
    data,
    device: torch.device,
    mode: str,
    schedule: str,
    algorithm_seed: int,
    rounds: int,
    local_epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    model = copy.deepcopy(reference_model).to(device)
    if mode == "scratch":
        reset_global_head(model, algorithm_seed + 1000)
    freeze_global_head_model(model)
    global_state = module_state(model.shared_head)
    client_sizes = {node: len(data[node].train_y) for node in NODES}
    metric_rows: list[dict[str, object]] = []
    round_rows: list[dict[str, object]] = []

    rows, summary = evaluate(model, data, device, "global_shared_head", mode, schedule, algorithm_seed, 0)
    metric_rows.extend(rows)
    round_rows.append(
        {
            "strategy": "global_shared_head",
            "mode": mode,
            "schedule": schedule,
            "algorithm_seed": algorithm_seed,
            "round": 0,
            "participants": "initial",
            **summary,
        }
    )
    for round_idx in range(1, rounds + 1):
        participants = participants_for_schedule(schedule, round_idx)
        states: list[dict[str, torch.Tensor]] = []
        weights: list[float] = []
        for node_idx, node in enumerate(participants):
            local_seed = algorithm_seed * 10_000 + round_idx * 100 + node_idx
            states.append(
                train_global_client(
                    model,
                    global_state,
                    node,
                    data[node],
                    device,
                    local_epochs,
                    batch_size,
                    lr,
                    weight_decay,
                    local_seed,
                )
            )
            weights.append(float(client_sizes[node]))
        global_state = average_states(states, weights)
        load_module_state(model.shared_head, global_state, device)
        rows, summary = evaluate(model, data, device, "global_shared_head", mode, schedule, algorithm_seed, round_idx)
        metric_rows.extend(rows)
        round_rows.append(
            {
                "strategy": "global_shared_head",
                "mode": mode,
                "schedule": schedule,
                "algorithm_seed": algorithm_seed,
                "round": round_idx,
                "participants": "+".join(participants),
                **summary,
            }
        )
    return metric_rows, round_rows


def run_personalized(
    template: SharedTrunkNodeCalibratedModel,
    data,
    device: torch.device,
    mode: str,
    schedule: str,
    algorithm_seed: int,
    rounds: int,
    local_epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    model = copy.deepcopy(template).to(device)
    if mode == "scratch":
        reset_personalized_modules(model, algorithm_seed + 2000)
    freeze_personalized_model(model)
    trunk_state = module_state(model.shared_trunk)
    local_heads = {node: module_state(model.node_calibrators[node]) for node in NODES}
    client_sizes = {node: len(data[node].train_y) for node in NODES}
    metric_rows: list[dict[str, object]] = []
    round_rows: list[dict[str, object]] = []

    rows, summary = evaluate(model, data, device, "personalized_shared_trunk", mode, schedule, algorithm_seed, 0)
    metric_rows.extend(rows)
    round_rows.append(
        {
            "strategy": "personalized_shared_trunk",
            "mode": mode,
            "schedule": schedule,
            "algorithm_seed": algorithm_seed,
            "round": 0,
            "participants": "initial",
            **summary,
        }
    )
    for round_idx in range(1, rounds + 1):
        participants = participants_for_schedule(schedule, round_idx)
        trunk_states: list[dict[str, torch.Tensor]] = []
        weights: list[float] = []
        for node_idx, node in enumerate(participants):
            local_seed = algorithm_seed * 10_000 + round_idx * 100 + node_idx
            trained_trunk, trained_head = train_personalized_client(
                model,
                trunk_state,
                local_heads[node],
                node,
                data[node],
                device,
                local_epochs,
                batch_size,
                lr,
                weight_decay,
                local_seed,
            )
            trunk_states.append(trained_trunk)
            local_heads[node] = trained_head
            weights.append(float(client_sizes[node]))
        trunk_state = average_states(trunk_states, weights)
        load_module_state(model.shared_trunk, trunk_state, device)
        for node in NODES:
            load_module_state(model.node_calibrators[node], local_heads[node], device)
        rows, summary = evaluate(
            model,
            data,
            device,
            "personalized_shared_trunk",
            mode,
            schedule,
            algorithm_seed,
            round_idx,
        )
        metric_rows.extend(rows)
        round_rows.append(
            {
                "strategy": "personalized_shared_trunk",
                "mode": mode,
                "schedule": schedule,
                "algorithm_seed": algorithm_seed,
                "round": round_idx,
                "participants": "+".join(participants),
                **summary,
            }
        )
    return metric_rows, round_rows


def confidence_interval(values: list[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1)) if len(array) > 1 else 0.0
    critical = float(student_t.ppf(0.975, df=len(array) - 1)) if len(array) > 1 else 0.0
    half_width = critical * std / math.sqrt(len(array)) if len(array) > 1 else 0.0
    return mean, std, half_width


def aggregate_final_rows(
    round_rows: list[dict[str, object]],
    rounds: int,
    seeds: list[int],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    final_rows = [row for row in round_rows if int(row["round"]) == rounds]
    summary_rows: list[dict[str, object]] = []
    for strategy in sorted({str(row["strategy"]) for row in final_rows}):
        for mode in sorted({str(row["mode"]) for row in final_rows}):
            for schedule in sorted({str(row["schedule"]) for row in final_rows}):
                selected = [
                    row
                    for row in final_rows
                    if row["strategy"] == strategy and row["mode"] == mode and row["schedule"] == schedule
                ]
                if not selected:
                    continue
                result: dict[str, object] = {
                    "strategy": strategy,
                    "mode": mode,
                    "schedule": schedule,
                    "n_seeds": len(selected),
                }
                for metric in [
                    "macro_f1_primary",
                    "macro_auc_primary",
                    "macro_brier_primary",
                    "macro_nll_primary",
                    "macro_ece_primary",
                    "macro_f1_all",
                ]:
                    mean, std, half_width = confidence_interval([float(row[metric]) for row in selected])
                    result[f"{metric}_mean"] = mean
                    result[f"{metric}_std"] = std
                    result[f"{metric}_ci95_half"] = half_width
                summary_rows.append(result)

    paired_rows: list[dict[str, object]] = []
    for mode in ["scratch", "continual"]:
        for schedule in ["full_participation", "rotating_missing_one", "sparse_one_node"]:
            deltas: list[float] = []
            for seed in seeds:
                global_row = next(
                    row
                    for row in final_rows
                    if row["strategy"] == "global_shared_head"
                    and row["mode"] == mode
                    and row["schedule"] == schedule
                    and int(row["algorithm_seed"]) == seed
                )
                personalized_row = next(
                    row
                    for row in final_rows
                    if row["strategy"] == "personalized_shared_trunk"
                    and row["mode"] == mode
                    and row["schedule"] == schedule
                    and int(row["algorithm_seed"]) == seed
                )
                deltas.append(float(personalized_row["macro_f1_primary"]) - float(global_row["macro_f1_primary"]))
            mean, std, half_width = confidence_interval(deltas)
            paired_rows.append(
                {
                    "mode": mode,
                    "schedule": schedule,
                    "metric": "macro_f1_primary",
                    "personalized_minus_global_mean": mean,
                    "std": std,
                    "ci95_low": mean - half_width,
                    "ci95_high": mean + half_width,
                    "n_seeds": len(deltas),
                }
            )
    return summary_rows, paired_rows


def participant_uploads(schedule: str, rounds: int) -> int:
    return sum(len(participants_for_schedule(schedule, round_idx)) for round_idx in range(1, rounds + 1))


def plot_summary(summary_rows: list[dict[str, object]], output_path: Path) -> None:
    selected = [row for row in summary_rows if row["mode"] == "scratch"]
    labels = [f"{row['strategy']}\n{row['schedule']}" for row in selected]
    means = [float(row["macro_f1_primary_mean"]) for row in selected]
    errors = [float(row["macro_f1_primary_ci95_half"]) for row in selected]
    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=140)
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=errors, capsize=4)
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylabel("VSN+ASN macro F1")
    ax.set_ylim(max(0.0, min(means) - 0.05), 1.01)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def markdown_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        values: list[str] = []
        for column in columns:
            value = row.get(column, "")
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    output_path: Path,
    summary_rows: list[dict[str, object]],
    paired_rows: list[dict[str, object]],
    shared_head_params: int,
    shared_trunk_params: int,
    local_head_params: int,
    rounds: int,
    seeds: list[int],
    elapsed: float,
) -> None:
    focus = [
        {
            "strategy": row["strategy"],
            "mode": row["mode"],
            "schedule": row["schedule"],
            "VSN+ASN F1": float(row["macro_f1_primary_mean"]),
            "F1 std": float(row["macro_f1_primary_std"]),
            "AUROC": float(row["macro_auc_primary_mean"]),
            "Brier": float(row["macro_brier_primary_mean"]),
            "ECE": float(row["macro_ece_primary_mean"]),
            "all-node F1": float(row["macro_f1_all_mean"]),
        }
        for row in summary_rows
    ]
    communication_rows = []
    for schedule in ["full_participation", "rotating_missing_one", "sparse_one_node"]:
        uploads = participant_uploads(schedule, rounds)
        communication_rows.append(
            {
                "schedule": schedule,
                "uploads": uploads,
                "global FP32 KB": uploads * shared_head_params * 4 / 1024,
                "personalized FP32 KB": uploads * shared_trunk_params * 4 / 1024,
                "global INT8 KB est": uploads * shared_head_params / 1024,
                "personalized INT8 KB est": uploads * shared_trunk_params / 1024,
            }
        )

    lines = [
        "# Grouped Personalized Federated Parameter-Sharing Experiment",
        "",
        "## Research Question",
        "",
        "Should heterogeneous sensing nodes federate one fully global higher risk head, or share only a common trunk while retaining a small node-specific output layer?",
        "",
        "## Controlled Design",
        "",
        "- Leakage-resistant grouped VSN/ASN splits and the fixed grouped embedding cache are used for every run.",
        "- Both strategies use identical frozen modality-specific encoders and projections.",
        "- `global_shared_head` communicates the complete `64 -> 32 -> 1` higher risk head.",
        "- `personalized_shared_trunk` communicates only `64 -> 32`; each node retains a private `32 -> 1` calibrator.",
        f"- Algorithm seeds: {', '.join(str(seed) for seed in seeds)}.",
        f"- Federated rounds: {rounds}; thresholds are selected on validation data and applied once to the corresponding test evaluation.",
        "- Primary inference metric: VSN+ASN macro performance. The all-node macro is secondary because VBN is a small proxy dataset.",
        "",
        "## Final Results",
        "",
        markdown_table(
            focus,
            ["strategy", "mode", "schedule", "VSN+ASN F1", "F1 std", "AUROC", "Brier", "ECE", "all-node F1"],
        ),
        "",
        "## Paired F1 Difference",
        "",
        "`personalized - global`; a 95% interval crossing zero does not support a significant advantage.",
        "",
        markdown_table(
            paired_rows,
            ["mode", "schedule", "personalized_minus_global_mean", "std", "ci95_low", "ci95_high", "n_seeds"],
        ),
        "",
        "## Parameter and Communication Cost",
        "",
        f"- Fully global shared head: {shared_head_params} communicated parameters.",
        f"- Personalized shared trunk: {shared_trunk_params} communicated parameters.",
        f"- Private node calibrator: {local_head_params} parameters per node; never uploaded.",
        "",
        markdown_table(
            communication_rows,
            [
                "schedule",
                "uploads",
                "global FP32 KB",
                "personalized FP32 KB",
                "global INT8 KB est",
                "personalized INT8 KB est",
            ],
        ),
        "",
        "## Interpretation Rules",
        "",
        "- Prefer the personalized design only if it improves calibration or participation robustness without a meaningful F1 loss.",
        "- Do not treat the all-node macro as strong physical evidence because VBN remains an ORION AE proxy with a very small test split.",
        "- This is a cached-embedding software simulation, not real wireless asynchronous federated learning.",
        "- Multiple algorithm seeds quantify optimization variability on one fixed grouped data sample; they do not replace independent datasets.",
        f"",
        f"Runtime: {elapsed:.1f} seconds.",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare global and personalized higher-layer federated sharing.")
    parser.add_argument("--vsn-index", type=Path, default=DEFAULT_VSN_INDEX)
    parser.add_argument("--asn-index", type=Path, default=DEFAULT_ASN_INDEX)
    parser.add_argument("--vbn-features", type=Path, default=DEFAULT_VBN_FEATURES)
    parser.add_argument("--vsn-checkpoint", type=Path, default=DEFAULT_VSN_CHECKPOINT)
    parser.add_argument("--asn-checkpoint", type=Path, default=DEFAULT_ASN_CHECKPOINT)
    parser.add_argument("--base-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "personalized_federated_sharing")
    parser.add_argument("--vsn-train-limit", type=int, default=3000)
    parser.add_argument("--vsn-eval-limit", type=int, default=1000)
    parser.add_argument("--asn-train-limit", type=int, default=1600)
    parser.add_argument("--asn-eval-limit", type=int, default=500)
    parser.add_argument("--extract-batch-size", type=int, default=128)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--algorithm-seeds", type=int, nargs="+", default=[42, 52, 62, 72, 82])
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    start = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    data_args = copy.copy(args)
    data_args.seed = args.data_seed
    data, metadata = load_clean_modality_data(data_args, device)
    input_dims = {node: node_data.train_x.shape[1] for node, node_data in data.items()}
    reference_model, saved_args = load_reference_model(args.base_output_dir / "shared_risk_head.pt", input_dims, device)
    personalized_template = build_personalized_template(reference_model, input_dims, saved_args, device)

    all_metric_rows: list[dict[str, object]] = []
    all_round_rows: list[dict[str, object]] = []
    schedules = ["full_participation", "rotating_missing_one", "sparse_one_node"]
    for seed in args.algorithm_seeds:
        for mode in ["scratch", "continual"]:
            for schedule in schedules:
                print(f"[run] global seed={seed} mode={mode} schedule={schedule}", flush=True)
                metric_rows, round_rows = run_global_shared(
                    reference_model,
                    data,
                    device,
                    mode,
                    schedule,
                    seed,
                    args.rounds,
                    args.local_epochs,
                    args.batch_size,
                    args.lr,
                    args.weight_decay,
                )
                all_metric_rows.extend(metric_rows)
                all_round_rows.extend(round_rows)

                print(f"[run] personalized seed={seed} mode={mode} schedule={schedule}", flush=True)
                metric_rows, round_rows = run_personalized(
                    personalized_template,
                    data,
                    device,
                    mode,
                    schedule,
                    seed,
                    args.rounds,
                    args.local_epochs,
                    args.batch_size,
                    args.lr,
                    args.weight_decay,
                )
                all_metric_rows.extend(metric_rows)
                all_round_rows.extend(round_rows)

    summary_rows, paired_rows = aggregate_final_rows(all_round_rows, args.rounds, args.algorithm_seeds)
    write_csv(args.output_dir / "per_node_metrics.csv", all_metric_rows)
    write_csv(args.output_dir / "round_metrics.csv", all_round_rows)
    write_csv(args.output_dir / "final_summary.csv", summary_rows)
    write_csv(args.output_dir / "paired_differences.csv", paired_rows)
    plot_summary(summary_rows, args.output_dir / "personalized_federated_summary.png")

    shared_head_params = sum(parameter.numel() for parameter in reference_model.shared_head.parameters())
    shared_trunk_params = sum(parameter.numel() for parameter in personalized_template.shared_trunk.parameters())
    local_head_params = sum(parameter.numel() for parameter in personalized_template.node_calibrators["vsn"].parameters())
    elapsed = time.perf_counter() - start
    payload = {
        "task": "grouped_personalized_federated_parameter_sharing",
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "metadata": metadata,
        "parameter_counts": {
            "global_shared_head": shared_head_params,
            "personalized_shared_trunk": shared_trunk_params,
            "private_node_calibrator": local_head_params,
        },
        "final_summary": summary_rows,
        "paired_differences": paired_rows,
        "elapsed_seconds": elapsed,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report_path = args.output_dir / "PERSONALIZED_FEDERATED_SHARING_REPORT.md"
    write_report(
        report_path,
        summary_rows,
        paired_rows,
        shared_head_params,
        shared_trunk_params,
        local_head_params,
        args.rounds,
        args.algorithm_seeds,
        elapsed,
    )
    print(report_path)


if __name__ == "__main__":
    main()
