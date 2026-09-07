"""Simulate round-based updates of only the compatible shared risk head.

Each client receives the same higher-layer state, trains that state on its local
representation and returns only the shared-head tensors. Private projections
remain frozen, so this is partial model collaboration rather than full-model
FedAvg.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
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
    SharedRiskHeadModel,
    best_threshold,
    extract_asn_embeddings,
    extract_vsn_embeddings,
    load_vbn_feature_embeddings,
    make_loader,
    make_modality_data,
    metric_row,
    predict_probs,
    save_json,
)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_clean_modality_data(args: argparse.Namespace, device: torch.device):
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


def load_reference_model(checkpoint_path: Path, input_dims: dict[str, int], device: torch.device) -> tuple[SharedRiskHeadModel, dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    saved_args = checkpoint.get("args", {})
    if not isinstance(saved_args, dict):
        saved_args = vars(saved_args)
    model = SharedRiskHeadModel(
        input_dims=input_dims,
        embedding_dim=int(saved_args.get("embedding_dim", 64)),
        projection_hidden=int(saved_args.get("projection_hidden", 96)),
        head_hidden=int(saved_args.get("head_hidden", 32)),
        dropout=float(saved_args.get("dropout", 0.1)),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, saved_args


def reset_shared_head(model: SharedRiskHeadModel, seed: int) -> None:
    torch.manual_seed(seed)
    for module in model.shared_head.modules():
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()


def freeze_all_but_shared_head(model: SharedRiskHeadModel) -> None:
    """Make the communication boundary explicit in the optimiser state."""
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.shared_head.parameters():
        parameter.requires_grad = True


def shared_head_state(model: SharedRiskHeadModel) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.shared_head.state_dict().items()}


def load_shared_head_state(model: SharedRiskHeadModel, state: dict[str, torch.Tensor], device: torch.device) -> None:
    model.shared_head.load_state_dict({name: value.to(device) for name, value in state.items()})


def average_states(states: list[dict[str, torch.Tensor]], weights: list[float]) -> dict[str, torch.Tensor]:
    """Aggregate compatible tensor dictionaries using normalised client weights."""
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
    pos_weight = torch.tensor(negatives / max(positives, 1.0), dtype=torch.float32, device=device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def local_train_shared_head(
    global_model: SharedRiskHeadModel,
    global_head_state: dict[str, torch.Tensor],
    modality: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    device: torch.device,
    local_epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
) -> dict[str, torch.Tensor]:
    """Run one client's local update without modifying its private projection."""
    model = copy.deepcopy(global_model).to(device)
    load_shared_head_state(model, global_head_state, device)
    freeze_all_but_shared_head(model)
    model.train()
    loader = make_loader(train_x, train_y, batch_size, shuffle=True)
    criterion = build_loss(train_y, device)
    optimizer = torch.optim.AdamW(model.shared_head.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(local_epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(modality, xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
    return shared_head_state(model)


def evaluate_global_model(
    model: SharedRiskHeadModel,
    data,
    device: torch.device,
    run_id: str,
    round_idx: int,
) -> tuple[list[dict[str, object]], dict[str, float], dict[str, float]]:
    thresholds: dict[str, float] = {}
    rows: list[dict[str, object]] = []
    for modality, modality_data in data.items():
        val_probs = predict_probs(model, modality, modality_data.val_x, device)
        thresholds[modality] = best_threshold(val_probs, modality_data.val_y)
    for split in ["val", "test"]:
        for modality, modality_data in data.items():
            x = getattr(modality_data, f"{split}_x")
            y = getattr(modality_data, f"{split}_y")
            probs = predict_probs(model, modality, x, device)
            metric = metric_row(y, probs, thresholds[modality])
            rows.append(
                {
                    "run_id": run_id,
                    "round": round_idx,
                    "split": split,
                    "modality": modality,
                    **metric,
                }
            )
    test_rows = [row for row in rows if row["split"] == "test"]
    macro = {
        "macro_f1": float(np.mean([float(row["f1_positive"]) for row in test_rows])),
        "macro_recall": float(np.mean([float(row["recall_positive"]) for row in test_rows])),
        "macro_accuracy": float(np.mean([float(row["accuracy"]) for row in test_rows])),
    }
    return rows, macro, thresholds


def shared_head_parameter_count(model: SharedRiskHeadModel) -> int:
    return sum(parameter.numel() for parameter in model.shared_head.parameters())


def run_fedavg(
    run_id: str,
    reference_model: SharedRiskHeadModel,
    data,
    device: torch.device,
    rounds: int,
    local_epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    aggregation: str,
    reset_head: bool,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, torch.Tensor]]:
    """Run the full-participation baseline from a fresh or pretrained head."""
    model = copy.deepcopy(reference_model).to(device)
    if reset_head:
        reset_shared_head(model, seed + 900)
    freeze_all_but_shared_head(model)
    global_state = shared_head_state(model)
    all_metric_rows: list[dict[str, object]] = []
    round_rows: list[dict[str, object]] = []
    client_sizes = {name: len(modality_data.train_y) for name, modality_data in data.items()}

    metric_rows, macro, _thresholds = evaluate_global_model(model, data, device, run_id, 0)
    all_metric_rows.extend(metric_rows)
    round_rows.append({"run_id": run_id, "round": 0, "aggregation": aggregation, "reset_head": reset_head, **macro})

    for round_idx in range(1, rounds + 1):
        local_states: list[dict[str, torch.Tensor]] = []
        local_weights: list[float] = []
        for modality, modality_data in data.items():
            state = local_train_shared_head(
                model,
                global_state,
                modality,
                modality_data.train_x,
                modality_data.train_y,
                device,
                local_epochs,
                batch_size,
                lr,
                weight_decay,
            )
            local_states.append(state)
            local_weights.append(float(client_sizes[modality]) if aggregation == "sample_weighted" else 1.0)
        # All returned dictionaries share names and shapes by construction;
        # incompatible private encoders never enter this aggregation.
        global_state = average_states(local_states, local_weights)
        load_shared_head_state(model, global_state, device)
        metric_rows, macro, _thresholds = evaluate_global_model(model, data, device, run_id, round_idx)
        all_metric_rows.extend(metric_rows)
        round_rows.append({"run_id": run_id, "round": round_idx, "aggregation": aggregation, "reset_head": reset_head, **macro})
    return all_metric_rows, round_rows, global_state


def plot_rounds(round_rows: list[dict[str, object]], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=140)
    for run_id in sorted({str(row["run_id"]) for row in round_rows}):
        rows = sorted([row for row in round_rows if row["run_id"] == run_id], key=lambda row: int(row["round"]))
        x = [int(row["round"]) for row in rows]
        axes[0].plot(x, [float(row["macro_f1"]) for row in rows], marker="o", label=run_id)
        axes[1].plot(x, [float(row["macro_recall"]) for row in rows], marker="o", label=run_id)
    axes[0].set_title("Test Macro F1")
    axes[1].set_title("Test Macro Recall")
    for ax in axes:
        ax.set_xlabel("Federated round")
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)
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
    metadata: dict[str, dict[str, object]],
    reference_rows: list[dict[str, object]],
    round_rows: list[dict[str, object]],
    final_metric_rows: list[dict[str, object]],
    shared_params: int,
    rounds: int,
    local_epochs: int,
    elapsed: float,
) -> None:
    final_round_rows = [row for row in round_rows if int(row["round"]) == rounds]
    best_final = max(final_round_rows, key=lambda row: float(row["macro_f1"]))
    reference_test = [row for row in reference_rows if row["split"] == "test"]
    reference_macro = {
        "run_id": "centralized_reference",
        "round": "reference",
        "macro_f1": float(np.mean([float(row["f1_positive"]) for row in reference_test])),
        "macro_recall": float(np.mean([float(row["recall_positive"]) for row in reference_test])),
        "macro_accuracy": float(np.mean([float(row["accuracy"]) for row in reference_test])),
    }
    final_focus = [
        {
            "run_id": row["run_id"],
            "round": row["round"],
            "macro_f1": row["macro_f1"],
            "macro_recall": row["macro_recall"],
            "macro_accuracy": row["macro_accuracy"],
        }
        for row in final_round_rows
    ]
    final_focus.insert(0, reference_macro)
    best_per_node = [
        {
            "modality": row["modality"],
            "f1": row["f1_positive"],
            "recall": row["recall_positive"],
            "accuracy": row["accuracy"],
            "threshold": row["threshold"],
        }
        for row in final_metric_rows
        if row["run_id"] == best_final["run_id"] and row["split"] == "test" and int(row["round"]) == int(best_final["round"])
    ]
    representation_rows = [
        {
            "node": name,
            "representation": meta["source_representation"],
            "dim": meta["embedding_dim"],
            "train/val/test": f"{meta['sample_counts']['train']}/{meta['sample_counts']['val']}/{meta['sample_counts']['test']}",
        }
        for name, meta in metadata.items()
    ]
    fp32_payload = shared_params * 4
    int8_payload = shared_params
    lines = [
        "# Shared-Head Federated / Continual Update Experiment",
        "",
        "## Purpose",
        "",
        "This experiment tests the supervisor's model-communication idea in a lightweight form: nodes keep their local encoders/projections private, and only the shared higher risk head is updated through federated averaging.",
        "",
        "## Setup",
        "",
        "- Local data is not pooled during the federated stage.",
        "- VSN, ASN, and VBN act as three clients with heterogeneous local representations.",
        "- Modality-specific projections are frozen from the existing shared-risk-head model.",
        "- Only the shared higher risk head is trainable and communicated.",
        f"- Federated rounds: {rounds}; local epochs per round: {local_epochs}.",
        "",
        "## Node Representations",
        "",
        markdown_table(representation_rows, ["node", "representation", "dim", "train/val/test"]),
        "",
        "## Final Macro Results",
        "",
        markdown_table(final_focus, ["run_id", "round", "macro_f1", "macro_recall", "macro_accuracy"]),
        "",
        "## Best Federated Run Per-Node Test Results",
        "",
        f"Best final federated run: `{best_final['run_id']}`.",
        "",
        markdown_table(best_per_node, ["modality", "f1", "recall", "accuracy", "threshold"]),
        "",
        "## Communication Size",
        "",
        f"- Shared head parameters: {shared_params}.",
        f"- One shared-head payload: {fp32_payload} bytes FP32 ({fp32_payload / 1024:.2f} KB), or about {int8_payload} bytes INT8 ({int8_payload / 1024:.2f} KB).",
        "- This is much smaller than transmitting raw images/audio or full node models.",
        "",
        "## Interpretation",
        "",
        "- This is a software-level proof that the shared higher risk layer can be updated by node-local data without sharing raw sensor samples.",
        "- The experiment is closer to Node Learning than pure score fusion because it communicates model parameters, not only risk scores.",
        "- It should be framed as a lightweight federated/continual update simulation, not as a full physical deployment.",
        "",
        "## Claims to Avoid / Cautious Wording",
        "",
        "- Do not claim full end-to-end federated learning of all encoders; only the shared head is federated here.",
        "- Do not claim real hardware communication yet; payload size is an estimate from parameter count.",
        "- Do not claim true vibration validation; VBN remains an ORION AE proxy.",
        "- Do not claim collapse probability; the model outputs a structural risk proxy.",
        "",
        "## Recommended Next Step",
        "",
        "- Combine this shared-head update result with robustness/status-aware fusion in the dissertation narrative.",
        "- If more time is available, repeat the same shared-head update under simulated degraded-node participation or node dropout.",
        f"",
        f"Runtime: {elapsed:.1f} seconds.",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Federated/continual update experiment for the shared higher risk head.")
    parser.add_argument("--vsn-index", type=Path, default=DEFAULT_VSN_INDEX)
    parser.add_argument("--asn-index", type=Path, default=DEFAULT_ASN_INDEX)
    parser.add_argument("--vbn-features", type=Path, default=DEFAULT_VBN_FEATURES)
    parser.add_argument("--vsn-checkpoint", type=Path, default=DEFAULT_VSN_CHECKPOINT)
    parser.add_argument("--asn-checkpoint", type=Path, default=DEFAULT_ASN_CHECKPOINT)
    parser.add_argument("--base-output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "parameter_sharing_cnn_embeddings")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "shared_head_federated_update")
    parser.add_argument("--vsn-train-limit", type=int, default=3000)
    parser.add_argument("--vsn-eval-limit", type=int, default=1000)
    parser.add_argument("--asn-train-limit", type=int, default=1600)
    parser.add_argument("--asn-eval-limit", type=int, default=500)
    parser.add_argument("--extract-batch-size", type=int, default=128)
    parser.add_argument("--rounds", type=int, default=8)
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

    data, metadata = load_clean_modality_data(args, device)
    input_dims = {name: modality_data.train_x.shape[1] for name, modality_data in data.items()}
    reference_model, saved_args = load_reference_model(args.base_output_dir / "shared_risk_head.pt", input_dims, device)
    freeze_all_but_shared_head(reference_model)

    reference_rows, reference_macro, reference_thresholds = evaluate_global_model(reference_model, data, device, "centralized_reference", 0)
    round_rows: list[dict[str, object]] = [
        {"run_id": "centralized_reference", "round": 0, "aggregation": "centralized", "reset_head": False, **reference_macro}
    ]
    all_metric_rows: list[dict[str, object]] = reference_rows.copy()

    run_specs = [
        ("fedavg_scratch_sample_weighted", "sample_weighted", True),
        ("fedavg_scratch_node_balanced", "node_balanced", True),
        ("continual_pretrained_sample_weighted", "sample_weighted", False),
    ]
    final_states: dict[str, dict[str, torch.Tensor]] = {}
    for run_id, aggregation, reset_head in run_specs:
        print(f"[run] {run_id}")
        metric_rows, rows, final_state = run_fedavg(
            run_id,
            reference_model,
            data,
            device,
            args.rounds,
            args.local_epochs,
            args.batch_size,
            args.lr,
            args.weight_decay,
            aggregation,
            reset_head,
            args.seed,
        )
        all_metric_rows.extend(metric_rows)
        round_rows.extend(rows)
        final_states[run_id] = final_state

    elapsed = time.perf_counter() - start
    write_csv(args.output_dir / "round_metrics.csv", round_rows)
    write_csv(args.output_dir / "per_node_metrics.csv", all_metric_rows)
    plot_rounds([row for row in round_rows if row["run_id"] != "centralized_reference"], args.output_dir / "fedavg_round_curves.png")
    shared_params = shared_head_parameter_count(reference_model)
    summary = {
        "task": "shared_head_federated_continual_update",
        "saved_args": saved_args,
        "metadata": metadata,
        "reference_macro": reference_macro,
        "reference_thresholds": reference_thresholds,
        "round_metrics": round_rows,
        "per_node_metrics": all_metric_rows,
        "shared_head_parameters": shared_params,
        "communication_payload": {
            "fp32_bytes_per_payload": shared_params * 4,
            "int8_bytes_per_payload_estimate": shared_params,
        },
        "elapsed_seconds": elapsed,
        "cautions": [
            "Only the shared risk head is federated; local encoders and projections are not federated.",
            "This is a software simulation using cached embeddings, not physical device communication.",
            "VBN remains a proxy vibration/time-series dataset.",
        ],
    }
    save_json(args.output_dir / "summary.json", summary)
    report_path = args.output_dir / "SHARED_HEAD_FEDERATED_UPDATE_REPORT.md"
    write_report(report_path, metadata, reference_rows, round_rows, all_metric_rows, shared_params, args.rounds, args.local_epochs, elapsed)

    torch.save({"final_shared_head_states": final_states, "args": vars(args)}, args.output_dir / "final_shared_head_states.pt")

    final_package = PROJECT_ROOT / "outputs" / "reports" / "final_results_package"
    if final_package.exists():
        shutil.copy2(report_path, final_package / report_path.name)
        shutil.copy2(args.output_dir / "round_metrics.csv", final_package / "shared_head_federated_round_metrics.csv")
        shutil.copy2(args.output_dir / "per_node_metrics.csv", final_package / "shared_head_federated_per_node_metrics.csv")
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
