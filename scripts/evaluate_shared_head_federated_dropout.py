"""Stress-test shared-head updates with intermittent node participation.

The schedules model availability rather than packet-level faults: missing nodes
perform no local update in that round, while the most recent valid global head
remains available to every node for inference.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from evaluate_parameter_sharing_cnn_embeddings import (
    DEFAULT_ASN_CHECKPOINT,
    DEFAULT_ASN_INDEX,
    DEFAULT_VBN_FEATURES,
    DEFAULT_VSN_CHECKPOINT,
    DEFAULT_VSN_INDEX,
    PROJECT_ROOT,
    save_json,
)
from evaluate_shared_head_federated_update import (
    average_states,
    evaluate_global_model,
    freeze_all_but_shared_head,
    load_clean_modality_data,
    load_reference_model,
    load_shared_head_state,
    local_train_shared_head,
    reset_shared_head,
    set_seed,
    shared_head_parameter_count,
    shared_head_state,
)


NODES = ["vsn", "asn", "vbn"]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def participants_for_schedule(schedule: str, round_idx: int) -> list[str]:
    """Return the deterministic participant set for one simulated round."""
    if schedule == "full_participation":
        return NODES.copy()
    if schedule == "vbn_absent":
        return ["vsn", "asn"]
    if schedule == "asn_intermittent":
        return ["vsn", "asn", "vbn"] if round_idx % 2 == 0 else ["vsn", "vbn"]
    if schedule == "rotating_missing_one":
        missing = ["vbn", "asn", "vsn"][(round_idx - 1) % 3]
        return [node for node in NODES if node != missing]
    if schedule == "sparse_one_node":
        return [NODES[(round_idx - 1) % 3]]
    raise ValueError(f"Unknown schedule: {schedule}")


def schedule_description(schedule: str) -> str:
    descriptions = {
        "full_participation": "All VSN, ASN, and VBN participate in every round.",
        "vbn_absent": "VBN never participates; only VSN and ASN update the shared head.",
        "asn_intermittent": "ASN participates only every second round, simulating unstable acoustic-node availability.",
        "rotating_missing_one": "Exactly one node is missing each round, rotating across VBN, ASN, and VSN.",
        "sparse_one_node": "Only one node participates per round, rotating across VSN, ASN, and VBN.",
    }
    return descriptions[schedule]


def run_partial_fedavg(
    run_id: str,
    schedule: str,
    reference_model,
    data,
    device: torch.device,
    rounds: int,
    local_epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    reset_head: bool,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """Aggregate only the clients available under the selected schedule."""
    model = torch.deepcopy(reference_model) if hasattr(torch, "deepcopy") else None
    if model is None:
        import copy

        model = copy.deepcopy(reference_model)
    model.to(device)
    if reset_head:
        reset_shared_head(model, seed + 1200)
    freeze_all_but_shared_head(model)
    global_state = shared_head_state(model)
    client_sizes = {node: len(data[node].train_y) for node in NODES}

    all_metric_rows: list[dict[str, object]] = []
    round_rows: list[dict[str, object]] = []
    participation_rows: list[dict[str, object]] = []

    metric_rows, macro, _thresholds = evaluate_global_model(model, data, device, run_id, 0)
    all_metric_rows.extend(metric_rows)
    round_rows.append(
        {
            "run_id": run_id,
            "schedule": schedule,
            "round": 0,
            "reset_head": reset_head,
            "participants": "initial",
            "participant_count": 0,
            **macro,
        }
    )

    for round_idx in range(1, rounds + 1):
        participants = participants_for_schedule(schedule, round_idx)
        local_states: list[dict[str, torch.Tensor]] = []
        local_weights: list[float] = []
        for node in participants:
            state = local_train_shared_head(
                model,
                global_state,
                node,
                data[node].train_x,
                data[node].train_y,
                device,
                local_epochs,
                batch_size,
                lr,
                weight_decay,
            )
            local_states.append(state)
            local_weights.append(float(client_sizes[node]))
        # A non-participant contributes neither stale gradients nor an implicit
        # zero update; aggregation is over the observed clients only.
        global_state = average_states(local_states, local_weights)
        load_shared_head_state(model, global_state, device)
        metric_rows, macro, _thresholds = evaluate_global_model(model, data, device, run_id, round_idx)
        all_metric_rows.extend(metric_rows)
        participant_text = "+".join(participants)
        round_rows.append(
            {
                "run_id": run_id,
                "schedule": schedule,
                "round": round_idx,
                "reset_head": reset_head,
                "participants": participant_text,
                "participant_count": len(participants),
                **macro,
            }
        )
        participation_rows.append(
            {
                "run_id": run_id,
                "schedule": schedule,
                "round": round_idx,
                "participants": participant_text,
                "participant_count": len(participants),
            }
        )
    return all_metric_rows, round_rows, participation_rows


def plot_rounds(round_rows: list[dict[str, object]], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), dpi=140)
    for run_id in sorted({str(row["run_id"]) for row in round_rows}):
        rows = sorted([row for row in round_rows if row["run_id"] == run_id], key=lambda row: int(row["round"]))
        axes[0].plot([int(row["round"]) for row in rows], [float(row["macro_f1"]) for row in rows], marker="o", label=run_id)
        axes[1].plot([int(row["round"]) for row in rows], [float(row["macro_recall"]) for row in rows], marker="o", label=run_id)
    axes[0].set_title("Partial Participation Test Macro F1")
    axes[1].set_title("Partial Participation Test Macro Recall")
    for ax in axes:
        ax.set_xlabel("Federated round")
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.25)
    axes[0].legend(fontsize=7)
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
    round_rows: list[dict[str, object]],
    per_node_rows: list[dict[str, object]],
    participation_rows: list[dict[str, object]],
    shared_params: int,
    rounds: int,
    elapsed: float,
) -> None:
    final_rows = [row for row in round_rows if int(row["round"]) == rounds]
    best = max(final_rows, key=lambda row: float(row["macro_f1"]))
    worst = min(final_rows, key=lambda row: float(row["macro_f1"]))
    final_summary = [
        {
            "run_id": row["run_id"],
            "schedule": row["schedule"],
            "macro_f1": row["macro_f1"],
            "macro_recall": row["macro_recall"],
            "macro_accuracy": row["macro_accuracy"],
        }
        for row in final_rows
    ]
    schedule_rows = [
        {"schedule": schedule, "description": schedule_description(schedule)}
        for schedule in ["full_participation", "vbn_absent", "asn_intermittent", "rotating_missing_one", "sparse_one_node"]
    ]
    payload_fp32 = shared_params * 4
    payload_int8 = shared_params
    communication_rows = []
    for run_id in sorted({str(row["run_id"]) for row in participation_rows}):
        rows = [row for row in participation_rows if row["run_id"] == run_id]
        avg_participants = float(np.mean([int(row["participant_count"]) for row in rows]))
        total_upload_fp32 = int(sum(int(row["participant_count"]) for row in rows) * payload_fp32)
        communication_rows.append(
            {
                "run_id": run_id,
                "avg_participants_per_round": avg_participants,
                "total_upload_fp32_kb": total_upload_fp32 / 1024,
                "total_upload_int8_kb_est": total_upload_fp32 / 4 / 1024,
            }
        )
    best_per_node = [
        {
            "modality": row["modality"],
            "f1": row["f1_positive"],
            "recall": row["recall_positive"],
            "accuracy": row["accuracy"],
        }
        for row in per_node_rows
        if row["run_id"] == best["run_id"] and row["split"] == "test" and int(row["round"]) == rounds
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
    lines = [
        "# Shared-Head Federated Update with Node Dropout",
        "",
        "## Purpose",
        "",
        "This experiment stress-tests the shared-head federated update under partial node participation. It simulates disaster-network conditions where some nodes are offline, low power, or unable to upload model updates in every round.",
        "",
        "## Node Representations",
        "",
        markdown_table(representation_rows, ["node", "representation", "dim", "train/val/test"]),
        "",
        "## Participation Schedules",
        "",
        markdown_table(schedule_rows, ["schedule", "description"]),
        "",
        "## Final Macro Results",
        "",
        markdown_table(final_summary, ["run_id", "schedule", "macro_f1", "macro_recall", "macro_accuracy"]),
        "",
        "## Best Run Per-Node Test Results",
        "",
        f"Best final run: `{best['run_id']}` with macro F1 {float(best['macro_f1']):.4f}. Worst final run: `{worst['run_id']}` with macro F1 {float(worst['macro_f1']):.4f}.",
        "",
        markdown_table(best_per_node, ["modality", "f1", "recall", "accuracy"]),
        "",
        "## Communication Estimate",
        "",
        f"- Shared head payload per upload: {payload_fp32} bytes FP32 ({payload_fp32 / 1024:.2f} KB), or about {payload_int8} bytes INT8 ({payload_int8 / 1024:.2f} KB).",
        markdown_table(communication_rows, ["run_id", "avg_participants_per_round", "total_upload_fp32_kb", "total_upload_int8_kb_est"]),
        "",
        "## Interpretation",
        "",
        "- Full participation is still the cleanest condition, but partial participation can remain viable when at least the stronger VSN/ASN nodes appear regularly.",
        "- Sparse one-node participation is a stress case; if it degrades, that supports using status-aware routing, MARH intervention, or delayed aggregation rather than blindly updating from a single node.",
        "- This strengthens the Node Learning narrative because the system now evaluates both score communication and model-parameter communication under unreliable-node conditions.",
        "",
        "## Claims to Avoid / Cautious Wording",
        "",
        "- This is a software simulation using cached embeddings, not real BLE/LoRa model transmission.",
        "- Only the shared higher risk head is updated; the low-level encoders are not federated.",
        "- VBN remains a proxy structural time-series dataset, not real MPU6050 building vibration validation.",
        "- The result supports risk-proxy learning, not certified collapse prediction.",
        "",
        f"Runtime: {elapsed:.1f} seconds.",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Shared-head federated update under partial node participation.")
    parser.add_argument("--vsn-index", type=Path, default=DEFAULT_VSN_INDEX)
    parser.add_argument("--asn-index", type=Path, default=DEFAULT_ASN_INDEX)
    parser.add_argument("--vbn-features", type=Path, default=DEFAULT_VBN_FEATURES)
    parser.add_argument("--vsn-checkpoint", type=Path, default=DEFAULT_VSN_CHECKPOINT)
    parser.add_argument("--asn-checkpoint", type=Path, default=DEFAULT_ASN_CHECKPOINT)
    parser.add_argument("--base-output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "parameter_sharing_cnn_embeddings")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "shared_head_federated_dropout")
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
    shared_params = shared_head_parameter_count(reference_model)

    run_specs = [
        ("scratch_full_participation", "full_participation", True),
        ("scratch_vbn_absent", "vbn_absent", True),
        ("scratch_asn_intermittent", "asn_intermittent", True),
        ("scratch_rotating_missing_one", "rotating_missing_one", True),
        ("scratch_sparse_one_node", "sparse_one_node", True),
        ("continual_rotating_missing_one", "rotating_missing_one", False),
    ]
    all_metric_rows: list[dict[str, object]] = []
    all_round_rows: list[dict[str, object]] = []
    all_participation_rows: list[dict[str, object]] = []
    for run_id, schedule, reset_head in run_specs:
        print(f"[run] {run_id}")
        metric_rows, round_rows, participation_rows = run_partial_fedavg(
            run_id,
            schedule,
            reference_model,
            data,
            device,
            args.rounds,
            args.local_epochs,
            args.batch_size,
            args.lr,
            args.weight_decay,
            reset_head,
            args.seed,
        )
        all_metric_rows.extend(metric_rows)
        all_round_rows.extend(round_rows)
        all_participation_rows.extend(participation_rows)

    elapsed = time.perf_counter() - start
    write_csv(args.output_dir / "round_metrics.csv", all_round_rows)
    write_csv(args.output_dir / "per_node_metrics.csv", all_metric_rows)
    write_csv(args.output_dir / "participation_schedule.csv", all_participation_rows)
    plot_rounds(all_round_rows, args.output_dir / "partial_participation_round_curves.png")
    summary = {
        "task": "shared_head_federated_update_node_dropout",
        "saved_args": saved_args,
        "metadata": metadata,
        "round_metrics": all_round_rows,
        "per_node_metrics": all_metric_rows,
        "participation_schedule": all_participation_rows,
        "shared_head_parameters": shared_params,
        "communication_payload": {
            "fp32_bytes_per_upload": shared_params * 4,
            "int8_bytes_per_upload_estimate": shared_params,
        },
        "elapsed_seconds": elapsed,
        "cautions": [
            "Software simulation only; not real wireless model transmission.",
            "Only the shared higher risk head is updated.",
            "VBN remains a proxy vibration/time-series dataset.",
        ],
    }
    save_json(args.output_dir / "summary.json", summary)
    report_path = args.output_dir / "SHARED_HEAD_FEDERATED_DROPOUT_REPORT.md"
    write_report(report_path, metadata, all_round_rows, all_metric_rows, all_participation_rows, shared_params, args.rounds, elapsed)

    final_package = PROJECT_ROOT / "outputs" / "reports" / "final_results_package"
    if final_package.exists():
        shutil.copy2(report_path, final_package / report_path.name)
        shutil.copy2(args.output_dir / "round_metrics.csv", final_package / "shared_head_dropout_round_metrics.csv")
        shutil.copy2(args.output_dir / "per_node_metrics.csv", final_package / "shared_head_dropout_per_node_metrics.csv")
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
