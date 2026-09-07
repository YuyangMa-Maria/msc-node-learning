"""Simulate representative alerts and optional MARH intervention.

The scenario exercises trend, alert and helper-recommendation policy without
pretending that a software timeline is a physical deployment. MARH availability
and MARH recommendation are separate states: a helper may join routinely and a
recommendation does not force intervention.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nlrisk.fusion.message_fusion import fuse_messages, result_to_dict


@dataclass(frozen=True)
class AlertDecision:
    should_alert: bool
    should_recommend_marh: bool
    marh_joined: bool
    marh_recommendation: str
    alert_reason: str
    marh_reason: str
    representative_node: str
    external_receiver: str
    channel: str


def message(
    node_id: str,
    timestamp: str,
    risk: float,
    confidence: float,
    status: str,
    modality: str,
    quality: str = "normal",
) -> dict[str, object]:
    return {
        "node_id": node_id,
        "timestamp": timestamp,
        "risk_score": risk,
        "confidence": confidence,
        "status": status,
        "metadata": {"modality": modality, "quality": quality},
    }


def scenario_windows() -> list[dict[str, object]]:
    start = datetime(2026, 6, 27, 10, 0, tzinfo=timezone.utc)
    scenarios = [
        {
            "scenario": "stable_low_risk_no_alert",
            "steps": [
                (0.05, 0.04, 0.06, 0.94, 0.92, 0.90, "normal", "normal", "normal"),
                (0.07, 0.05, 0.08, 0.94, 0.92, 0.90, "normal", "normal", "normal"),
                (0.06, 0.06, 0.07, 0.94, 0.92, 0.90, "normal", "normal", "normal"),
                (0.08, 0.05, 0.06, 0.94, 0.92, 0.90, "normal", "normal", "normal"),
            ],
        },
        {
            "scenario": "healthy_nodes_optional_marh_camera_check",
            "proactive_marh_steps": {3},
            "steps": [
                (0.06, 0.05, 0.07, 0.94, 0.93, 0.91, "normal", "normal", "normal"),
                (0.08, 0.07, 0.09, 0.94, 0.93, 0.91, "normal", "normal", "normal"),
                (0.10, 0.08, 0.11, 0.94, 0.93, 0.91, "normal", "normal", "normal"),
                (0.07, 0.06, 0.08, 0.94, 0.93, 0.91, "normal", "normal", "normal"),
            ],
        },
        {
            "scenario": "rising_multimodal_alert_then_marh",
            "steps": [
                (0.18, 0.20, 0.16, 0.92, 0.90, 0.88, "normal", "normal", "normal"),
                (0.42, 0.48, 0.44, 0.90, 0.89, 0.87, "warning", "warning", "warning"),
                (0.70, 0.74, 0.68, 0.90, 0.88, 0.86, "warning", "warning", "warning"),
                (0.91, 0.88, 0.83, 0.90, 0.88, 0.86, "critical", "critical", "critical"),
            ],
        },
        {
            "scenario": "visual_degraded_vsn_still_relay",
            "steps": [
                (0.10, 0.18, 0.16, 0.30, 0.92, 0.88, "warning", "normal", "normal"),
                (0.08, 0.56, 0.52, 0.28, 0.93, 0.88, "warning", "warning", "warning"),
                (0.07, 0.87, 0.79, 0.25, 0.94, 0.88, "warning", "critical", "warning"),
                (0.06, 0.91, 0.84, 0.25, 0.94, 0.88, "warning", "critical", "critical"),
            ],
        },
        {
            "scenario": "representative_low_power_marh_relay_support",
            "steps": [
                (0.22, 0.24, 0.21, 0.58, 0.88, 0.84, "low_power", "normal", "normal"),
                (0.36, 0.44, 0.40, 0.45, 0.89, 0.84, "low_power", "warning", "warning"),
                (0.54, 0.70, 0.66, 0.35, 0.90, 0.84, "low_power", "warning", "warning"),
                (0.58, 0.82, 0.78, 0.30, 0.91, 0.84, "low_power", "critical", "warning"),
            ],
        },
        {
            "scenario": "uncertain_multinode_request_marh",
            "steps": [
                (0.44, 0.47, 0.42, 0.34, 0.36, 0.38, "warning", "warning", "warning"),
                (0.48, 0.50, 0.46, 0.32, 0.35, 0.36, "warning", "warning", "warning"),
                (0.51, 0.54, 0.50, 0.31, 0.34, 0.35, "warning", "warning", "warning"),
                (0.55, 0.57, 0.53, 0.30, 0.33, 0.34, "warning", "warning", "warning"),
            ],
        },
    ]
    windows: list[dict[str, object]] = []
    for scenario in scenarios:
        for step_idx, (v_risk, a_risk, b_risk, v_conf, a_conf, b_conf, v_status, a_status, b_status) in enumerate(scenario["steps"]):
            timestamp = (start + timedelta(minutes=len(windows))).isoformat()
            windows.append(
                {
                    "scenario": scenario["scenario"],
                    "step": step_idx + 1,
                    "timestamp": timestamp,
                    "proactive_marh": step_idx + 1 in scenario.get("proactive_marh_steps", set()),
                    "messages": [
                        message("VSN_01", timestamp, v_risk, v_conf, v_status, "vision", quality_from_confidence(v_conf, v_status)),
                        message("ASN_01", timestamp, a_risk, a_conf, a_status, "audio", quality_from_confidence(a_conf, a_status)),
                        message("VBN_01", timestamp, b_risk, b_conf, b_status, "vibration_proxy", quality_from_confidence(b_conf, b_status)),
                    ],
                }
            )
    return windows


def quality_from_confidence(confidence: float, status: str) -> str:
    if status == "offline":
        return "offline"
    if status == "low_power":
        return "low_power"
    if confidence < 0.45:
        return "degraded"
    return "normal"


def decide_alert(
    fusion: dict[str, object],
    messages: list[dict[str, object]],
    representative_node: str,
    external_receiver: str,
    proactive_marh: bool,
) -> AlertDecision:
    level = int(fusion["risk_level"])
    trend = str(fusion["risk_trend"])
    score = float(fusion["fused_risk_score"])
    confidence = float(fusion["confidence"])
    triggered = list(fusion["triggered_nodes"])
    rep_msg = next((item for item in messages if item["node_id"] == representative_node), None)
    rep_status = str(rep_msg.get("status", "normal")) if rep_msg else "offline"
    low_conf_nodes = [item["node_id"] for item in messages if float(item["confidence"]) < 0.45]
    offline_like_nodes = [item["node_id"] for item in messages if str(item["status"]) in {"offline", "low_power"}]

    should_alert = level >= 4 or (level >= 3 and trend == "rising") or (score >= 0.60 and len(triggered) >= 2)
    alert_reason = "no external alert"
    if should_alert:
        if level >= 4:
            alert_reason = f"risk_level={level}"
        elif trend == "rising":
            alert_reason = f"rising trend with risk_level={level}"
        else:
            alert_reason = "multiple triggered nodes"

    should_recommend_marh = (
        level >= 5
        or (level >= 4 and trend == "rising")
        or rep_status in {"low_power", "offline"}
        or len(low_conf_nodes) >= 2
        or len(offline_like_nodes) >= 1 and level >= 3
        or (level >= 3 and confidence < 0.45)
        or proactive_marh
    )
    marh_recommendation = "none"
    marh_reason = "not required"
    marh_joined = False
    if should_recommend_marh:
        if proactive_marh:
            marh_recommendation = "optional"
            marh_reason = "scheduled MARH camera inspection"
            marh_joined = True
        elif rep_status in {"low_power", "offline"}:
            marh_recommendation = "urgent" if level >= 4 else "suggested"
            marh_reason = f"representative node {representative_node} status={rep_status}"
            marh_joined = level >= 4
        elif len(low_conf_nodes) >= 2:
            marh_recommendation = "suggested"
            marh_reason = "multiple low-confidence nodes"
            marh_joined = trend == "rising"
        elif level >= 5:
            marh_recommendation = "urgent"
            marh_reason = "critical risk level"
            marh_joined = True
        elif trend == "rising":
            marh_recommendation = "urgent"
            marh_reason = "high and rising risk"
            marh_joined = True
        else:
            marh_recommendation = "suggested"
            marh_reason = "moderate risk with low confidence"
            marh_joined = False

    return AlertDecision(
        should_alert=should_alert,
        should_recommend_marh=should_recommend_marh,
        marh_joined=marh_joined,
        marh_recommendation=marh_recommendation,
        alert_reason=alert_reason,
        marh_reason=marh_reason,
        representative_node=representative_node,
        external_receiver=external_receiver,
        channel="LoRa",
    )


def marh_message(timestamp: str, local_score: float, reason: str) -> dict[str, object]:
    if "scheduled" in reason:
        risk = max(0.04, local_score - 0.01)
        status = "normal"
    elif "low-confidence" in reason:
        risk = max(0.58, local_score + 0.08)
        status = "warning"
    elif "representative node" in reason:
        if local_score < 0.60:
            risk = local_score + 0.03
            status = "normal" if risk < 0.40 else "warning"
        else:
            risk = max(0.70, local_score + 0.05)
            status = "warning" if risk < 0.80 else "critical"
    elif "critical" in reason or "rising" in reason:
        risk = max(0.86, local_score + 0.04)
        status = "critical"
    else:
        risk = max(local_score, 0.65)
        status = "warning"
    helper = message("MARH_01", timestamp, min(risk, 0.98), 0.96, status, "mobile_camera", "normal")
    helper["metadata"]["has_camera"] = True  # type: ignore[index]
    helper["metadata"]["role"] = "temporary_mobile_helper"  # type: ignore[index]
    return helper


def compact_alert_payload(
    fusion: dict[str, object],
    decision: AlertDecision,
    messages: list[dict[str, object]],
    marh_fusion: dict[str, object] | None = None,
) -> dict[str, object] | None:
    if not decision.should_alert and marh_fusion is None:
        return None
    source = marh_fusion if marh_fusion is not None else fusion
    return {
        "sender_node_id": decision.representative_node if marh_fusion is None else "MARH_01",
        "receiver": decision.external_receiver,
        "channel": decision.channel if marh_fusion is None else "MARH uplink",
        "timestamp": source["timestamp"],
        "fused_risk_score": round(float(source["fused_risk_score"]), 4),
        "risk_level": source["risk_level"],
        "risk_trend": source["risk_trend"],
        "triggered_nodes": source["triggered_nodes"],
        "node_status": {str(item["node_id"]): str(item["status"]) for item in messages},
        "alert_message": source["alert_message"],
    }


def run_simulation(method: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    histories: dict[str, list[float]] = {}
    marh_histories: dict[str, list[float]] = {}
    for window in scenario_windows():
        scenario = str(window["scenario"])
        history = histories.setdefault(scenario, [])
        messages = list(window["messages"])
        result = fuse_messages(messages, history=history, method=method, timestamp=str(window["timestamp"]))
        fusion = result_to_dict(result)
        history.append(result.fused_risk_score)
        decision = decide_alert(fusion, messages, "VSN_01", "ExternalReceiver_01", bool(window.get("proactive_marh", False)))

        marh_fusion = None
        marh_payload = None
        if decision.marh_joined:
            helper_message = marh_message(str(window["timestamp"]), float(fusion["fused_risk_score"]), decision.marh_reason)
            marh_messages = messages + [helper_message]
            marh_history = marh_histories.setdefault(scenario, [])
            refined = fuse_messages(marh_messages, history=marh_history, method="quality_filtered_max", timestamp=str(window["timestamp"]))
            marh_fusion = result_to_dict(refined)
            marh_history.append(refined.fused_risk_score)
            marh_payload = compact_alert_payload(marh_fusion, decision, marh_messages, marh_fusion=marh_fusion)

        rows.append(
            {
                "scenario": scenario,
                "step": window["step"],
                "timestamp": window["timestamp"],
                "representative_node": "VSN_01",
                "proactive_marh": window.get("proactive_marh", False),
                "fusion_method": method,
                "local_fusion": fusion,
                "decision": asdict(decision),
                "representative_alert_payload": compact_alert_payload(fusion, decision, messages),
                "marh_fusion": marh_fusion,
                "marh_alert_payload": marh_payload,
                "node_messages": messages,
            }
        )
    return rows


def row_value(row: dict[str, object], key: str, default: object = "") -> object:
    current: object = row
    for part in key.split("."):
        if not isinstance(current, dict):
            return default
        current = current.get(part, default)
    return current


def write_outputs(rows: list[dict[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "representative_alert_windows.jsonl"
    csv_path = output_dir / "representative_alert_summary.csv"
    plot_path = output_dir / "representative_alert_timeline.png"
    md_path = output_dir / "REPORT.md"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    fieldnames = [
        "scenario",
        "step",
        "timestamp",
        "representative_node",
        "fusion_method",
        "local_fusion.fused_risk_score",
        "local_fusion.risk_level",
        "local_fusion.risk_trend",
        "local_fusion.confidence",
        "decision.should_alert",
        "decision.alert_reason",
        "decision.should_recommend_marh",
        "decision.marh_recommendation",
        "decision.marh_joined",
        "decision.marh_reason",
        "marh_fusion.fused_risk_score",
        "marh_fusion.risk_level",
        "marh_fusion.risk_trend",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row_value(row, field) for field in fieldnames})

    plot_timeline(rows, plot_path)
    md_path.write_text(report_text(rows), encoding="utf-8")
    print(f"Wrote {jsonl_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {plot_path}")
    print(f"Wrote {md_path}")


def plot_timeline(rows: list[dict[str, object]], path: Path) -> None:
    scenarios = list(dict.fromkeys(str(row["scenario"]) for row in rows))
    fig, axes = plt.subplots(len(scenarios), 1, figsize=(10.5, 2.4 * len(scenarios)), dpi=140, sharex=False)
    if len(scenarios) == 1:
        axes = [axes]
    for ax, scenario in zip(axes, scenarios):
        subset = [row for row in rows if row["scenario"] == scenario]
        steps = [int(row["step"]) for row in subset]
        local_scores = [float(row_value(row, "local_fusion.fused_risk_score", 0.0)) for row in subset]
        marh_scores = [
            float(row_value(row, "marh_fusion.fused_risk_score", float("nan")))
            if row.get("marh_fusion")
            else float("nan")
            for row in subset
        ]
        ax.plot(steps, local_scores, marker="o", label="Local fused risk")
        ax.plot(steps, marh_scores, marker="s", linestyle="--", label="MARH refined risk")
        for row in subset:
            if row_value(row, "decision.should_alert", False):
                ax.axvline(int(row["step"]), color="tab:red", alpha=0.15)
        ax.set_title(scenario)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Risk")
        ax.set_xticks(steps)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Time window")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def report_text(rows: list[dict[str, object]]) -> str:
    lines = [
        "# Representative Alert and MARH Simulation Report",
        "",
        "## Setup",
        "",
        "- Nodes exchange lightweight summaries from VSN, ASN, and VBN.",
        "- Local fusion uses `quality_filtered_max` by default.",
        "- VSN acts as the representative node for normal external LoRa alert forwarding.",
        "- MARH is a temporary mobile/resource-rich helper with its own camera, not a permanent center.",
        "- MARH may be recommended under critical, rising, low-confidence, or representative-node-degraded conditions, but recommendation does not mean it always joins.",
        "- MARH may also enter proactively for scheduled camera inspection even when local nodes are healthy.",
        "- External receiver is treated as an information display and decision endpoint.",
        "",
        "## Decision Rules",
        "",
        "- Send external alert if `risk_level >= 4`, or if `risk_level >= 3` with rising trend, or if multiple nodes trigger with fused risk above 0.60.",
        "- Recommend MARH if risk is critical, high and rising, multiple nodes are low-confidence, or the representative VSN is low-power/offline.",
        "- Optional MARH camera inspection can occur when nodes are healthy, representing proactive external support.",
        "- MARH joins only when the recommendation is accepted in the simulation policy, then adds a high-confidence camera helper message and produces a refined fused risk score.",
        "",
        "## Results",
        "",
        "| Scenario | Step | Local risk | Level | Trend | Alert? | MARH rec. | Joined? | Reason | MARH risk |",
        "| --- | ---: | ---: | ---: | --- | --- | --- | --- | --- | ---: |",
    ]
    for row in rows:
        marh_score = row_value(row, "marh_fusion.fused_risk_score", "")
        if isinstance(marh_score, float):
            marh_text = f"{marh_score:.3f}"
        else:
            marh_text = ""
        lines.append(
            "| {scenario} | {step} | {risk:.3f} | {level} | {trend} | {alert} | {marh} | {joined} | {reason} | {marh_risk} |".format(
                scenario=row["scenario"],
                step=row["step"],
                risk=float(row_value(row, "local_fusion.fused_risk_score", 0.0)),
                level=row_value(row, "local_fusion.risk_level"),
                trend=row_value(row, "local_fusion.risk_trend"),
                alert="yes" if row_value(row, "decision.should_alert", False) else "no",
                marh=str(row_value(row, "decision.marh_recommendation", "none")),
                joined="yes" if row_value(row, "decision.marh_joined", False) else "no",
                reason=row_value(row, "decision.marh_reason") if row_value(row, "decision.should_recommend_marh", False) else row_value(row, "decision.alert_reason"),
                marh_risk=marh_text,
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Stable low-risk windows produce no external alert and no required MARH intervention.",
            "- MARH can still join as a proactive camera helper during scheduled inspection when local nodes are healthy.",
            "- Rising multimodal risk is first forwarded by VSN and later escalated with urgent MARH recommendation when it becomes high/critical.",
            "- When VSN sensing quality is degraded but the node is still available, it can still serve as representative relay while ASN/VBN provide the main risk evidence.",
            "- If VSN is low-power, MARH can be suggested for relay/support; it joins once risk becomes high enough in the simulation policy.",
            "- If several nodes have low confidence, MARH is suggested to refine the local decision rather than overclaiming certainty, but may remain pending until the trend rises.",
            "",
            "## Files",
            "",
            "- Full event log: `representative_alert_windows.jsonl`",
            "- Summary CSV: `representative_alert_summary.csv`",
            "- Timeline plot: `representative_alert_timeline.png`",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate representative-node alert forwarding and MARH intervention.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "representative_alert_marh")
    parser.add_argument(
        "--method",
        default="quality_filtered_max",
        choices=["confidence_weighted", "max_risk", "quality_gate", "quality_filtered_max"],
    )
    args = parser.parse_args()
    rows = run_simulation(args.method)
    write_outputs(rows, args.output_dir)


if __name__ == "__main__":
    main()
