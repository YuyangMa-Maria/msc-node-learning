"""Exercise the reference risk-message fusion logic on controlled scenarios."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from nlrisk.fusion.message_fusion import fuse_messages, result_to_dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def message(node_id: str, timestamp: str, risk: float, confidence: float, status: str, modality: str) -> dict[str, object]:
    return {
        "node_id": node_id,
        "timestamp": timestamp,
        "risk_score": risk,
        "confidence": confidence,
        "status": status,
        "metadata": {"modality": modality},
    }


def scenario_windows() -> list[dict[str, object]]:
    start = datetime(2026, 6, 27, 9, 0, tzinfo=timezone.utc)
    scenarios = [
        {
            "scenario": "stable_low_risk",
            "steps": [
                (0.05, 0.04, "normal", "normal"),
                (0.07, 0.05, "normal", "normal"),
                (0.06, 0.06, "normal", "normal"),
                (0.08, 0.05, "normal", "normal"),
            ],
        },
        {
            "scenario": "rising_multimodal_risk",
            "steps": [
                (0.20, 0.18, "normal", "normal"),
                (0.35, 0.42, "warning", "warning"),
                (0.62, 0.70, "warning", "warning"),
                (0.88, 0.91, "critical", "critical"),
            ],
        },
        {
            "scenario": "visual_degraded_audio_reliable",
            "steps": [
                (0.08, 0.18, "normal", "normal"),
                (0.10, 0.55, "warning", "warning"),
                (0.07, 0.86, "warning", "critical"),
                (0.06, 0.91, "warning", "critical"),
            ],
            "visual_confidence": 0.25,
            "audio_confidence": 0.94,
        },
        {
            "scenario": "audio_noisy_visual_reliable",
            "steps": [
                (0.18, 0.12, "normal", "normal"),
                (0.52, 0.10, "warning", "warning"),
                (0.84, 0.09, "critical", "warning"),
                (0.90, 0.12, "critical", "warning"),
            ],
            "visual_confidence": 0.95,
            "audio_confidence": 0.25,
        },
        {
            "scenario": "both_degraded_uncertain",
            "steps": [
                (0.45, 0.48, "warning", "warning"),
                (0.50, 0.52, "warning", "warning"),
                (0.47, 0.55, "warning", "warning"),
                (0.53, 0.50, "warning", "warning"),
            ],
            "visual_confidence": 0.35,
            "audio_confidence": 0.35,
        },
    ]
    windows: list[dict[str, object]] = []
    for scenario in scenarios:
        visual_conf = float(scenario.get("visual_confidence", 0.92))
        audio_conf = float(scenario.get("audio_confidence", 0.90))
        for step_idx, (visual_risk, audio_risk, visual_status, audio_status) in enumerate(scenario["steps"]):
            timestamp = (start + timedelta(minutes=len(windows))).isoformat()
            windows.append(
                {
                    "scenario": scenario["scenario"],
                    "step": step_idx + 1,
                    "timestamp": timestamp,
                    "messages": [
                        message("VSN_01", timestamp, visual_risk, visual_conf, visual_status, "vision"),
                        message("ASN_01", timestamp, audio_risk, audio_conf, audio_status, "audio"),
                    ],
                }
            )
    return windows


def run_simulation(methods: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    histories: dict[tuple[str, str], list[float]] = {}
    for window in scenario_windows():
        scenario = str(window["scenario"])
        for method in methods:
            key = (scenario, method)
            history = histories.setdefault(key, [])
            result = fuse_messages(window["messages"], history=history, method=method, timestamp=str(window["timestamp"]))
            result_dict = result_to_dict(result)
            history.append(result.fused_risk_score)
            rows.append(
                {
                    "scenario": scenario,
                    "step": window["step"],
                    "method": method,
                    **result_dict,
                    "node_messages": window["messages"],
                }
            )
    return rows


def write_outputs(rows: list[dict[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "message_fusion_windows.jsonl"
    csv_path = output_dir / "message_fusion_summary.csv"
    md_path = output_dir / "REPORT.md"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "scenario",
            "step",
            "method",
            "fused_risk_score",
            "confidence",
            "risk_level",
            "risk_trend",
            "triggered_nodes",
            "alert_message",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})

    lines = [
        "# Message-Level Fusion Simulation Report",
        "",
        "## Setup",
        "",
        "- Input: lightweight VSN/ASN node messages, not raw image/audio.",
        "- Output: `fused_risk_score`, `risk_level`, `risk_trend`, `triggered_nodes`, and rule-based `alert_message`.",
        "- Methods: confidence-weighted, max-risk, and quality-gate fusion.",
        "- Trend is computed from previous fused scores in the same scenario.",
        "",
        "## Results",
        "",
        "| Scenario | Step | Method | Fused risk | Confidence | Level | Trend | Triggered nodes | Alert |",
        "| --- | ---: | --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {scenario} | {step} | {method} | {score:.3f} | {conf:.3f} | {level} | {trend} | {nodes} | {alert} |".format(
                scenario=row["scenario"],
                step=row["step"],
                method=row["method"],
                score=row["fused_risk_score"],
                conf=row["confidence"],
                level=row["risk_level"],
                trend=row["risk_trend"],
                nodes=", ".join(row["triggered_nodes"]),
                alert=row["alert_message"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Low-risk windows remain Level 1 and produce monitoring messages.",
            "- Rising multimodal risk moves from low/moderate levels to Level 5 critical alerts.",
            "- When one modality is degraded, quality-gate fusion follows the reliable node.",
            "- When both modalities are uncertain, the fused score remains moderate and the alert recommends caution rather than overclaiming critical risk.",
            "",
            "## Files",
            "",
            "- Full window JSONL: `message_fusion_windows.jsonl`",
            "- Summary CSV: `message_fusion_summary.csv`",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {jsonl_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate message-level VSN/ASN local fusion.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "message_fusion")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["confidence_weighted", "max_risk", "quality_gate"],
        choices=["confidence_weighted", "max_risk", "quality_gate"],
    )
    args = parser.parse_args()
    rows = run_simulation(args.methods)
    write_outputs(rows, args.output_dir)


if __name__ == "__main__":
    main()
