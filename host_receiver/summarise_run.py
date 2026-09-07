"""Generate a compact JSON and Markdown report from a receiver run."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path


def load_events(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def calculate_ble_outages(events: list[dict]) -> list[dict]:
    outages = []
    for index, event in enumerate(events):
        if event.get("event") != "ble_state" or event.get("connected"):
            continue
        disconnected_at = datetime.fromisoformat(event["host_time_iso"])
        reconnect = next(
            (
                candidate
                for candidate in events[index + 1 :]
                if candidate.get("event") == "ble_state" and candidate.get("connected")
            ),
            None,
        )
        if reconnect is None:
            outages.append(
                {
                    "reason": event.get("reason"),
                    "disconnected_at": event["host_time_iso"],
                    "recovered": False,
                }
            )
            continue
        reconnected_at = datetime.fromisoformat(reconnect["host_time_iso"])
        first_dual = next(
            (
                candidate
                for candidate in events[index + 1 :]
                if candidate.get("event") == "fusion"
                and candidate.get("mode") == "dual"
                and datetime.fromisoformat(candidate["host_time_iso"]) >= reconnected_at
            ),
            None,
        )
        record = {
            "reason": event.get("reason"),
            "disconnected_at": event["host_time_iso"],
            "reconnected_at": reconnect["host_time_iso"],
            "recovered": True,
            "reconnect_delay_s": (reconnected_at - disconnected_at).total_seconds(),
        }
        if first_dual is not None:
            first_dual_at = datetime.fromisoformat(first_dual["host_time_iso"])
            record.update(
                {
                    "first_dual_at": first_dual["host_time_iso"],
                    "recovery_to_dual_s": (first_dual_at - disconnected_at).total_seconds(),
                    "post_connect_to_dual_s": (first_dual_at - reconnected_at).total_seconds(),
                }
            )
        outages.append(record)
    return outages


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    events = load_events(args.run_dir / "events.jsonl")
    fusions = [event for event in events if event.get("event") == "fusion"]
    ble = [event for event in events if event.get("event") == "ble_state"]
    commands = [event for event in events if event.get("event") == "command_result"]
    if not fusions:
        raise SystemExit("No fusion events were recorded.")

    host_times = [datetime.fromisoformat(event["host_time_iso"]) for event in events]
    risks = [event["fusion"]["risk_score"] for event in fusions]
    outages = calculate_ble_outages(events)
    summary = {
        "duration_s": (max(host_times) - min(host_times)).total_seconds(),
        "event_count": len(events),
        "fusion_count": len(fusions),
        "dual_fusion_count": sum(event["mode"] == "dual" for event in fusions),
        "vsn_only_count": sum(event["mode"] == "vsn_only" for event in fusions),
        "ble_disconnect_count": sum(not event["connected"] for event in ble),
        "ble_connect_count": sum(event["connected"] for event in ble),
        "risk_mean": statistics.fmean(risks),
        "risk_min": min(risks),
        "risk_max": max(risks),
        "risk_level_counts": dict(Counter(str(event["fusion"]["risk_level"]) for event in fusions)),
        "alert_action_counts": dict(Counter(event["alert"]["action"] for event in fusions)),
        "marh_recommendation_count": sum(event["marh"]["recommended"] for event in fusions),
        "model_versions": sorted({event["model"]["version"] for event in fusions}),
        "command_count": len(commands),
        "successful_command_count": sum(bool(event.get("success")) for event in commands),
        "ble_outages": outages,
    }
    (args.run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    lines = [
        "# Final-System Hardware Run",
        "",
        f"- Duration: {summary['duration_s']:.1f} s",
        f"- Structured events: {summary['event_count']}",
        f"- Fusion outputs: {summary['fusion_count']} "
        f"({summary['dual_fusion_count']} dual, {summary['vsn_only_count']} VSN-only)",
        f"- BLE connections/disconnections: {summary['ble_connect_count']}/"
        f"{summary['ble_disconnect_count']}",
        f"- Fused risk mean/range: {summary['risk_mean']:.4f} "
        f"[{summary['risk_min']:.4f}, {summary['risk_max']:.4f}]",
        f"- Risk-level counts: {summary['risk_level_counts']}",
        f"- Alert actions: {summary['alert_action_counts']}",
        f"- MARH recommendations: {summary['marh_recommendation_count']}",
        f"- Active model versions observed: {summary['model_versions']}",
        f"- Successful host commands: {summary['successful_command_count']}/"
        f"{summary['command_count']}",
    ]
    for outage in outages:
        if outage["recovered"]:
            lines.append(
                "- BLE outage recovery: "
                f"reconnected in {outage['reconnect_delay_s']:.3f} s; "
                f"first fresh dual fusion in {outage.get('recovery_to_dual_s', float('nan')):.3f} s"
            )
        else:
            lines.append("- BLE outage recovery: not observed before capture ended")
    lines.extend(
        [
            "",
            "The PC recorded representative-node outputs and did not recompute the online fusion decision.",
        ]
    )
    (args.run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.run_dir / "summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
