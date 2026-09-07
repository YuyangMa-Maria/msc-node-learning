"""Extract report-ready evidence from one physical final-system run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


KEY_VALUE = re.compile(r"([A-Za-z_]+)=([^\s]+)")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
    return records


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str]) -> None:
    fields = list(fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def numeric(values: Iterable[Any], allow_negative: bool = False) -> list[float]:
    output = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number) and (allow_negative or number >= 0):
            output.append(number)
    return output


def describe(values: Iterable[Any], allow_negative: bool = False) -> dict[str, float | int | None]:
    data = sorted(numeric(values, allow_negative=allow_negative))
    if not data:
        return {"n": 0, "mean": None, "median": None, "p95": None, "min": None, "max": None}
    rank = max(0, math.ceil(0.95 * len(data)) - 1)
    return {
        "n": len(data),
        "mean": statistics.fmean(data),
        "median": statistics.median(data),
        "p95": data[rank],
        "min": data[0],
        "max": data[-1],
    }


def parse_time(line: str) -> str:
    return line.split(" ", 1)[0] if "T" in line[:40] else ""


def values_after(line: str, marker: str) -> dict[str, str] | None:
    if marker not in line:
        return None
    suffix = line.split(marker, 1)[1]
    return {match.group(1): match.group(2).strip(",") for match in KEY_VALUE.finditer(suffix)}


def parse_vsn_raw(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    samples: list[dict[str, Any]] = []
    link: list[dict[str, Any]] = []
    model: list[dict[str, Any]] = []
    if not path.exists():
        return samples, link, model
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        when = parse_time(line)
        values = values_after(line, "VSN_LOCAL ")
        if values:
            samples.append({"host_time_iso": when, "node": "VSN", **values})
            continue
        values = values_after(line, "BLE_TARGET_FOUND ")
        if values:
            link.append({"host_time_iso": when, "event": "target_found", **values})
            continue
        if "BLE_CONNECTED" in line or "BLE_DISCONNECTED" in line:
            marker = "BLE_CONNECTED" if "BLE_CONNECTED" in line else "BLE_DISCONNECTED"
            values = values_after(line, marker) or {}
            link.append({"host_time_iso": when, "event": marker.lower(), **values})
            continue
        for marker in (
            "MODEL_PUSH_RESULT ",
            "MODEL_PULL_RESULT ",
            "RUNTIME_HEAD_ACTIVATE ",
            "RUNTIME_HEAD_INSTALL ",
            "RUNTIME_HEAD_HOST_ROLLBACK ",
            "RUNTIME_HEAD_ROLLBACK ",
        ):
            values = values_after(line, marker)
            if values:
                model.append({"host_time_iso": when, "operation": marker.strip().lower(), **values})
                break
    return samples, link, model


def parse_asn_raw(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    samples: list[dict[str, Any]] = []
    link: list[dict[str, Any]] = []
    memories: list[dict[str, Any]] = []
    pending_windows: list[dict[str, str]] = []
    if not path.exists():
        return samples, link, memories
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        when = parse_time(line)
        values = values_after(line, "MIC_WINDOW ")
        if values:
            pending_windows.append({"host_time_iso": when, **values})
            continue
        values = values_after(line, "NODE_OUTPUT ")
        if values and values.get("source") == "microphone":
            window = pending_windows.pop(0) if pending_windows else {}
            samples.append({**window, "host_time_iso": when, "node": "ASN", **values})
            continue
        values = values_after(line, "BLE_ACK ")
        if values:
            link.append({"host_time_iso": when, "event": "summary_ack", **values})
            continue
        values = values_after(line, "MEMORY stage=")
        if values:
            # The marker consumes the key, so recover the stage explicitly.
            stage = line.split("MEMORY stage=", 1)[1].split()[0]
            memories.append({"host_time_iso": when, "node": "ASN", "stage": stage, **values})
    return samples, link, memories


def parse_memory(path: Path, node: str) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        lower = line.lower()
        if "memory stage=" in lower:
            lower_index = lower.find("memory stage=")
            suffix = line[lower_index + len("memory stage="):]
            stage = suffix.split()[0]
        elif "online_memory " in lower:
            lower_index = lower.find("online_memory ")
            suffix = line[lower_index + len("online_memory "):]
            stage = "online"
        else:
            continue
        values = {m.group(1): m.group(2).strip(",") for m in KEY_VALUE.finditer(suffix)}
        rows.append({"host_time_iso": parse_time(line), "node": node, "stage": stage, **values})
    return rows


def flatten_fusions(events: list[dict[str, Any]], existing: list[dict[str, str]]) -> list[dict[str, Any]]:
    if existing:
        return existing
    rows = []
    for record in events:
        if record.get("event") != "fusion":
            continue
        vsn, asn, fusion = record["vsn"], record["asn"], record["fusion"]
        rows.append({
            "host_time_iso": record.get("host_time_iso"),
            "device_uptime_ms": record.get("uptime_ms"),
            "fusion_index": record.get("fusion_index"),
            "mode": record.get("mode"),
            "vsn_sequence": vsn.get("sequence"),
            "vsn_risk_score": vsn.get("risk_score"),
            "vsn_confidence": vsn.get("confidence"),
            "vsn_status": vsn.get("status"),
            "asn_available": asn.get("available"),
            "asn_sequence": asn.get("sequence"),
            "asn_age_ms": asn.get("age_ms"),
            "asn_risk_score": asn.get("risk_score"),
            "asn_confidence": asn.get("confidence"),
            "asn_status": asn.get("status"),
            "fused_risk_score": fusion.get("risk_score"),
            "fused_confidence": fusion.get("confidence"),
            "risk_level": fusion.get("risk_level"),
            "risk_trend": fusion.get("risk_trend"),
            "active_nodes": fusion.get("active_nodes"),
            "alert_attention_required": record.get("alert", {}).get("attention_required"),
            "alert_severity": record.get("alert", {}).get("severity"),
            "alert_headline": record.get("alert", {}).get("headline"),
            "alert_evidence": record.get("alert", {}).get("evidence"),
            "alert_action": record.get("alert", {}).get("action"),
            "alert_message": record.get("alert", {}).get("message"),
            "marh_active": record.get("marh", {}).get("active"),
            "marh_recommended": record.get("marh", {}).get("recommended"),
            "model_version": record.get("model", {}).get("version"),
        })
    return rows


def duration_seconds(events: list[dict[str, Any]]) -> float | None:
    times = []
    for event in events:
        try:
            times.append(datetime.fromisoformat(event["host_time_iso"]))
        except (KeyError, TypeError, ValueError):
            pass
    return (max(times) - min(times)).total_seconds() if len(times) >= 2 else None


def event_time(event: dict[str, Any]) -> datetime | None:
    try:
        return datetime.fromisoformat(event["host_time_iso"])
    except (KeyError, TypeError, ValueError):
        return None


def manual_trials(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for index, command_event in enumerate(events):
        command = command_event.get("command")
        if command_event.get("event") != "command_result" or command not in {
            "SAMPLE_VSN", "SAMPLE_ASN", "SAMPLE_BOTH"
        }:
            continue
        command_time = event_time(command_event)
        later = events[index + 1:]
        visual = next(
            (event for event in later if event.get("event") == "node_output" and event.get("node") == "VSN"),
            None,
        )
        acoustic = next(
            (event for event in later if event.get("event") == "node_output" and event.get("node") == "ASN"),
            None,
        )
        if command == "SAMPLE_VSN":
            acoustic = None
        elif command == "SAMPLE_ASN":
            visual = None
        fusion = None
        if command == "SAMPLE_BOTH" and visual and acoustic:
            fusion = next(
                (
                    event for event in later
                    if event.get("event") == "fusion"
                    and event.get("vsn", {}).get("sequence") == visual.get("sequence")
                    and event.get("asn", {}).get("sequence") == acoustic.get("sequence")
                ),
                None,
            )

        def delay(target: dict[str, Any] | None) -> float | None:
            target_time = event_time(target) if target else None
            if command_time is None or target_time is None:
                return None
            return (target_time - command_time).total_seconds() * 1000.0

        rows.append({
            "command_time_iso": command_event.get("host_time_iso"),
            "command": command,
            "accepted": command_event.get("success"),
            "vsn_sequence": visual.get("sequence") if visual else None,
            "asn_sequence": acoustic.get("sequence") if acoustic else None,
            "fusion_index": fusion.get("fusion_index") if fusion else None,
            "vsn_response_ms": delay(visual),
            "asn_response_ms": delay(acoustic),
            "fusion_response_ms": delay(fusion),
            "fresh_pair_fused": bool(fusion) if command == "SAMPLE_BOTH" else None,
        })
    return rows


def svg_line(path: Path, fusions: list[dict[str, Any]]) -> None:
    values = numeric(row.get("fused_risk_score") for row in fusions)
    width, height = 900, 320
    left, right, top, bottom = 70, 30, 30, 55
    plot_w, plot_h = width - left - right, height - top - bottom
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#263238;font-size:12px}.title{font-size:16px;font-weight:bold}</style>',
        '<text class="title" x="70" y="20">Hardware fused-risk timeline</text>',
    ]
    for tick in range(5):
        value = tick / 4
        y = top + (1 - value) * plot_h
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#dbe3e8"/>')
        lines.append(f'<text x="{left-38}" y="{y+4:.1f}">{value:.2f}</text>')
    if values:
        points = []
        for index, value in enumerate(values):
            x = left + (index / max(1, len(values) - 1)) * plot_w
            y = top + (1 - value) * plot_h
            points.append(f"{x:.1f},{y:.1f}")
        lines.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="#087f8c" stroke-width="2.5"/>')
        for point in points:
            x, y = point.split(",")
            lines.append(f'<circle cx="{x}" cy="{y}" r="2.5" fill="#087f8c"/>')
    lines.extend([
        f'<line x1="{left}" y1="{top+plot_h}" x2="{width-right}" y2="{top+plot_h}" stroke="#263238"/>',
        f'<text x="{width/2-45:.1f}" y="{height-15}">Fusion index</text>',
        f'<text transform="translate(18 {height/2+35:.1f}) rotate(-90)">Risk proxy score</text>',
        '</svg>',
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def svg_latency(path: Path, summary: dict[str, Any]) -> None:
    bars = [
        ("VSN inference", summary["vsn"]["inference_ms"]["mean"]),
        ("VSN pipeline", summary["vsn"]["total_ms"]["mean"]),
        ("ASN frontend", summary["asn"]["frontend_ms"]["mean"]),
        ("ASN inference", summary["asn"]["inference_ms"]["mean"]),
    ]
    bars = [(label, value) for label, value in bars if value is not None]
    width, height = 900, 330
    left, right, top, bottom = 160, 35, 45, 35
    max_value = max((value for _, value in bars), default=1.0) * 1.15
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#263238;font-size:13px}.title{font-size:16px;font-weight:bold}</style>',
        '<text class="title" x="30" y="24">Measured node processing latency</text>',
    ]
    bar_h = 42
    for index, (label, value) in enumerate(bars):
        y = top + index * 62
        bar_w = (width - left - right) * value / max_value
        lines.append(f'<text x="15" y="{y+27}">{label}</text>')
        lines.append(f'<rect x="{left}" y="{y}" width="{bar_w:.1f}" height="{bar_h}" fill="#087f8c"/>')
        lines.append(f'<text x="{left+bar_w+8:.1f}" y="{y+27}">{value:.2f} ms</text>')
    lines.append('</svg>')
    path.write_text("\n".join(lines), encoding="utf-8")


def markdown(summary: dict[str, Any], run_dir: Path) -> str:
    def metric(group: str, name: str, key: str = "mean") -> str:
        value = summary[group][name][key]
        return "n/a" if value is None else f"{value:.3f}"

    def memory(node: str, name: str) -> str:
        value = summary["memory"][node][name]["min"]
        return "n/a" if value is None else f"{int(value):,}"

    return f"""# Detailed Hardware Run Report

Run: `{run_dir.resolve()}`

## Scope

This report describes engineering behaviour observed on the physical two-node
system. Live stimuli were not labelled; the risk outputs below are therefore
operational observations, not accuracy measurements.

## Run overview

| Measure | Result |
|---|---:|
| Structured duration | {summary['duration_s'] if summary['duration_s'] is not None else 'n/a'} s |
| VSN samples | {summary['vsn']['samples']} |
| ASN samples | {summary['asn']['samples']} |
| Fusion events transmitted | {summary['fusion']['transmitted_count']} |
| Unique online fusion decisions | {summary['fusion']['count']} |
| Unique dual-node decisions | {summary['fusion']['dual_count']} |
| BLE disconnect events | {summary['ble']['disconnects']} |
| Successful host commands | {summary['commands']['successful']} / {summary['commands']['total']} |

## Timing

| Stage | Mean (ms) | p95 (ms) |
|---|---:|---:|
| VSN inference | {metric('vsn', 'inference_ms')} | {metric('vsn', 'inference_ms', 'p95')} |
| VSN camera-to-decision | {metric('vsn', 'total_ms')} | {metric('vsn', 'total_ms', 'p95')} |
| ASN log-Mel frontend | {metric('asn', 'frontend_ms')} | {metric('asn', 'frontend_ms', 'p95')} |
| ASN inference | {metric('asn', 'inference_ms')} | {metric('asn', 'inference_ms', 'p95')} |
| ASN capture window | {metric('asn', 'capture_ms')} | {metric('asn', 'capture_ms', 'p95')} |
| BLE decision-summary RTT | {metric('ble', 'summary_rtt_ms')} | {metric('ble', 'summary_rtt_ms', 'p95')} |

## Observed free memory

| Node | Minimum free internal heap (bytes) | Minimum free PSRAM (bytes) | Snapshots |
|---|---:|---:|---:|
| VSN | {memory('VSN', 'internal_free_bytes')} | {memory('VSN', 'psram_free_bytes')} | {summary['memory']['VSN']['snapshots']} |
| ASN | {memory('ASN', 'internal_free_bytes')} | {memory('ASN', 'psram_free_bytes')} | {summary['memory']['ASN']['snapshots']} |

## Risk outputs

| Measure | Mean | p95 |
|---|---:|---:|
| VSN risk proxy | {metric('vsn', 'risk_score')} | {metric('vsn', 'risk_score', 'p95')} |
| ASN risk proxy | {metric('asn', 'risk_score')} | {metric('asn', 'risk_score', 'p95')} |
| Fused risk proxy | {metric('fusion', 'risk_score')} | {metric('fusion', 'risk_score', 'p95')} |

Risk-level counts: `{summary['fusion']['risk_levels']}`.

## Evidence generated

- `vsn_samples.csv`: one row per visual decision.
- `asn_samples.csv`: microphone acquisition, signal and inference fields.
- `fusion_outputs.csv`: inputs, confidence-aware result, level, trend and alert.
- `ble_events.csv`: discovery, connection and decision acknowledgement evidence.
- `model_exchange.csv`: shared-head transfer and runtime installation evidence.
- `memory_snapshots.csv`: device heap counters by execution stage.
- `manual_control_trials.csv`: command-to-node/fusion latency and sequence freshness.
- `figures/risk_timeline.svg` and `figures/latency_summary.svg`: report-ready vector plots.

## Interpretation boundary

These files support latency, memory, communication, control-flow and functional
integration claims. Accuracy, sensitivity and false-positive claims require a
separately labelled controlled-scene protocol. Energy also requires external
power instrumentation and is not inferred from software counters.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir
    output = run_dir / "analysis"
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    events = read_jsonl(run_dir / "events.jsonl")
    node_rows = read_csv(run_dir / "node_outputs.csv")
    fusions = flatten_fusions(events, read_csv(run_dir / "fusion.csv"))
    vsn_raw, vsn_link, model_rows = parse_vsn_raw(run_dir / "vsn_serial_raw.log")
    asn_rows, asn_link, asn_memory = parse_asn_raw(run_dir / "asn_serial_debug.log")

    vsn_rows = [row for row in node_rows if row.get("node") == "VSN"] or vsn_raw
    if not asn_rows:
        asn_rows = [row for row in node_rows if row.get("node") == "ASN"]
    ble_rows = vsn_link + asn_link
    memory_rows = parse_memory(run_dir / "vsn_serial_raw.log", "VSN") + asn_memory
    trials = manual_trials(events)

    seen_fusion_indices: set[str] = set()
    unique_fusions = []
    for row in fusions:
        fusion_index = str(row.get("fusion_index"))
        row["record_role"] = "status_replay" if fusion_index in seen_fusion_indices else "online_decision"
        if fusion_index not in seen_fusion_indices:
            seen_fusion_indices.add(fusion_index)
            unique_fusions.append(row)

    node_fields = [
        "host_time_iso", "node", "sequence", "index", "risk_score", "confidence", "status",
        "samples", "elapsed_ms", "rms", "min", "max", "validity", "validity_index",
        "capture_ms", "decode_ms", "preprocess_ms", "frontend_ms", "inference_ms", "total_ms",
        "head_version", "generation", "format", "threshold",
    ]
    write_csv(output / "vsn_samples.csv", vsn_rows, node_fields)
    write_csv(output / "asn_samples.csv", asn_rows, node_fields)
    fusion_fields = list(fusions[0].keys()) if fusions else ["host_time_iso", "fusion_index"]
    write_csv(output / "fusion_outputs.csv", fusions, fusion_fields)
    ble_fields = sorted({key for row in ble_rows for key in row}) or ["host_time_iso", "event"]
    write_csv(output / "ble_events.csv", ble_rows, ble_fields)
    model_fields = sorted({key for row in model_rows for key in row}) or ["host_time_iso", "operation"]
    write_csv(output / "model_exchange.csv", model_rows, model_fields)
    memory_fields = sorted({key for row in memory_rows for key in row}) or ["host_time_iso", "node", "stage"]
    write_csv(output / "memory_snapshots.csv", memory_rows, memory_fields)
    write_csv(
        output / "manual_control_trials.csv",
        trials,
        [
            "command_time_iso", "command", "accepted", "vsn_sequence", "asn_sequence",
            "fusion_index", "vsn_response_ms", "asn_response_ms", "fusion_response_ms",
            "fresh_pair_fused",
        ],
    )

    command_events = [event for event in events if event.get("event") == "command_result"]
    ble_events = [event for event in events if event.get("event") == "ble_state"]
    memory_summary = {}
    for node in ("VSN", "ASN"):
        rows = [row for row in memory_rows if row.get("node") == node]
        memory_summary[node] = {
            "snapshots": len(rows),
            "internal_free_bytes": describe(row.get("internal_free") for row in rows),
            "internal_largest_bytes": describe(row.get("internal_largest") for row in rows),
            "psram_free_bytes": describe(row.get("psram_free") for row in rows),
        }
    summary = {
        "run_dir": str(run_dir.resolve()),
        "duration_s": duration_seconds(events),
        "vsn": {
            "samples": len(vsn_rows),
            "risk_score": describe(row.get("risk_score") for row in vsn_rows),
            "confidence": describe(row.get("confidence") for row in vsn_rows),
            "inference_ms": describe(row.get("inference_ms") for row in vsn_rows),
            "total_ms": describe(row.get("total_ms") for row in vsn_rows),
        },
        "asn": {
            "samples": len(asn_rows),
            "risk_score": describe(row.get("risk_score") for row in asn_rows),
            "confidence": describe(row.get("confidence") for row in asn_rows),
            "capture_ms": describe(row.get("elapsed_ms", row.get("capture_ms")) for row in asn_rows),
            "frontend_ms": describe(row.get("frontend_ms") for row in asn_rows),
            "inference_ms": describe(row.get("inference_ms") for row in asn_rows),
        },
        "fusion": {
            "transmitted_count": len(fusions),
            "count": len(unique_fusions),
            "dual_count": sum(str(row.get("mode", "")).lower() == "dual" for row in unique_fusions),
            "risk_score": describe(row.get("fused_risk_score") for row in unique_fusions),
            "confidence": describe(row.get("fused_confidence") for row in unique_fusions),
            "risk_levels": dict(Counter(str(row.get("risk_level")) for row in unique_fusions)),
            "alert_actions": dict(Counter(str(row.get("alert_action")) for row in unique_fusions)),
        },
        "ble": {
            "connects": sum(bool(event.get("connected")) for event in ble_events),
            "disconnects": sum(not bool(event.get("connected")) for event in ble_events),
            "summary_rtt_ms": describe(row.get("rtt_ms") for row in asn_link),
            "discovery_rssi_dbm": describe(
                (row.get("rssi") for row in vsn_link), allow_negative=True
            ),
        },
        "commands": {
            "total": len(command_events),
            "successful": sum(bool(event.get("success")) for event in command_events),
            "results": [
                {"command": event.get("command"), "success": event.get("success"), "detail": event.get("detail")}
                for event in command_events
            ],
        },
        "model_exchange_records": len(model_rows),
        "memory_snapshots": len(memory_rows),
        "memory": memory_summary,
        "manual_control_trials": trials,
        "claims": {
            "supports": ["functional integration", "latency", "software memory counters", "BLE behaviour", "manual control"],
            "does_not_support": ["accuracy on live stimuli", "collapse probability", "energy without instrumentation"],
        },
    }
    (output / "detailed_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output / "detailed_summary.md").write_text(markdown(summary, run_dir), encoding="utf-8")
    svg_line(figures / "risk_timeline.svg", unique_fusions)
    svg_latency(figures / "latency_summary.svg", summary)
    print(f"Detailed evidence: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
