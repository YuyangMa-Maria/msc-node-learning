"""Parser and state helpers for the VSN external-receiver serial protocol.

Only lines prefixed with ``NLJSON`` belong to the machine-readable interface;
ESP-IDF diagnostics may share the same serial stream. Fusion and alert fields
are validated here, but are never recomputed by the receiver.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


PREFIX = "NLJSON "


class ProtocolError(ValueError):
    """Raised when a prefixed record violates the versioned host contract."""


def parse_event(line: str) -> dict[str, Any] | None:
    """Parse one serial line, returning ``None`` for ordinary firmware logs."""
    marker = line.find(PREFIX)
    if marker < 0:
        return None
    payload = line[marker + len(PREFIX) :].strip()
    try:
        event = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"invalid JSON: {exc.msg}") from exc
    if not isinstance(event, dict):
        raise ProtocolError("event payload must be a JSON object")
    if event.get("schema") != 1:
        raise ProtocolError(f"unsupported schema: {event.get('schema')!r}")
    if not isinstance(event.get("event"), str):
        raise ProtocolError("event name is missing")
    if event["event"] == "fusion":
        _validate_fusion_event(event)
    elif event["event"] == "node_output":
        required = {
            "uptime_ms", "node", "phase", "source", "transport", "sequence",
            "risk_score", "confidence", "status", "latency_ms",
        }
        missing = sorted(required.difference(event))
        if missing:
            raise ProtocolError(f"node_output event is missing fields: {', '.join(missing)}")
        if event["node"] not in {"VSN", "ASN", "VBN"}:
            raise ProtocolError(f"unsupported node: {event['node']!r}")
    return event


def _require_mapping(event: dict[str, Any], key: str) -> dict[str, Any]:
    value = event.get(key)
    if not isinstance(value, dict):
        raise ProtocolError(f"fusion event section {key!r} is missing")
    return value


def _validate_fusion_event(event: dict[str, Any]) -> None:
    """Reject incomplete events before they can reach logs or the dashboard."""
    required_top_level = {"uptime_ms", "fusion_index", "mode"}
    missing = sorted(required_top_level.difference(event))
    if missing:
        raise ProtocolError(f"fusion event is missing fields: {', '.join(missing)}")

    required_sections = {
        "vsn": {"sequence", "risk_score", "confidence", "status"},
        "asn": {
            "available", "sequence", "age_ms", "risk_score", "confidence", "status"
        },
        "fusion": {
            "risk_score", "confidence", "risk_level", "risk_trend", "active_nodes"
        },
        "alert": {
            "attention_required", "severity", "headline", "triggered_mask",
            "evidence", "action", "message",
        },
        "marh": {"active", "recommended", "reason"},
        "model": {"version", "generation", "format"},
    }
    for section_name, required_fields in required_sections.items():
        section = _require_mapping(event, section_name)
        missing = sorted(required_fields.difference(section))
        if missing:
            raise ProtocolError(
                f"fusion event section {section_name!r} is missing fields: "
                + ", ".join(missing)
            )

    alert = event["alert"]
    if not isinstance(alert["attention_required"], bool):
        raise ProtocolError("alert attention_required must be a Boolean")
    allowed_severities = {
        "info", "advisory", "caution", "high", "critical", "unavailable"
    }
    if alert["severity"] not in allowed_severities:
        raise ProtocolError(f"unsupported alert severity: {alert['severity']!r}")
    for field_name in ("headline", "evidence", "action", "message"):
        value = alert[field_name]
        if not isinstance(value, str) or not value.strip():
            raise ProtocolError(f"alert {field_name} must be a non-empty string")
    if not isinstance(alert["triggered_mask"], int) or alert["triggered_mask"] < 0:
        raise ProtocolError("alert triggered_mask must be a non-negative integer")


@dataclass
class ReceiverState:
    """Latest host-visible state reconstructed from validated device events."""

    boot: dict[str, Any] | None = None
    ble_connected: bool = False
    marh_active: bool = False
    latest_fusion: dict[str, Any] | None = None
    latest_nodes: dict[str, dict[str, Any]] = field(default_factory=dict)
    node_states: dict[str, dict[str, Any]] = field(default_factory=dict)
    # A host can attach after the devices are already running, so their current
    # mode is unknown until an explicit MODE command is acknowledged.
    sampling_mode: str = "unknown"
    model_version: int | None = None
    event_counts: dict[str, int] = field(default_factory=dict)

    def update(self, event: dict[str, Any]) -> None:
        """Apply one event without inferring decisions absent from the wire data."""
        name = event["event"]
        self.event_counts[name] = self.event_counts.get(name, 0) + 1
        if name == "boot":
            self.boot = event
        elif name == "ble_state":
            self.ble_connected = bool(event.get("connected"))
        elif name == "marh_state":
            self.marh_active = bool(event.get("active"))
        elif name == "fusion":
            self.latest_fusion = event
            self.marh_active = bool(event.get("marh", {}).get("active"))
            self.model_version = event.get("model", {}).get("version")
        elif name == "node_output":
            self.latest_nodes[event["node"]] = event
            if event["node"] == "ASN" and event.get("transport") == "ble_summary":
                # A receiver can attach after the original BLE connection
                # event. A valid ASN summary is itself proof of a live link.
                self.ble_connected = True
        elif name == "node_state":
            nodes = event.get("nodes", {})
            if isinstance(nodes, dict):
                self.node_states = nodes
                self.ble_connected = bool(nodes.get("ASN", {}).get("online"))
        elif name == "sampling_state":
            self.sampling_mode = str(event.get("mode", self.sampling_mode))
        elif name == "command_result":
            self.model_version = event.get("model", {}).get("version", self.model_version)


def format_event(event: dict[str, Any]) -> str:
    """Render a compact terminal line for an already validated event."""
    name = event["event"]
    if name == "fusion":
        fusion = event.get("fusion", {})
        vsn = event.get("vsn", {})
        asn = event.get("asn", {})
        alert = event.get("alert", {})
        marh = event.get("marh", {})
        asn_text = (
            f"{asn.get('risk_score', 0):.3f}/{asn.get('status', '?')}"
            if asn.get("available")
            else "offline"
        )
        recommendation = (
            f" | MARH suggested: {marh.get('reason')}"
            if marh.get("recommended")
            else ""
        )
        return (
            f"[FUSION {event.get('fusion_index')}] "
            f"VSN {vsn.get('risk_score', 0):.3f}/{vsn.get('status', '?')} | "
            f"ASN {asn_text} -> risk {fusion.get('risk_score', 0):.3f} "
            f"L{fusion.get('risk_level')} {fusion.get('risk_trend')} | "
            f"{str(alert.get('severity', 'info')).upper()}: "
            f"{alert.get('headline')} | {alert.get('action')}{recommendation}"
        )
    if name == "ble_state":
        return (
            f"[BLE] ASN {'connected' if event.get('connected') else 'disconnected'} "
            f"(connections={event.get('connection_count')}, reason={event.get('reason')})"
        )
    if name == "boot":
        return (
            "[BOOT] VSN camera={camera_ready} model={model_ready} BLE={ble_ready}"
        ).format(**event)
    if name == "marh_state":
        return f"[MARH] {'active' if event.get('active') else 'inactive'} ({event.get('source')})"
    if name == "command_result":
        return (
            f"[COMMAND] {event.get('command')} "
            f"{'PASS' if event.get('success') else 'FAIL'}: {event.get('detail')}"
        )
    if name == "node_output":
        return (
            f"[{event.get('node')} OUTPUT {event.get('sequence')}] "
            f"risk={event.get('risk_score', 0):.3f} "
            f"confidence={event.get('confidence', 0):.3f} "
            f"status={event.get('status')} via {event.get('transport')}"
        )
    if name == "sampling_state":
        return (
            f"[SAMPLING] mode={event.get('mode')} "
            f"ASN synchronised={event.get('asn_synchronised')}"
        )
    if name == "node_state":
        nodes = event.get("nodes", {})
        return "[NODES] " + ", ".join(
            f"{name}={'online' if value.get('online') else 'offline'}"
            for name, value in nodes.items()
        )
    return f"[{name.upper()}] {event}"


def fusion_csv_row(event: dict[str, Any], host_time_iso: str) -> dict[str, Any]:
    """Flatten a fusion event while retaining device and host timestamps."""
    vsn = event["vsn"]
    asn = event["asn"]
    fusion = event["fusion"]
    alert = event["alert"]
    marh = event["marh"]
    model = event["model"]
    return {
        "host_time_iso": host_time_iso,
        "device_uptime_ms": event["uptime_ms"],
        "fusion_index": event["fusion_index"],
        "mode": event["mode"],
        "vsn_sequence": vsn["sequence"],
        "vsn_risk_score": vsn["risk_score"],
        "vsn_confidence": vsn["confidence"],
        "vsn_status": vsn["status"],
        "asn_available": asn["available"],
        "asn_sequence": asn["sequence"],
        "asn_age_ms": asn["age_ms"],
        "asn_risk_score": asn["risk_score"],
        "asn_confidence": asn["confidence"],
        "asn_status": asn["status"],
        "fused_risk_score": fusion["risk_score"],
        "fused_confidence": fusion["confidence"],
        "risk_level": fusion["risk_level"],
        "risk_trend": fusion["risk_trend"],
        "active_nodes": fusion["active_nodes"],
        "alert_attention_required": alert["attention_required"],
        "alert_severity": alert["severity"],
        "alert_headline": alert["headline"],
        "alert_evidence": alert["evidence"],
        "alert_action": alert["action"],
        "alert_message": alert["message"],
        "triggered_mask": alert["triggered_mask"],
        "marh_active": marh["active"],
        "marh_recommended": marh["recommended"],
        "marh_reason": marh["reason"],
        "model_version": model["version"],
        "model_generation": model["generation"],
        "model_format": model["format"],
    }
