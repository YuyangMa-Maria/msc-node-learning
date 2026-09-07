"""Reference implementation of status-aware node-message fusion.

This module mirrors the decision policy used by the representative VSN. A
node's risk score describes the current observation, whereas ``status`` says
whether that observation is admissible. Keeping the two concepts separate is
essential when a sensor is offline, invalid or degraded.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import mean
from typing import Iterable, Literal

NodeStatus = Literal[
    "normal", "warning", "critical", "degraded", "invalid", "low_power", "offline"
]
RiskTrend = Literal["rising", "falling", "stable", "unknown"]
AlertSeverity = Literal["info", "advisory", "caution", "high", "critical", "unavailable"]


@dataclass(frozen=True)
class NodeMessage:
    """Compact modality-independent output produced by one sensing node."""

    node_id: str
    timestamp: str
    risk_score: float
    confidence: float
    status: NodeStatus
    modality: str = "unknown"


@dataclass(frozen=True)
class FusionResult:
    """Region-level decision and the fixed-schema warning derived from it."""

    timestamp: str
    fused_risk_score: float
    confidence: float
    risk_level: int
    risk_trend: RiskTrend
    triggered_nodes: list[str]
    alert_attention_required: bool
    alert_severity: AlertSeverity
    alert_headline: str
    alert_evidence: str
    alert_action: str
    alert_message: str
    method: str


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def parse_message(raw: dict[str, object]) -> NodeMessage:
    """Convert a decoded wire/message object into the internal typed contract."""
    metadata = raw.get("metadata", {})
    modality = "unknown"
    if isinstance(metadata, dict):
        modality = str(metadata.get("modality", modality))
    return NodeMessage(
        node_id=str(raw["node_id"]),
        timestamp=str(raw["timestamp"]),
        risk_score=clamp(float(raw["risk_score"])),
        confidence=clamp(float(raw["confidence"])),
        status=str(raw.get("status", "normal")),  # type: ignore[arg-type]
        modality=modality,
    )


def effective_weight(message: NodeMessage) -> float:
    """Convert confidence and health state into a fusion weight."""
    # Status is a reliability gate, separate from the model's risk estimate.
    if message.status in {"offline", "invalid"}:
        return 0.0
    if message.status in {"low_power", "degraded"}:
        return message.confidence * 0.5
    if message.status == "critical":
        return max(message.confidence, 0.8)
    return message.confidence


def confidence_weighted_score(messages: Iterable[NodeMessage]) -> tuple[float, float]:
    """Average admissible risk scores using their effective reliability."""
    active = [message for message in messages if message.status != "offline"]
    if not active:
        return 0.0, 0.0
    weights = [effective_weight(message) for message in active]
    total = sum(weights)
    if total <= 1e-9:
        return mean(message.risk_score for message in active), 0.0
    score = sum(weight * message.risk_score for weight, message in zip(weights, active)) / total
    confidence = clamp(total / max(len(active), 1))
    return clamp(score), confidence


def max_risk_score(messages: Iterable[NodeMessage]) -> tuple[float, float]:
    """Return the strongest reliability-adjusted item of evidence."""
    active = [message for message in messages if message.status != "offline"]
    if not active:
        return 0.0, 0.0
    selected = max(active, key=lambda message: message.risk_score * max(effective_weight(message), 1e-6))
    return selected.risk_score, effective_weight(selected)


def quality_gate_score(messages: Iterable[NodeMessage]) -> tuple[float, float]:
    """Apply hard validity gates before combining nodes in the best status band."""
    # Preserve critical evidence; otherwise combine nodes in the same status band.
    active = [
        message
        for message in messages
        if message.status not in {"offline", "invalid", "low_power"}
    ]
    critical = [message for message in active if message.status == "critical"]
    if critical:
        return max_risk_score(critical)
    warning = [message for message in active if message.status == "warning"]
    if warning:
        return confidence_weighted_score(warning)
    return confidence_weighted_score(active)


def quality_filtered_max_score(messages: Iterable[NodeMessage], min_confidence: float = 0.5) -> tuple[float, float]:
    reliable = [
        message
        for message in messages
        if message.status not in {"offline", "low_power"} and message.confidence >= min_confidence
    ]
    if reliable:
        return max_risk_score(reliable)
    return confidence_weighted_score(messages)


def risk_level(score: float) -> int:
    """Map the continuous proxy to the prototype's five reporting bands."""
    # Prototype decision bands, not probabilities of structural failure.
    score = clamp(score)
    if score < 0.20:
        return 1
    if score < 0.40:
        return 2
    if score < 0.60:
        return 3
    if score < 0.80:
        return 4
    return 5


def risk_trend(history: list[float], current: float, min_delta: float = 0.05) -> RiskTrend:
    """Compare the current score with a short recent mean."""
    # A short mean suppresses isolated spikes without hiding a sustained rise.
    if len(history) < 2:
        return "unknown"
    previous = mean(history[-3:]) if len(history) >= 3 else mean(history)
    delta = current - previous
    if delta > min_delta:
        return "rising"
    if delta < -min_delta:
        return "falling"
    return "stable"


def triggered_nodes(messages: Iterable[NodeMessage], threshold: float = 0.60) -> list[str]:
    """Identify admissible nodes that supplied explicit abnormal evidence."""
    return [
        message.node_id
        for message in messages
        if message.status not in {"offline", "invalid", "low_power"}
        and (
            message.status in {"warning", "critical"}
            or (message.risk_score >= threshold and message.confidence >= 0.5)
        )
    ]


def alert_decision(
    level: int,
    trend: RiskTrend,
    triggered: list[str],
    messages: list[NodeMessage],
) -> tuple[bool, AlertSeverity, str, str, str, str]:
    """Derive a deterministic warning without treating the proxy as certification."""
    active = [
        message
        for message in messages
        if message.status not in {"offline", "invalid", "low_power"}
    ]
    evidence_nodes = triggered if triggered else [message.node_id for message in active]
    evidence = "+".join(evidence_nodes) if evidence_nodes else "none"
    limited_coverage = len(active) < len(messages) or any(
        message.status == "degraded" for message in active
    )

    if not active:
        return (
            True,
            "unavailable",
            "Insufficient sensor evidence",
            evidence,
            "insufficient_evidence",
            "No valid sensor evidence is available. Do not interpret this as low risk; repeat sampling or request external assessment.",
        )

    if level >= 5:
        corroborated = len(triggered) >= 2
        return (
            True,
            "critical",
            "Critical corroborated risk" if corroborated else "Critical local risk",
            evidence,
            "move_away_multi_node" if corroborated else "move_away",
            (
                "Critical corroborated risk proxy from multiple nodes. Move away from the monitored area and await professional assessment."
                if corroborated
                else f"Critical risk proxy from {evidence} evidence. Move away from the monitored area and await professional assessment."
            ),
        )
    if level == 4:
        return (
            True,
            "high",
            "High local risk",
            evidence,
            "avoid_entry",
            f"High risk proxy from {evidence} evidence. Avoid entering the monitored area and request further assessment.",
        )
    if level == 3:
        rising = trend == "rising"
        return (
            True,
            "caution",
            "Risk proxy is rising" if rising else "Moderate local risk",
            evidence,
            "keep_distance_and_monitor" if rising else "maintain_caution",
            (
                f"Moderate and rising risk proxy from {evidence} evidence. Keep distance, repeat sampling, and request further assessment."
                if rising
                else f"Moderate risk proxy from {evidence} evidence. Maintain caution and continue monitoring."
            ),
        )

    if limited_coverage:
        return (
            True,
            "advisory",
            "Reduced sensor coverage",
            evidence,
            "continue_monitoring_limited_evidence",
            "Available evidence indicates a low current risk proxy, but sensor coverage is limited. Continue monitoring and repeat sampling.",
        )
    return (
        False,
        "info",
        "Low local risk",
        evidence,
        "continue_monitoring",
        "Current local risk proxy is low. Continue monitoring.",
    )


def alert_text(level: int, trend: RiskTrend, triggered: list[str]) -> str:
    """Retain the original text-only helper for earlier experiment scripts."""
    if level <= 2:
        return "Current local structural risk is low. Continue monitoring."
    if level == 3:
        if trend == "rising":
            return "Current local structural risk is moderate and rising. Keep distance and continue monitoring."
        return "Current local structural risk is moderate. Maintain caution and continue monitoring."
    if level == 4:
        return "Current local structural risk is high. Avoid entering the area and wait for further assessment."
    if len(triggered) >= 2:
        return "Current local structural risk is critical. Multiple nodes report abnormal evidence. Move away immediately and request further assessment."
    return "Current local structural risk is critical. Move away immediately and request further assessment."


def fuse_messages(
    raw_messages: Iterable[dict[str, object] | NodeMessage],
    history: list[float] | None = None,
    method: Literal["confidence_weighted", "max_risk", "quality_gate", "quality_filtered_max"] = "quality_gate",
    timestamp: str | None = None,
) -> FusionResult:
    """Fuse one decision window and derive its level, trend and warning fields."""
    messages = [parse_message(message) if isinstance(message, dict) else message for message in raw_messages]
    history = history or []
    if method == "confidence_weighted":
        score, confidence = confidence_weighted_score(messages)
    elif method == "max_risk":
        score, confidence = max_risk_score(messages)
    elif method == "quality_gate":
        score, confidence = quality_gate_score(messages)
    elif method == "quality_filtered_max":
        score, confidence = quality_filtered_max_score(messages)
    else:
        raise ValueError(f"Unsupported fusion method: {method}")
    level = risk_level(score)
    trend = risk_trend(history, score)
    triggered = triggered_nodes(messages)
    attention, severity, headline, evidence, action, message = alert_decision(
        level, trend, triggered, messages
    )
    return FusionResult(
        timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
        fused_risk_score=score,
        confidence=clamp(confidence),
        risk_level=level,
        risk_trend=trend,
        triggered_nodes=triggered,
        alert_attention_required=attention,
        alert_severity=severity,
        alert_headline=headline,
        alert_evidence=evidence,
        alert_action=action,
        alert_message=message,
        method=method,
    )


def result_to_dict(result: FusionResult) -> dict[str, object]:
    return {
        "timestamp": result.timestamp,
        "fused_risk_score": result.fused_risk_score,
        "confidence": result.confidence,
        "risk_level": result.risk_level,
        "risk_trend": result.risk_trend,
        "triggered_nodes": result.triggered_nodes,
        "alert_attention_required": result.alert_attention_required,
        "alert_severity": result.alert_severity,
        "alert_headline": result.alert_headline,
        "alert_evidence": result.alert_evidence,
        "alert_action": result.alert_action,
        "alert_message": result.alert_message,
        "method": result.method,
    }
