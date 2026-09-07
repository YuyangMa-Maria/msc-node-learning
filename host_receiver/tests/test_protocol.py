"""Contract tests for parsing, validating and storing VSN host events."""

import json
import tempfile
import unittest
from pathlib import Path

from protocol import ProtocolError, ReceiverState, fusion_csv_row, parse_event
from receiver import load_command_plan


def fusion_event():
    return {
        "schema": 1,
        "event": "fusion",
        "uptime_ms": 12000,
        "fusion_index": 7,
        "mode": "dual",
        "vsn": {"sequence": 12, "risk_score": 0.7, "confidence": 0.8, "status": "warning"},
        "asn": {"available": True, "sequence": 4, "age_ms": 25, "risk_score": 0.4, "confidence": 0.6, "status": "warning"},
        "fusion": {"risk_score": 0.58, "confidence": 0.7, "risk_level": 3, "risk_trend": "rising", "active_nodes": 2, "method": "quality_gate"},
        "alert": {
            "attention_required": True,
            "severity": "caution",
            "headline": "Risk proxy is rising",
            "triggered_mask": 3,
            "evidence": "VSN+ASN",
            "action": "keep_distance_and_monitor",
            "message": "test",
        },
        "marh": {"active": False, "recommended": True, "reason": "rising_risk"},
        "model": {"version": 514, "generation": 2, "format": "INT8"},
    }


class ProtocolTests(unittest.TestCase):
    def test_ignores_non_protocol_log(self):
        self.assertIsNone(parse_event("I (123) ordinary log"))

    def test_parses_prefixed_event(self):
        expected = fusion_event()
        actual = parse_event("I (1) x NLJSON " + json.dumps(expected))
        self.assertEqual(actual, expected)

    def test_rejects_invalid_schema(self):
        with self.assertRaises(ProtocolError):
            parse_event('NLJSON {"schema":2,"event":"boot"}')

    def test_rejects_incomplete_fusion(self):
        with self.assertRaises(ProtocolError):
            parse_event('NLJSON {"schema":1,"event":"fusion","uptime_ms":1}')

    def test_rejects_invalid_alert_contract(self):
        event = fusion_event()
        event["alert"]["severity"] = "danger"
        with self.assertRaises(ProtocolError):
            parse_event("NLJSON " + json.dumps(event))

        event = fusion_event()
        event["alert"]["headline"] = ""
        with self.assertRaises(ProtocolError):
            parse_event("NLJSON " + json.dumps(event))

    def test_state_and_csv(self):
        event = fusion_event()
        state = ReceiverState()
        state.update(event)
        self.assertEqual(state.model_version, 514)
        self.assertTrue(state.latest_fusion["marh"]["recommended"])
        row = fusion_csv_row(event, "2026-08-16T12:00:00+00:00")
        self.assertEqual(row["fused_risk_score"], 0.58)
        self.assertEqual(row["asn_status"], "warning")
        self.assertTrue(row["alert_attention_required"])
        self.assertEqual(row["alert_evidence"], "VSN+ASN")

    def test_node_output_and_sampling_state(self):
        node = {
            "schema": 1,
            "event": "node_output",
            "uptime_ms": 4000,
            "node": "VSN",
            "phase": "local_decision",
            "source": "camera",
            "transport": "local",
            "sequence": 9,
            "source_uptime_ms": 4000,
            "received_uptime_ms": 4000,
            "risk_score": 0.62,
            "confidence": 0.74,
            "status": "warning",
            "latency_ms": {"capture": 0.7, "decode": 13.2, "preprocess": 4.5, "inference": 15.1, "total": 34.0},
        }
        state = ReceiverState()
        state.update(parse_event("NLJSON " + json.dumps(node)))
        state.update(parse_event('NLJSON {"schema":1,"event":"sampling_state","mode":"manual"}'))
        self.assertEqual(state.latest_nodes["VSN"]["sequence"], 9)
        self.assertEqual(state.sampling_mode, "manual")

        acoustic = dict(node, node="ASN", transport="ble_summary", sequence=10)
        state.update(parse_event("NLJSON " + json.dumps(acoustic)))
        self.assertTrue(state.ble_connected)

    def test_restart_command_plan_is_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            path.write_text(
                '[{"at_s": 2, "command": "SYSTEM RESTART"}]',
                encoding="utf-8",
            )
            self.assertEqual(
                load_command_plan(path),
                [{"at_s": 2.0, "command": "SYSTEM RESTART"}],
            )

    def test_manual_sampling_commands_are_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            path.write_text(
                '[{"at_s":1,"command":"MODE MANUAL"},'
                '{"at_s":2,"command":"SAMPLE BOTH"}]',
                encoding="utf-8",
            )
            self.assertEqual(
                load_command_plan(path),
                [
                    {"at_s": 1.0, "command": "MODE MANUAL"},
                    {"at_s": 2.0, "command": "SAMPLE BOTH"},
                ],
            )


if __name__ == "__main__":
    unittest.main()
