"""Behavioural tests for the reference decision-fusion implementation."""

import unittest

from nlrisk.fusion.message_fusion import NodeMessage, fuse_messages, risk_level


class MessageFusionTests(unittest.TestCase):
    def test_offline_node_has_no_vote(self) -> None:
        messages = [
            NodeMessage("VSN", "t0", 0.20, 0.80, "normal", "visual"),
            NodeMessage("ASN", "t0", 0.95, 0.99, "offline", "acoustic"),
        ]
        result = fuse_messages(messages, method="confidence_weighted", timestamp="t0")
        self.assertAlmostEqual(result.fused_risk_score, 0.20)

    def test_quality_gate_preserves_critical_evidence(self) -> None:
        messages = [
            NodeMessage("VSN", "t0", 0.30, 0.90, "normal", "visual"),
            NodeMessage("ASN", "t0", 0.88, 0.70, "critical", "acoustic"),
        ]
        result = fuse_messages(messages, method="quality_gate", timestamp="t0")
        self.assertAlmostEqual(result.fused_risk_score, 0.88)
        self.assertEqual(result.risk_level, 5)
        self.assertEqual(result.alert_severity, "critical")
        self.assertEqual(result.alert_evidence, "ASN")
        self.assertEqual(result.alert_action, "move_away")

    def test_rising_history_changes_alert_wording(self) -> None:
        messages = [NodeMessage("VSN", "t1", 0.55, 0.80, "warning", "visual")]
        result = fuse_messages(messages, history=[0.20, 0.25, 0.30], timestamp="t1")
        self.assertEqual(result.risk_trend, "rising")
        self.assertIn("rising", result.alert_message)
        self.assertTrue(result.alert_attention_required)

    def test_invalid_scores_do_not_become_warning_evidence(self) -> None:
        messages = [
            NodeMessage("VSN", "t0", 0.99, 0.99, "invalid", "visual"),
            NodeMessage("ASN", "t0", 0.95, 0.99, "invalid", "acoustic"),
        ]
        result = fuse_messages(messages, method="quality_gate", timestamp="t0")
        self.assertEqual(result.fused_risk_score, 0.0)
        self.assertEqual(result.triggered_nodes, [])
        self.assertEqual(result.alert_severity, "unavailable")
        self.assertEqual(result.alert_action, "insufficient_evidence")

    def test_low_risk_with_missing_node_is_advisory(self) -> None:
        messages = [
            NodeMessage("VSN", "t0", 0.10, 0.90, "normal", "visual"),
            NodeMessage("ASN", "t0", 0.90, 0.90, "offline", "acoustic"),
        ]
        result = fuse_messages(messages, method="quality_gate", timestamp="t0")
        self.assertEqual(result.risk_level, 1)
        self.assertEqual(result.alert_severity, "advisory")
        self.assertTrue(result.alert_attention_required)
        self.assertEqual(result.alert_evidence, "VSN")

    def test_active_nodes_remain_evidence_below_trigger_threshold(self) -> None:
        messages = [
            NodeMessage("VSN", "t0", 0.45, 0.90, "normal", "visual"),
            NodeMessage("ASN", "t0", 0.50, 0.80, "normal", "acoustic"),
        ]
        result = fuse_messages(messages, method="quality_gate", timestamp="t0")
        self.assertEqual(result.risk_level, 3)
        self.assertEqual(result.triggered_nodes, [])
        self.assertEqual(result.alert_evidence, "VSN+ASN")

    def test_risk_levels_cover_unit_interval(self) -> None:
        self.assertEqual([risk_level(x) for x in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)], [1, 2, 3, 4, 5, 5])


if __name__ == "__main__":
    unittest.main()
