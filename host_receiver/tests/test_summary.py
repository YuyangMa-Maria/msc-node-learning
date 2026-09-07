"""Regression tests for BLE outage and recovery timing summaries."""

import unittest

from summarise_run import calculate_ble_outages


class SummaryTests(unittest.TestCase):
    def test_calculates_reconnect_and_first_dual_delays(self):
        events = [
            {
                "host_time_iso": "2026-08-16T12:00:00+00:00",
                "event": "ble_state",
                "connected": False,
                "reason": 520,
            },
            {
                "host_time_iso": "2026-08-16T12:00:10.500+00:00",
                "event": "ble_state",
                "connected": True,
                "reason": 0,
            },
            {
                "host_time_iso": "2026-08-16T12:00:16.125+00:00",
                "event": "fusion",
                "mode": "dual",
            },
        ]
        outage = calculate_ble_outages(events)[0]
        self.assertTrue(outage["recovered"])
        self.assertEqual(outage["reconnect_delay_s"], 10.5)
        self.assertEqual(outage["recovery_to_dual_s"], 16.125)
        self.assertEqual(outage["post_connect_to_dual_s"], 5.625)


if __name__ == "__main__":
    unittest.main()
