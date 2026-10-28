import json
import tempfile
import unittest
from pathlib import Path

from recorder import assign_round_trip, split_round_trip


class RoundTripTests(unittest.TestCase):
    def test_split_preserves_steering_at_turnaround(self):
        outbound, returning = split_round_trip(
            [(0.0, ""), (2.0, "A"), (7.0, ""), (11.0, "D")],
            end_time=16.0,
            turnaround_time=9.0,
        )

        self.assertEqual(outbound, [
            {"steer": "", "seconds": 2.0},
            {"steer": "A", "seconds": 5.0},
            {"steer": "", "seconds": 2.0},
        ])
        self.assertEqual(returning, [
            {"steer": "", "seconds": 2.0},
            {"steer": "D", "seconds": 5.0},
        ])
        self.assertAlmostEqual(
            sum(segment["seconds"] for segment in outbound + returning), 16.0
        )

    def test_assign_round_trip_writes_both_route_legs(self):
        outbound = [{"steer": "A", "seconds": 2.0}]
        returning = [{"steer": "D", "seconds": 3.0}]
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "config.json"
            config.write_text('{"routes": {}}', encoding="utf-8")

            assign_round_trip(config, "to_fob", outbound, returning, "Bakurani")

            saved = json.loads(config.read_text(encoding="utf-8"))
        self.assertEqual(saved["routes"]["Bakurani"]["to_fob"], outbound)
        self.assertEqual(saved["routes"]["Bakurani"]["to_hq"], returning)

    def test_assign_round_trip_requires_both_legs(self):
        with self.assertRaisesRegex(ValueError, "Both round-trip legs"):
            assign_round_trip("unused.json", "to_fob", [{"steer": "", "seconds": 1}], [])


if __name__ == "__main__":
    unittest.main()
