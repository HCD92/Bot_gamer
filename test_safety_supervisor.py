import unittest

from safety_supervisor import (
    SafetyContext,
    SafetyDecision,
    SafetySupervisor,
)


class SafetySupervisorTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.0
        self.supervisor = SafetySupervisor(clock=lambda: self.now)

    def context(self, **overrides):
        values = {
            "fsm_state": "STATE_DRIVE_TO_FOB",
            "capture_timestamp": self.now,
            "state_elapsed_seconds": 1.0,
            "state_timeout_seconds": 15.0,
            "route_available": True,
            "vehicle_speed": 20.0,
            "vehicle_damage": 5.0,
        }
        values.update(overrides)
        return SafetyContext(**values)

    def test_allows_fresh_valid_drive_request(self):
        result = self.supervisor.evaluate(self.context(), "hold_keys", ("W",))

        self.assertEqual(result.decision, SafetyDecision.ALLOW)

    def test_missing_or_stale_capture_stops(self):
        for timestamp in (None, self.now - 2.0):
            with self.subTest(timestamp=timestamp):
                result = self.supervisor.evaluate(
                    self.context(capture_timestamp=timestamp), "key", ("W",)
                )
                self.assertEqual(result.decision, SafetyDecision.STOP)

    def test_unknown_fsm_state_stops(self):
        result = self.supervisor.evaluate(
            self.context(fsm_state="STATE_UNKNOWN"), "key", ("W",)
        )

        self.assertEqual(result.decision, SafetyDecision.STOP)

    def test_state_timeout_stops(self):
        result = self.supervisor.evaluate(
            self.context(state_elapsed_seconds=15.0), "key", ("W",)
        )

        self.assertEqual(result.decision, SafetyDecision.STOP)

    def test_drive_requires_route_and_speed_telemetry(self):
        for overrides in (
            {"route_available": False},
            {"vehicle_speed": None},
        ):
            with self.subTest(overrides=overrides):
                result = self.supervisor.evaluate(
                    self.context(**overrides), "key", ("W",)
                )
                self.assertEqual(result.decision, SafetyDecision.STOP)

    def test_unknown_or_excessive_damage_stops(self):
        for damage in (None, 75.0):
            with self.subTest(damage=damage):
                result = self.supervisor.evaluate(
                    self.context(vehicle_damage=damage), "key", ("W",)
                )
                self.assertEqual(result.decision, SafetyDecision.STOP)

    def test_low_model_confidence_requests_brake(self):
        result = self.supervisor.evaluate(
            self.context(model_confidence=0.4, minimum_model_confidence=0.6),
            "key",
            ("W",),
        )

        self.assertEqual(result.decision, SafetyDecision.BRAKE)

    def test_recovery_allows_only_controlled_recovery_keys(self):
        context = self.context(fsm_state="STATE_RECOVERY")

        allowed = self.supervisor.evaluate(context, "hold_keys", ("S", "A"))
        denied = self.supervisor.evaluate(context, "hold_keys", ("W",))
        denied_click = self.supervisor.evaluate(context, "click", ())

        self.assertEqual(allowed.decision, SafetyDecision.ALLOW)
        self.assertEqual(denied.decision, SafetyDecision.STOP)
        self.assertEqual(denied_click.decision, SafetyDecision.STOP)

    def test_emergency_stop_remains_latched(self):
        self.supervisor.latch_emergency_stop()

        result = self.supervisor.evaluate(self.context(), "key", ("W",))

        self.assertEqual(result.decision, SafetyDecision.EMERGENCY_STOP)
        self.assertEqual(
            self.supervisor.evaluate(self.context(), "key", ("W",)).decision,
            SafetyDecision.EMERGENCY_STOP,
        )


if __name__ == "__main__":
    unittest.main()
