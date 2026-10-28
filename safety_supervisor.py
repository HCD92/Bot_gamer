"""Fail-closed approval for requested automation input."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum


class SafetyDecision(str, Enum):
    ALLOW = "ALLOW"
    CLAMP = "CLAMP"
    BRAKE = "BRAKE"
    STOP = "STOP"
    EMERGENCY_STOP = "EMERGENCY_STOP"


@dataclass(frozen=True)
class SafetyContext:
    fsm_state: str
    capture_timestamp: float | None
    state_elapsed_seconds: float
    state_timeout_seconds: float
    route_available: bool = True
    vehicle_speed: float | None = None
    vehicle_damage: float | None = None
    max_damage: float = 75.0
    model_confidence: float | None = None
    minimum_model_confidence: float = 0.0
    emergency_stop: bool = False


@dataclass(frozen=True)
class SafetyResult:
    decision: SafetyDecision
    reason: str


class SafetySupervisor:
    """Authorize each input request using the latest known automation evidence."""

    ACTIVE_STATES = frozenset({
        "STATE_BUY_PALLET",
        "STATE_CHECK_MAP",
        "STATE_DRIVE_TO_FOB",
        "STATE_DELIVER",
        "STATE_RETURN_HQ",
        "STATE_RECOVERY",
    })
    DRIVE_STATES = frozenset({"STATE_DRIVE_TO_FOB", "STATE_RETURN_HQ"})
    RECOVERY_KEYS = frozenset({"S", "A", "D", "SPACE"})

    def __init__(
        self,
        max_capture_age_seconds: float = 1.5,
        clock=time.time,
    ):
        self.max_capture_age_seconds = max(0.1, float(max_capture_age_seconds))
        self._clock = clock
        self._emergency_latched = False

    @property
    def emergency_latched(self) -> bool:
        return self._emergency_latched

    def latch_emergency_stop(self) -> None:
        self._emergency_latched = True

    def evaluate(
        self,
        context: SafetyContext,
        action: str,
        keys: tuple[str, ...] = (),
    ) -> SafetyResult:
        if self._emergency_latched or context.emergency_stop:
            return SafetyResult(SafetyDecision.EMERGENCY_STOP, "Emergency stop is latched")

        if context.fsm_state not in self.ACTIVE_STATES:
            return SafetyResult(
                SafetyDecision.STOP,
                f"Input is not permitted in FSM state {context.fsm_state!r}",
            )

        if context.capture_timestamp is None:
            return SafetyResult(SafetyDecision.STOP, "No successful capture is available")
        capture_age = self._clock() - context.capture_timestamp
        if capture_age < 0 or capture_age > self.max_capture_age_seconds:
            return SafetyResult(
                SafetyDecision.STOP,
                f"Capture is stale ({capture_age:.2f}s old)",
            )

        if context.state_timeout_seconds <= 0 or (
            context.state_elapsed_seconds >= context.state_timeout_seconds
        ):
            return SafetyResult(SafetyDecision.STOP, "FSM state timeout expired")

        if context.vehicle_damage is None:
            return SafetyResult(SafetyDecision.STOP, "Vehicle damage telemetry is unavailable")
        if context.vehicle_damage >= context.max_damage:
            return SafetyResult(SafetyDecision.STOP, "Vehicle damage limit reached")

        if context.fsm_state in self.DRIVE_STATES:
            if not context.route_available:
                return SafetyResult(SafetyDecision.STOP, "No calibrated route or active model")
            if context.vehicle_speed is None:
                return SafetyResult(SafetyDecision.STOP, "Vehicle speed telemetry is unavailable")

        if context.fsm_state == "STATE_RECOVERY":
            requested = {key.upper() for key in keys}
            if (
                action not in {"key", "hold_keys"}
                or not requested
                or not requested.issubset(self.RECOVERY_KEYS)
            ):
                return SafetyResult(
                    SafetyDecision.STOP,
                    f"Action {action!r} is not permitted during recovery",
                )

        if context.model_confidence is not None and (
            context.model_confidence < context.minimum_model_confidence
        ):
            return SafetyResult(
                SafetyDecision.BRAKE,
                "Neural-model confidence is below the configured minimum",
            )

        return SafetyResult(SafetyDecision.ALLOW, "Safety conditions satisfied")
