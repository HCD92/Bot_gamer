"""Timeout-guarded logistics finite state machine."""

from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import audio
from capture import _put_latest
from input_controller import InputController
from safety_supervisor import SafetyContext, SafetyDecision, SafetyResult, SafetySupervisor
from vision import VisionObservation


class State(str, Enum):
    IDLE = "STATE_IDLE"
    PAUSED = "STATE_PAUSED"
    BUY_PALLET = "STATE_BUY_PALLET"
    CHECK_MAP = "STATE_CHECK_MAP"
    DRIVE_TO_FOB = "STATE_DRIVE_TO_FOB"
    DELIVER = "STATE_DELIVER"
    RETURN_HQ = "STATE_RETURN_HQ"
    RECOVERY = "STATE_RECOVERY"


@dataclass(frozen=True)
class AgentConfig:
    timeouts: dict[str, float]
    signals: dict[str, str]
    menu_clicks: dict[str, Any]
    route_segments: dict[str, list[dict[str, Any]]]
    unload_seconds: float = 8.0
    menu_settle_seconds: float = 0.35
    map_settle_seconds: float = 0.8
    recovery_reverse_seconds: float = 1.2
    max_damage: float = 75.0
    max_deliveries: int = 0
    prompt_clear_seconds: float = 1.5
    close_menu_with_escape: bool = False
    config_path: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any], config_path: str | None = None) -> "AgentConfig":
        return cls(
            timeouts={key: float(value) for key, value in data.get("timeouts", {}).items()},
            signals=data.get("signals", {}),
            menu_clicks=data.get("menu_clicks", {}),
            route_segments=data.get("route_segments", {}),
            unload_seconds=float(data.get("unload_seconds", 8.0)),
            menu_settle_seconds=float(data.get("menu_settle_seconds", 0.35)),
            map_settle_seconds=float(data.get("map_settle_seconds", 0.8)),
            recovery_reverse_seconds=float(data.get("recovery_reverse_seconds", 1.2)),
            max_damage=float(data.get("max_damage", 75.0)),
            max_deliveries=int(data.get("max_deliveries", 0)),
            prompt_clear_seconds=float(data.get("prompt_clear_seconds", 1.5)),
            close_menu_with_escape=bool(data.get("close_menu_with_escape", False)),
            config_path=config_path,
        )


class LogisticsAgent(threading.Thread):
    def __init__(
        self,
        config: AgentConfig,
        input_controller: InputController,
        commands: queue.Queue,
        observations: queue.Queue,
        telemetry: queue.Queue,
        initial_priority: str = "Build",
    ):
        super().__init__(name="logistics-agent", daemon=True)
        self.config = config
        self.input = input_controller
        self.safety = SafetySupervisor()
        self._last_safety_result = SafetyResult(SafetyDecision.STOP, "No input request evaluated")
        self.input.set_safety_gate(self._authorize_input)
        self.commands = commands
        self.observations = observations
        self.telemetry = telemetry
        self.priority = initial_priority
        self.state = State.IDLE
        self._resume_state = State.IDLE
        self._stop_event = threading.Event()
        self._last_observation: VisionObservation | None = None
        self._state_started = time.monotonic()
        self._loop_started: float | None = None
        self._deliveries = 0
        self._delivery_target = config.max_deliveries
        self._action = "Standby"
        self._route_index = 0
        self._routes = {k: list(v) for k, v in config.route_segments.items()}
        self._recovery_attempts = 0
        self._unload_started = False
        self._prompt_seen = False
        self._prompt_gone_since: float | None = None
        self._last_speed = None
        self._zero_speed_since: float | None = None
        self._recovery_stage = 0
        self._doors_locked = False
        self.policy = None
        self.frame_source = None  # callable -> (forward_bgr, minimap_bgr) | None
        self._policy_lock = threading.Lock()
        self.policy_step_seconds = 0.2

    def set_policy(self, policy) -> None:
        """Install (or clear with None) a trained DrivingPolicy for autonomous driving."""
        with self._policy_lock:
            self.policy = policy

    def stop(self) -> None:
        self._stop_event.set()
        self.input.halt()

    def emergency_stop(self) -> None:
        self.safety.latch_emergency_stop()
        self.input.halt()
        self._set_state(State.IDLE, "Emergency stop")

    def _authorize_input(self, action: str, keys: tuple[str, ...]) -> SafetyResult:
        observation = self._last_observation
        route_name = "to_fob" if self.state == State.DRIVE_TO_FOB else "to_hq"
        route_available = (
            self.policy is not None or bool(self._routes.get(route_name))
            if self.state in (State.DRIVE_TO_FOB, State.RETURN_HQ)
            else True
        )
        timeout = self._timeout()
        context = SafetyContext(
            fsm_state=self.state.value,
            capture_timestamp=observation.timestamp if observation else None,
            state_elapsed_seconds=time.monotonic() - self._state_started,
            state_timeout_seconds=timeout,
            route_available=route_available,
            vehicle_speed=(
                float(observation.values["speed"])
                if observation and observation.values.get("speed") is not None
                else None
            ),
            vehicle_damage=(
                float(observation.values["vehicle_damage"])
                if observation and observation.values.get("vehicle_damage") is not None
                else None
            ),
            max_damage=self.config.max_damage,
        )
        self._last_safety_result = self.safety.evaluate(context, action, keys)
        return self._last_safety_result

    def _set_state(self, state: State, action: str | None = None) -> None:
        self.state = state
        self._state_started = time.monotonic()
        self._action = action or state.value.removeprefix("STATE_").replace("_", " ").title()
        self._publish()

    def _publish(self) -> None:
        now = time.monotonic()
        _put_latest(
            self.telemetry,
            {
                "state": self.state.value,
                "active_key": self.input.active_key,
                "action": self._action,
                "deliveries": self._deliveries,
                "delivery_target": self._delivery_target,
                "loop_seconds": round(now - self._loop_started, 1) if self._loop_started else 0.0,
                "priority": self.priority,
                "safety_decision": self._last_safety_result.decision.value,
                "safety_reason": self._last_safety_result.reason,
            },
        )

    def _signal(self, logical_name: str) -> str:
        return self.config.signals.get(logical_name, logical_name)

    def _detected(self, logical_name: str) -> bool:
        return bool(self._last_observation and self._last_observation.detected(self._signal(logical_name)))

    def _value(self, name: str) -> int | float | None:
        if self._last_observation is None:
            return None
        return self._last_observation.values.get(name)

    def _timeout(self) -> float:
        return float(self.config.timeouts.get(self.state.value, 20.0))

    def _expired(self) -> bool:
        return time.monotonic() - self._state_started > self._timeout()

    def _drain_inputs(self) -> None:
        while True:
            try:
                self._last_observation = self.observations.get_nowait()
            except queue.Empty:
                break
        while True:
            try:
                command = self.commands.get_nowait()
            except queue.Empty:
                break
            kind = command.get("kind") if isinstance(command, dict) else command
            if kind == "priority":
                self.priority = str(command.get("value", self.priority))
            elif kind == "start" and self.state == State.IDLE:
                self._deliveries = 0
                self._delivery_target = max(
                    0, int(command.get("deliveries", self.config.max_deliveries))
                )
                self.input.enable()
                self._loop_started = time.monotonic()
                self._route_index = 0
                self._doors_locked = False
                self._set_state(State.BUY_PALLET)
            elif kind == "pause" and self.state not in (State.IDLE, State.PAUSED):
                self._resume_state = self.state
                self.input.halt()
                self._set_state(State.PAUSED, "Paused")
            elif kind == "resume" and self.state == State.PAUSED:
                self.input.enable()
                self._set_state(self._resume_state)
            elif kind in ("stop", "emergency"):
                self.emergency_stop()

    def _fail_to_recovery(self, reason: str) -> None:
        self.input.release_all()
        self._recovery_stage = 0
        audio.low_double()
        self._set_state(State.RECOVERY, reason)

    def _check_vehicle(self) -> bool:
        damage = self._value("vehicle_damage")
        if damage is not None and damage >= self.config.max_damage:
            self._fail_to_recovery("Damage limit reached")
            return False
        speed = self._value("speed")
        if self.state == State.DRIVE_TO_FOB and speed is not None and speed <= 0:
            if self._zero_speed_since is None:
                self._zero_speed_since = time.monotonic()
            elif time.monotonic() - self._zero_speed_since > 8.0:
                self._fail_to_recovery("No movement detected")
                return False
        else:
            self._zero_speed_since = None
        return True

    def _buy_pallet(self) -> None:
        if not self._doors_locked:
            self._action = "Locking doors"
            if not self.input.press("L", 0.08, 0.3):
                return
            self._doors_locked = True
        self._action = "Opening pallet menu"
        if not self.input.press("B", 0.12):
            return
        deadline = time.monotonic() + self.config.menu_settle_seconds
        while time.monotonic() < deadline and not self._stop_event.is_set() and not self.input.halted:
            self._drain_observations_only()
            if self._detected("pallet_menu_open"):
                break
            time.sleep(0.05)
        if not self._detected("pallet_menu_open"):
            self._fail_to_recovery("Pallet menu not verified")
            return
        if not self._click("pallets_tab"):
            return
        category = {"mechanical": "repair"}.get(self.priority.lower(), self.priority.lower())
        card = self.config.menu_clicks.get("cards", {}).get(category)
        if not card:
            self._fail_to_recovery(f"No card click configured for {self.priority}")
            return
        if not self.input.click_at(int(card["x"]), int(card["y"])):
            return
        time.sleep(0.4)
        audio.high_single()
        if self.config.close_menu_with_escape:
            # Note: Esc is also the global emergency hotkey, so this is off by default.
            self.input.press("ESC", 0.1)
        elif not self._click("confirm_exit"):
            return
        self._route_index = 0
        self._set_state(State.CHECK_MAP)

    def _click(self, name: str) -> bool:
        point = self.config.menu_clicks.get(name)
        if not point:
            self._fail_to_recovery(f"No click target configured for {name}")
            return False
        if not self.input.click_at(int(point["x"]), int(point["y"])):
            return False
        time.sleep(0.4)
        return True

    def _drain_observations_only(self) -> None:
        while True:
            try:
                self._last_observation = self.observations.get_nowait()
            except queue.Empty:
                break

    def _check_map(self) -> None:
        self._action = "Checking friendly FOB status"
        if not self.input.press("M", 0.12):
            return
        deadline = time.monotonic() + self.config.map_settle_seconds
        while time.monotonic() < deadline and not self._stop_event.is_set() and not self.input.halted:
            self._drain_observations_only()
            time.sleep(0.05)
        self.input.press("M", 0.12)
        if not self._detected("friendly_fob"):
            self._fail_to_recovery("Friendly FOB not detected on map")
            return
        self._begin_route(State.DRIVE_TO_FOB)

    def _reload_routes(self) -> None:
        """Refresh route segments from config.json so recorded routes apply immediately."""
        path = self.config.config_path
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError):
            return
        legacy = data.get("agent", {}).get("route_segments", {})
        active_map = data.get("agent", {}).get("active_map", "")
        scoped = data.get("routes", {}).get(active_map, {})
        self._routes = {
            name: scoped.get(name) or legacy.get(name) or [] for name in ("to_fob", "to_hq")
        }

    def _routes_have(self, route_name: str) -> bool:
        self._reload_routes()
        return self.policy is not None or bool(self._routes.get(route_name))

    def _begin_route(self, state: State) -> None:
        route_name = "to_fob" if state == State.DRIVE_TO_FOB else "to_hq"
        self._reload_routes()
        if self.policy is None and not self._routes.get(route_name):
            self._fail_to_recovery(f"Configure route_segments.{route_name}")
            return
        self._route_index = 0
        self._set_state(state)

    def _drive_with_policy(self, policy, route_name: str, next_state: State) -> None:
        destination_signal = "delivery_zone" if route_name == "to_fob" else "hq_zone"
        if self._detected(destination_signal):
            self.input.release_all()
            self._set_state(next_state)
            return
        sample = self.frame_source() if self.frame_source else None
        if sample is None:
            self._action = "Waiting for frames"
            time.sleep(0.05)
            return
        label, _ = policy.predict(*sample)
        self._action = f"Model drive: {('STRAIGHT', 'LEFT', 'RIGHT', 'BRAKE')[label]}"
        step = self.policy_step_seconds
        if label == 3:
            self.input.hold_keys(("S",), step)
        elif label in (1, 2):
            self.input.hold_keys(("W", "A" if label == 1 else "D"), step)
        else:
            self.input.hold_keys(("W",), step)

    def _drive(self, route_name: str, next_state: State) -> None:
        with self._policy_lock:
            policy = self.policy
        if policy is not None:
            self._drive_with_policy(policy, route_name, next_state)
            return
        segments = self._routes.get(route_name, [])
        if not segments:
            self._fail_to_recovery(f"Configure route_segments.{route_name}")
            return
        if self._route_index >= len(segments):
            self._set_state(next_state)
            return
        destination_signal = "delivery_zone" if route_name == "to_fob" else "hq_zone"
        if self._detected(destination_signal):
            self._set_state(next_state)
            return
        segment = segments[self._route_index]
        steer = str(segment.get("steer", "")).upper()
        keys = ["W"]
        if steer in ("A", "D"):
            keys.append(steer)
        duration = min(8.0, max(0.15, float(segment.get("seconds", 1.0))))
        self._action = f"Driving {route_name} segment {self._route_index + 1}/{len(segments)}"
        if self.input.hold_keys(keys, duration):
            self._route_index += 1
            if self._route_index >= len(segments) and not self._detected(destination_signal):
                self._fail_to_recovery(f"{destination_signal} not verified at end of route")
                return
        if self._expired():
            self._fail_to_recovery(f"{route_name} route timed out")

    def _prompt_visible(self) -> bool:
        return self._detected("deploy_prompt") or self._detected("delivery_zone")

    def _deliver(self) -> None:
        if not self._unload_started:
            if not self._prompt_visible():
                self._fail_to_recovery("Delivery zone signal lost")
                return
            self._prompt_seen = True
            self._action = "Dropping pallet"
            if not self.input.press("G", 0.10):
                return
            time.sleep(0.5)
            self._action = "Unpacking pallet"
            if not self.input.hold("F", self.config.unload_seconds):
                return
            self._unload_started = True
            self._prompt_gone_since = None
            return
        # Verified by the F-hold finishing plus delivery_complete, or the
        # interaction prompt above the truck bed clearing. Never by scoreboard OCR.
        self._drain_observations_only()
        complete = (
            self._detected("delivery_complete")
            or self._detected("supplies_delivered")
            or self._detected("delivery_payout")
        )
        if self._prompt_visible():
            self._prompt_gone_since = None
        elif self._prompt_gone_since is None:
            self._prompt_gone_since = time.monotonic()
        prompt_cleared = (
            self._prompt_seen
            and self._prompt_gone_since is not None
            and time.monotonic() - self._prompt_gone_since >= self.config.prompt_clear_seconds
        )
        if complete or prompt_cleared:
            self._deliveries += 1
            audio.high_double()
            self._recovery_attempts = 0
            self._unload_started = False
            self._prompt_seen = False
            self._prompt_gone_since = None
            self._route_index = 0
            self._loop_started = time.monotonic()
            if self._delivery_target > 0 and self._deliveries >= self._delivery_target:
                self.input.halt()
                self._set_state(State.IDLE, "Delivery target reached")
            elif self._routes_have("to_hq"):
                self._begin_route(State.RETURN_HQ)
            else:
                self._set_state(State.BUY_PALLET)
        elif self._expired():
            self._unload_started = False
            self._fail_to_recovery("Delivery completion not verified")

    def _recover(self) -> None:
        if self._recovery_stage == 0:
            if self._recovery_attempts >= 3:
                self.input.halt()
                self._set_state(State.IDLE, "Recovery limit reached; stopped")
                return
            self._recovery_attempts += 1
            self._action = f"Recovery reverse ({self._recovery_attempts}/3)"
            if self.input.hold_keys(("S", "A" if self._recovery_attempts % 2 else "D"), self.config.recovery_reverse_seconds):
                self._recovery_stage = 1
        elif self._recovery_stage == 1:
            self._action = "Recovery pause"
            self.input.press("SPACE", 0.25)
            self._recovery_stage = 2
        else:
            self._route_index = 0
            if self._last_observation and self._detected("delivery_zone"):
                self._set_state(State.DELIVER)
            elif self._routes_have("to_fob"):
                self._begin_route(State.DRIVE_TO_FOB)
            else:
                self.input.halt()
                self._set_state(State.IDLE, "No calibrated route; stopped")

    def run(self) -> None:
        self._publish()
        while not self._stop_event.is_set():
            self._drain_inputs()
            if self._last_observation and not self._check_vehicle():
                self._publish()
                continue
            if self.state == State.BUY_PALLET:
                self._buy_pallet()
            elif self.state == State.CHECK_MAP:
                self._check_map()
            elif self.state == State.DRIVE_TO_FOB:
                self._drive("to_fob", State.DELIVER)
            elif self.state == State.DELIVER:
                self._deliver()
            elif self.state == State.RETURN_HQ:
                self._drive("to_hq", State.BUY_PALLET)
            elif self.state == State.RECOVERY:
                self._recover()
            if self.state not in (State.IDLE, State.PAUSED) and self._expired():
                self._fail_to_recovery(f"{self.state.value.removeprefix('STATE_')} timed out")
            self._publish()
            time.sleep(0.05)
        self.input.halt()
