"""Circuit breaker: çok fazla hata → OPEN (bekle), sonra degrade. Wall-clock limiti oturumu sonlandırabilir."""

import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque, Optional

from config.settings import Settings


class ResilienceState(str, Enum):
    NORMAL = "normal"
    DEGRADED = "degraded"
    OPEN = "open"


@dataclass
class ResilienceStats:
    breaker_open_count: int = 0
    degrade_entries: int = 0
    degrade_frames: int = 0
    recovered_count: int = 0
    transient_wall_time_sec: float = 0.0
    telemetry_alarm_count: int = 0
    duplicate_streak_alarm_count: int = 0
    permanent_reject_alarm_count: int = 0
    fetch_attempt_count: int = 0
    success_cycle_count: int = 0
    ack_failure_count: int = 0
    throughput_fps: float = 0.0
    success_rate: float = 0.0


class SessionResilienceController:
    """Circuit breaker ve degrade kontrolü (wall-clock tabanlı)."""

    def __init__(self, log) -> None:
        self.log = log
        self.state: ResilienceState = ResilienceState.NORMAL
        self.stats = ResilienceStats()

        self._fetch_events: Deque[float] = deque()
        self._ack_events: Deque[float] = deque()
        self._started_at_monotonic: float = time.monotonic()

        self._non_normal_since: Optional[float] = None
        self._open_until_monotonic: float = 0.0
        self._degrade_frame_counter: int = 0

    def _now(self) -> float:
        return time.monotonic()

    def _prune_window(self, q: Deque[float], now: float) -> None:
        cutoff = now - float(Settings.CB_TRANSIENT_WINDOW_SEC)
        while q and q[0] < cutoff:
            q.popleft()

    def _decay_window(self, q: Deque[float]) -> None:
        q.clear()

    def _current_transient_wall_time(self, now: Optional[float] = None) -> float:
        ts = self.stats.transient_wall_time_sec
        if self._non_normal_since is not None:
            curr = now if now is not None else self._now()
            ts += max(0.0, curr - self._non_normal_since)
        return ts

    def _transition(self, new_state: ResilienceState, reason: str, now: Optional[float] = None) -> None:
        curr = now if now is not None else self._now()
        old_state = self.state
        if old_state == new_state:
            return

        if old_state == ResilienceState.NORMAL and new_state != ResilienceState.NORMAL:
            self._non_normal_since = curr
        elif old_state != ResilienceState.NORMAL and new_state == ResilienceState.NORMAL:
            if self._non_normal_since is not None:
                self.stats.transient_wall_time_sec += max(0.0, curr - self._non_normal_since)
            self._non_normal_since = None

        if old_state == ResilienceState.NORMAL and new_state == ResilienceState.DEGRADED:
            self.stats.degrade_entries += 1
        elif old_state == ResilienceState.OPEN and new_state == ResilienceState.DEGRADED:
            self.stats.degrade_entries += 1
        elif old_state != ResilienceState.NORMAL and new_state == ResilienceState.NORMAL:
            self.stats.recovered_count += 1

        self.state = new_state
        self.log.info(
            "Resilience transition | "
            f"breaker_state={self.state.value} "
            f"reason={reason} "
            f"window_events_fetch={len(self._fetch_events)} "
            f"window_events_ack={len(self._ack_events)} "
            f"elapsed_transient_sec={self._current_transient_wall_time(curr):.1f}"
        )

    def _enter_open(self, reason: str, now: Optional[float] = None) -> None:
        curr = now if now is not None else self._now()
        self.stats.breaker_open_count += 1
        self._open_until_monotonic = curr + float(Settings.CB_OPEN_COOLDOWN_SEC)
        self._transition(ResilienceState.OPEN, reason=reason, now=curr)

    def on_fetch_transient(self) -> None:
        now = self._now()
        self._fetch_events.append(now)
        self._prune_window(self._fetch_events, now)
        if len(self._fetch_events) >= int(Settings.CB_TRANSIENT_MAX_EVENTS):
            self._enter_open(reason="fetch_transient_storm", now=now)
        elif self.state == ResilienceState.NORMAL:
            self._transition(ResilienceState.DEGRADED, reason="fetch_transient_detected", now=now)

    def on_fetch_attempt(self) -> None:
        self.stats.fetch_attempt_count += 1

    def on_ack_failure(self) -> None:
        self.stats.ack_failure_count += 1
        now = self._now()
        self._ack_events.append(now)
        self._prune_window(self._ack_events, now)
        if len(self._ack_events) >= int(Settings.CB_TRANSIENT_MAX_EVENTS):
            self._enter_open(reason="ack_transient_storm", now=now)
        elif self.state == ResilienceState.NORMAL:
            self._transition(ResilienceState.DEGRADED, reason="ack_transient_detected", now=now)

    def on_success_cycle(self) -> None:
        self.stats.success_cycle_count += 1
        now = self._now()
        self._prune_window(self._fetch_events, now)
        self._prune_window(self._ack_events, now)
        if self.state != ResilienceState.NORMAL:
            self._decay_window(self._fetch_events)
            self._decay_window(self._ack_events)
            self._transition(ResilienceState.NORMAL, reason="fetch_and_ack_success", now=now)
        self._degrade_frame_counter = 0

    def before_fetch(self) -> bool:
        now = self._now()
        if self.state != ResilienceState.OPEN:
            return True
        if now < self._open_until_monotonic:
            return False
        self._transition(ResilienceState.DEGRADED, reason="open_cooldown_elapsed_half_open", now=now)
        return True

    def open_cooldown_remaining(self) -> float:
        return max(0.0, self._open_until_monotonic - self._now())

    def record_degraded_frame(self) -> int:
        self._degrade_frame_counter += 1
        self.stats.degrade_frames += 1
        return self._degrade_frame_counter

    def on_duplicate_streak_alarm(self, streak: int, threshold: int) -> None:
        self.stats.telemetry_alarm_count += 1
        self.stats.duplicate_streak_alarm_count += 1
        now = self._now()
        if self.state == ResilienceState.NORMAL:
            self._transition(
                ResilienceState.DEGRADED,
                reason=f"duplicate_streak_alarm_{streak}_over_{threshold}",
                now=now,
            )

    def on_permanent_reject_alarm(self, streak: int, threshold: int) -> None:
        self.stats.telemetry_alarm_count += 1
        self.stats.permanent_reject_alarm_count += 1
        now = self._now()
        if self.state == ResilienceState.NORMAL:
            self._transition(
                ResilienceState.DEGRADED,
                reason=f"permanent_reject_streak_alarm_{streak}_over_{threshold}",
                now=now,
            )

    def should_abort(self) -> Optional[str]:
        if self.state == ResilienceState.NORMAL:
            return None
        wall_time = self._current_transient_wall_time()
        limit = float(Settings.CB_SESSION_MAX_TRANSIENT_SEC)
        if wall_time >= limit:
            self.stats.telemetry_alarm_count += 1
            return (
                f"Transient wall time {wall_time:.0f}s >= limit {limit:.0f}s; "
                "telemetry alarm only, continuing degraded session"
            )
        return None

    def is_degraded(self) -> bool:
        return self.state in {ResilienceState.DEGRADED, ResilienceState.OPEN}

    def finalize(self) -> ResilienceStats:
        if self._non_normal_since is not None:
            now = self._now()
            self.stats.transient_wall_time_sec += max(0.0, now - self._non_normal_since)
            self._non_normal_since = None
        elapsed = max(1e-6, self._now() - self._started_at_monotonic)
        self.stats.throughput_fps = float(self.stats.fetch_attempt_count) / elapsed
        attempts = max(1, int(self.stats.success_cycle_count + self.stats.ack_failure_count))
        self.stats.success_rate = float(self.stats.success_cycle_count) / float(attempts)
        return self.stats
