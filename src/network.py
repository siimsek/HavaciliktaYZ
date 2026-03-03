"""Sunucu HTTP iletişimi: frame al, sonuç gönder. Retry, circuit breaker, idempotency destekli."""

import time
import random
from collections import OrderedDict, deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple

import math
import cv2
import numpy as np
import requests

from config.settings import Settings
from src.payload_schema import CompetitionPayloadSchema
from src.utils import Logger, log_json_to_disk


class FrameFetchStatus(str, Enum):
    OK = "ok"
    END_OF_STREAM = "end_of_stream"
    TRANSIENT_ERROR = "transient_error"
    FATAL_ERROR = "fatal_error"


@dataclass
class FrameFetchResult:
    status: FrameFetchStatus
    frame_data: Optional[Dict[str, Any]] = None
    error_type: str = ""
    http_status: Optional[int] = None
    is_duplicate: bool = False


class SendResultStatus(str, Enum):
    ACKED = "acked"
    FALLBACK_ACKED = "fallback_acked"
    RETRYABLE_FAILURE = "retryable_failure"
    PERMANENT_REJECTED = "permanent_rejected"


class NetworkManager:
    """Sunucu ile iletişimi yöneten ana sınıf."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        simulation_mode: Optional[bool] = None,
    ) -> None:
        self.base_url = base_url or Settings.BASE_URL
        self.simulation_mode = (
            Settings.SIMULATION_MODE if simulation_mode is None else simulation_mode
        )
        self.log = Logger("Network")
        self.session = requests.Session()
        self.session.verify = True
        self._frame_counter: int = 0
        self._result_counter: int = 0
        self._sim_image_cache: Optional[np.ndarray] = None
        self._seen_frames_lru: "OrderedDict[str, None]" = OrderedDict()
        self._submitted_frames_lru: "OrderedDict[str, None]" = OrderedDict()
        self._force_fallback_frames_lru: "OrderedDict[str, None]" = OrderedDict()
        self._timeout_counters: Dict[str, int] = {
            "fetch": 0,
            "image": 0,
            "submit": 0,
        }
        self._payload_guard_counters: Dict[str, int] = {
            "preflight_reject": 0,
            "payload_clipped": 0,
        }
        self._clip_ratio_window: Deque[int] = deque(maxlen=100)
        self._clip_frame_events: List[Dict[str, Any]] = []
        self._clip_aggregate_stats: Dict[str, Any] = {
            "frames_with_clip": 0,
            "total_dropped": 0,
            "total_raw_objects": 0,
            "total_post_cap_objects": 0,
            "dropped_by_class": {"0": 0, "1": 0, "2": 0, "3": 0},
        }
        self._session_id: str = str(int(time.time()))
        self._task3_references: list = []

    def get_task3_references(self) -> list:
        return list(self._task3_references)

    def assert_contract_ready(self) -> None:
        CompetitionPayloadSchema.self_check()

    def start_session(self) -> bool:
        if self.simulation_mode:
            self.log.success(f"[SIMULATION] Session started -> {self.base_url}")
            return True

        for attempt in range(1, Settings.MAX_RETRIES + 1):
            try:
                self.log.info(
                    f"Connecting server... (Attempt {attempt}/{Settings.MAX_RETRIES})"
                )
                response = self.session.get(
                    self.base_url,
                    timeout=self._timeout_tuple(self._read_timeout_frame_meta()),
                )
                if response.status_code == 200:
                    self.log.success(f"Server connection successful -> {self.base_url}")

                    try:
                        data = response.json()
                        if isinstance(data, dict):
                            max_f = data.get("max_frames")
                            if max_f is not None:
                                Settings.MAX_FRAMES = int(max_f)
                                self.log.info(f"Oturum: MAX_FRAMES güncellendi -> {Settings.MAX_FRAMES}")
                            refs = data.get("task3_references")
                            if refs and isinstance(refs, list):
                                self._task3_references = refs
                                self.log.info(f"Oturum: Sunucudan {len(refs)} Görev 3 referansı alındı.")
                            else:
                                self._task3_references = []
                    except ValueError:
                        pass

                    return True
                self.log.warn(f"Unexpected server response: {response.status_code}")
            except requests.ConnectionError:
                self.log.error(
                    f"Connection error! Retrying in {Settings.RETRY_DELAY}s..."
                )
            except requests.Timeout:
                self._increment_timeout_counter("fetch")
                self.log.error(
                    f"Connection timeout! Retrying in {Settings.RETRY_DELAY}s..."
                )
            except Exception as exc:
                self.log.error(f"Unexpected startup error: {exc}")
            self._sleep_with_backoff(attempt)

        self.log.error("Server unavailable after all retries.")
        return False

    def get_frame(self) -> FrameFetchResult:
        if self.simulation_mode:
            return FrameFetchResult(
                status=FrameFetchStatus.OK,
                frame_data=self._get_simulation_frame(),
            )

        url = f"{self.base_url}{Settings.ENDPOINT_NEXT_FRAME}"

        for attempt in range(1, Settings.MAX_RETRIES + 1):
            try:
                response = self.session.get(
                    url,
                    timeout=self._timeout_tuple(self._read_timeout_frame_meta()),
                )

                if response.status_code == 200:
                    data = response.json()
                    if not self._validate_frame_data(data):
                        self.log.error("Invalid frame schema from server.")
                        return FrameFetchResult(
                            status=FrameFetchStatus.FATAL_ERROR,
                            error_type="invalid_frame_schema",
                            http_status=200,
                        )

                    if self._should_log_json(self._frame_counter):
                        log_json_to_disk(
                            data,
                            direction="incoming",
                            tag=f"frame_{self._frame_counter}",
                        )
                    self._frame_counter += 1
                    is_duplicate = self._mark_seen_frame(data.get("frame_id"))
                    if is_duplicate:
                        self.log.warn(
                            f"Duplicate frame_id dropped by client dedup: {data.get('frame_id')}"
                        )
                    return FrameFetchResult(
                        status=FrameFetchStatus.OK,
                        frame_data=data,
                        http_status=200,
                        is_duplicate=is_duplicate,
                    )

                if response.status_code == 204:
                    # Sunucu tüm kareleri bitirdi, oturum sonu
                    self.log.info("Video finished (204 No Content)")
                    return FrameFetchResult(
                        status=FrameFetchStatus.END_OF_STREAM,
                        http_status=204,
                    )

                if 500 <= response.status_code < 600:
                    self.log.warn(
                        f"Server temporary error: HTTP {response.status_code}"
                    )
                    self._sleep_with_backoff(attempt)
                    continue

                self.log.error(f"Unexpected frame response: HTTP {response.status_code}")
                return FrameFetchResult(
                    status=FrameFetchStatus.FATAL_ERROR,
                    error_type="unexpected_http",
                    http_status=response.status_code,
                )

            except (requests.ConnectionError, requests.Timeout) as exc:
                if isinstance(exc, requests.Timeout):
                    self._increment_timeout_counter("fetch")
                self.log.warn(
                    f"Frame fetch transient error ({type(exc).__name__}) "
                    f"Attempt {attempt}/{Settings.MAX_RETRIES}"
                )
            except (ValueError, requests.exceptions.JSONDecodeError) as exc:
                self.log.warn(
                    f"JSON parse transient error ({exc}), "
                    f"attempt {attempt}/{Settings.MAX_RETRIES}"
                )
            except Exception as exc:
                self.log.warn(f"Frame fetch transient exception: {exc}")

            self._sleep_with_backoff(attempt)

        return FrameFetchResult(
            status=FrameFetchStatus.TRANSIENT_ERROR,
            error_type="retries_exhausted",
        )

    def download_image(self, frame_data: Dict[str, Any]) -> Optional[np.ndarray]:
        if self.simulation_mode:
            return self._load_simulation_image()

        frame_url = frame_data.get("frame_url", "") or frame_data.get("image_url", "")
        if not frame_url:
            self.log.error("Frame URL is missing in frame metadata")
            return None

        full_url = frame_url if str(frame_url).startswith("http") else f"{self.base_url}{frame_url}"

        for attempt in range(1, Settings.MAX_RETRIES + 1):
            try:
                response = self.session.get(
                    full_url,
                    timeout=self._timeout_tuple(self._read_timeout_image()),
                )
                if response.status_code == 200:
                    img_array = np.frombuffer(response.content, dtype=np.uint8)
                    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    if frame is None:
                        self.log.error("Image decode failed")
                        return None
                    self.log.debug(
                        f"Image downloaded: {frame.shape[1]}x{frame.shape[0]}"
                    )
                    return frame

                self.log.warn(f"Image download HTTP {response.status_code}")
            except requests.Timeout:
                self._increment_timeout_counter("image")
                self.log.warn(
                    f"Image download timeout attempt {attempt}/{Settings.MAX_RETRIES}"
                )
            except Exception as exc:
                self.log.warn(f"Image download transient error: {exc}")

            self._sleep_with_backoff(attempt)

        self.log.error("Image download failed after all retries")
        return None

    def send_result(
        self,
        frame_id: Any,
        detected_objects: List[Dict],
        detected_translation: Dict[str, float],
        frame_data: Optional[Dict[str, Any]] = None,
        frame_shape: Optional[tuple] = None,
        degrade: bool = False,
        detected_undefined_objects: Optional[List[Dict]] = None,
    ) -> SendResultStatus:
        frame_key = self._normalize_frame_key(frame_id)
        if self._was_already_submitted(frame_key):
            self.log.warn(
                f"Frame {frame_key}: duplicate submit prevented by idempotent client guard, sending gracefully."
            )
        raw_payload = self.build_competition_payload(
            frame_id=frame_id,
            detected_objects=detected_objects,
            detected_translation=detected_translation,
            frame_data=frame_data,
            frame_shape=frame_shape,
            detected_undefined_objects=detected_undefined_objects,
        )
        force_fallback = self._should_force_fallback(frame_key)
        if force_fallback:
            payload = self._build_safe_fallback_payload(raw_payload)
            preflight_rejected = True
            payload_clipped = False
        else:
            payload, preflight_rejected, payload_clipped = (
                self._preflight_validate_and_normalize_payload(
                    raw_payload,
                    frame_shape=frame_shape,
                    frame_id=frame_id,
                )
            )

        if preflight_rejected:
            self._payload_guard_counters["preflight_reject"] += 1
            self._mark_force_fallback(frame_key)
        if payload_clipped:
            self._payload_guard_counters["payload_clipped"] += 1
        self._record_clip_event(payload_clipped)

        if self._should_log_json(self._result_counter):
            log_json_to_disk(payload, direction="outgoing", tag=f"result_{frame_id}")
        self._result_counter += 1

        if self.simulation_mode:
            self.log.success(
                f"[SIMULATION] Result prepared -> Frame: {frame_id} | "
                f"Objects: {len(payload['detected_objects'])} | "
                f"Degrade: {'ON' if degrade else 'OFF'}"
            )
            return (
                SendResultStatus.FALLBACK_ACKED
                if preflight_rejected
                else SendResultStatus.ACKED
            )

        url = f"{self.base_url}{Settings.ENDPOINT_SUBMIT_RESULT}"
        idempotency_key = self._build_idempotency_key(frame_key)
        fallback_sent = preflight_rejected
        saw_4xx = False

        for attempt in range(1, Settings.MAX_RETRIES + 1):
            try:
                response = self.session.post(
                    url,
                    json=payload,
                    timeout=self._timeout_tuple(self._read_timeout_submit()),
                    headers={
                        "Content-Type": "application/json",
                        "Idempotency-Key": idempotency_key,
                    },
                )
                if response.status_code == 200:
                    self.log.debug(
                        f"Result sent successfully: Frame {frame_id} "
                        f"(degrade={'ON' if degrade else 'OFF'})"
                    )
                    self._mark_submitted(frame_key)
                    self._unmark_force_fallback(frame_key)
                    if preflight_rejected or fallback_sent:
                        return SendResultStatus.FALLBACK_ACKED
                    return SendResultStatus.ACKED

                if 400 <= response.status_code < 500:
                    saw_4xx = True
                    self.log.warn(
                        f"Submit response HTTP {response.status_code} (4xx permanent reject candidate)"
                    )
                    break

                self.log.warn(
                    f"Submit response HTTP {response.status_code} "
                    f"(attempt {attempt}/{Settings.MAX_RETRIES})"
                )
            except requests.Timeout:
                self._increment_timeout_counter("submit")
                self.log.warn(
                    f"Submit timeout attempt {attempt}/{Settings.MAX_RETRIES}"
                )
            except Exception as exc:
                self.log.warn(
                    f"Submit transient error ({type(exc).__name__}): {exc} "
                    f"(attempt {attempt}/{Settings.MAX_RETRIES})"
                )

            self._sleep_with_backoff(attempt)

        if saw_4xx:
            self._mark_force_fallback(frame_key)
            if fallback_sent:
                self.log.error(
                    f"Frame {frame_id}: fallback payload also rejected (4xx), marking permanent reject"
                )
                return SendResultStatus.PERMANENT_REJECTED

            fallback_payload = self._build_safe_fallback_payload(raw_payload)
            try:
                response = self.session.post(
                    url,
                    json=fallback_payload,
                    timeout=self._timeout_tuple(self._read_timeout_submit()),
                    headers={
                        "Content-Type": "application/json",
                        "Idempotency-Key": idempotency_key,
                    },
                )
                if response.status_code == 200:
                    self.log.warn(
                        f"Frame {frame_id}: 4xx recovered with safe fallback payload"
                    )
                    self._mark_submitted(frame_key)
                    self._unmark_force_fallback(frame_key)
                    return SendResultStatus.FALLBACK_ACKED
                if 400 <= response.status_code < 500:
                    self.log.error(
                        f"Frame {frame_id}: fallback payload rejected with HTTP {response.status_code}"
                    )
                    return SendResultStatus.PERMANENT_REJECTED

                self.log.warn(
                    f"Frame {frame_id}: fallback payload non-ACK HTTP {response.status_code}, retryable"
                )
                return SendResultStatus.RETRYABLE_FAILURE
            except requests.Timeout:
                self._increment_timeout_counter("submit")
                self.log.warn(f"Frame {frame_id}: fallback submit timeout, retryable")
                return SendResultStatus.RETRYABLE_FAILURE
            except Exception as exc:
                self.log.warn(
                    f"Frame {frame_id}: fallback submit transient error ({type(exc).__name__}): {exc}"
                )
                return SendResultStatus.RETRYABLE_FAILURE

        self.log.error(f"Result submission failed after retries for frame {frame_id}")
        return SendResultStatus.RETRYABLE_FAILURE

    @staticmethod
    def build_competition_payload(
        frame_id: Any,
        detected_objects: List[Dict],
        detected_translation: Dict[str, float],
        frame_data: Optional[Dict[str, Any]] = None,
        frame_shape: Optional[tuple] = None,
        detected_undefined_objects: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        frame_data = frame_data or {}
        frame_h = frame_w = None
        if frame_shape and len(frame_shape) >= 2:
            frame_h = int(frame_shape[0])
            frame_w = int(frame_shape[1])

        clean_objects, alias_count = CompetitionPayloadSchema.canonicalize_objects(
            detected_objects,
            frame_shape=(frame_h, frame_w) if frame_h is not None and frame_w is not None else None,
        )

        tx = NetworkManager._safe_float(detected_translation.get("translation_x", 0.0))
        ty = NetworkManager._safe_float(detected_translation.get("translation_y", 0.0))
        tz = NetworkManager._safe_float(detected_translation.get("translation_z", 0.0))

        payload_id = frame_data.get("id", frame_id)
        payload_user = frame_data.get("user", Settings.TEAM_NAME)
        payload_frame = frame_data.get("url", frame_data.get("frame", frame_id))

        payload = {
            "id": payload_id,
            "user": payload_user,
            "frame": payload_frame,
            "detected_objects": clean_objects,
            "detected_translations": [
                {
                    "translation_x": tx,
                    "translation_y": ty,
                    "translation_z": tz,
                }
            ],
            "detected_undefined_objects": detected_undefined_objects or [],
        }
        if alias_count > 0:
            Logger("Network").warn(
                f"event=payload_alias_used alias_field={CompetitionPayloadSchema.LEGACY_MOTION_FIELD} "
                f"count={alias_count}"
            )
        return payload

    def _get_simulation_frame(self) -> Dict[str, Any]:
        frame_id = self._frame_counter
        self._frame_counter += 1
        return {
            "id": frame_id,
            "url": f"/simulation/frames/{frame_id}",
            "image_url": "dummy_path",
            "session": "/simulation/session/1",
            "frame_url": "dummy_path",
            "frame_id": frame_id,
            "video_name": "simulation_video",
            "translation_x": 0.0,
            "translation_y": 0.0,
            "translation_z": 50.0,
            "gps_health_status": 1,
            "gps_health": 1,
        }

    def _load_simulation_image(self) -> Optional[np.ndarray]:
        if self._sim_image_cache is not None:
            return self._sim_image_cache.copy()

        frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
        self.log.warn("Using generated blank image for network pure-simulation mode")

        self._sim_image_cache = frame
        return frame

    def _validate_frame_data(self, data: Dict[str, Any]) -> bool:
        if not isinstance(data, dict):
            self.log.warn("Frame metadata is not dict")
            return False

        frame_id = data.get("frame_id")
        if frame_id is None:
            frame_id = data.get("id")
        if frame_id is None:
            frame_id = data.get("url")
        if frame_id is None:
            frame_id = data.get("frame")

        if frame_id is None:
            self.log.warn("Missing required frame identifier (frame_id/id/url/frame)")
            return False
        data["frame_id"] = frame_id

        if not data.get("frame_url") and data.get("image_url"):
            data["frame_url"] = data.get("image_url")
        if not data.get("image_url") and data.get("frame_url"):
            data["image_url"] = data.get("frame_url")

        health_val = data.get("gps_health")
        if health_val is None:
            health_val = data.get("gps_health_status", 0)
        try:
            if health_val is None or str(health_val).strip().lower() in {
                "unknown",
                "none",
                "null",
                "",
            }:
                data["gps_health"] = 0
            else:
                data["gps_health"] = int(float(health_val))
        except (ValueError, TypeError):
            self.log.warn(f"Corrupt gps_health value: {health_val!r}, forcing 0")
            data["gps_health"] = 0
        data["gps_health_status"] = data["gps_health"]

        for key in ["translation_x", "translation_y", "translation_z", "altitude"]:
            if key in data:
                val = data.get(key)
                try:
                    if val is None or str(val).strip().lower() in {
                        "unknown",
                        "none",
                        "null",
                        "",
                        "nan",
                    }:
                        data[key] = 0.0
                    else:
                        data[key] = float(val)
                        if math.isnan(data[key]):
                            data[key] = 0.0
                except (ValueError, TypeError):
                    self.log.warn(f"Corrupt {key}: {val!r}, forcing 0.0")
                    data[key] = 0.0

        return True

    def _preflight_validate_and_normalize_payload(
        self,
        payload: Dict[str, Any],
        frame_shape: Optional[tuple],
        frame_id: Any,
    ) -> Tuple[Dict[str, Any], bool, bool]:
        try:
            CompetitionPayloadSchema.validate_top_level_payload(payload)
        except Exception:
            self.log.error(
                f"Frame {frame_id}: preflight reject (missing top-level fields), forcing safe fallback"
            )
            return self._build_safe_fallback_payload(payload), True, False

        translations = payload.get("detected_translations")
        if not isinstance(translations, list) or len(translations) != 1:
            self.log.error(
                f"Frame {frame_id}: preflight reject (detected_translations must be list with len=1)"
            )
            return self._build_safe_fallback_payload(payload), True, False

        t0 = translations[0]
        if not isinstance(t0, dict):
            self.log.error(
                f"Frame {frame_id}: preflight reject (translation object invalid), forcing safe fallback"
            )
            return self._build_safe_fallback_payload(payload), True, False

        tx = self._safe_float(t0.get("translation_x", 0.0))
        ty = self._safe_float(t0.get("translation_y", 0.0))
        tz = self._safe_float(t0.get("translation_z", 0.0))

        objects = payload.get("detected_objects")
        if not isinstance(objects, list):
            self.log.error(
                f"Frame {frame_id}: preflight reject (detected_objects must be list), forcing safe fallback"
            )
            return self._build_safe_fallback_payload(payload), True, False

        frame_h = frame_w = None
        if frame_shape and len(frame_shape) >= 2:
            frame_h = self._safe_int(frame_shape[0])
            frame_w = self._safe_int(frame_shape[1])

        normalized_objects, alias_count = CompetitionPayloadSchema.canonicalize_objects(
            objects,
            frame_shape=(frame_h, frame_w) if frame_h is not None and frame_w is not None else None,
        )
        if alias_count > 0:
            self.log.warn(
                f"event=payload_alias_used alias_field={CompetitionPayloadSchema.LEGACY_MOTION_FIELD} "
                f"count={alias_count} frame_id={frame_id}"
            )

        capped_objects, clip_stats = self._apply_object_caps(
            normalized_objects=normalized_objects,
            frame_id=frame_id,
        )
        payload_clipped = bool(clip_stats.get("dropped_total", 0) > 0)

        clean_capped = [
            {
                "cls": obj["cls"],
                "landing_status": obj["landing_status"],
                CompetitionPayloadSchema.CANONICAL_MOTION_FIELD: obj.get(
                    CompetitionPayloadSchema.CANONICAL_MOTION_FIELD,
                    -1,
                ),
                "top_left_x": obj["top_left_x"],
                "top_left_y": obj["top_left_y"],
                "bottom_right_x": obj["bottom_right_x"],
                "bottom_right_y": obj["bottom_right_y"],
            }
            for obj in capped_objects
        ]

        return (
            {
                "id": payload.get("id"),
                "user": payload.get("user"),
                "frame": payload.get("frame"),
                "detected_objects": clean_capped,
                "detected_translations": [
                    {
                        "translation_x": tx,
                        "translation_y": ty,
                        "translation_z": tz,
                    }
                ],
                "detected_undefined_objects": payload.get("detected_undefined_objects", []),
            },
            False,
            payload_clipped,
        )

    def _apply_object_caps(
        self,
        normalized_objects: List[Dict[str, Any]],
        frame_id: Any,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        class_order = ["0", "1", "2", "3"]
        quota_raw = dict(getattr(Settings, "RESULT_CLASS_QUOTA", {"0": 40, "1": 40, "2": 10, "3": 10}))
        class_quota = {
            cls: max(0, self._safe_int(quota_raw.get(cls, 0)))
            for cls in class_order
        }
        global_cap = max(0, self._safe_int(getattr(Settings, "RESULT_MAX_OBJECTS", 100)))

        base_quota_total = sum(class_quota.values())
        if base_quota_total > 0 and base_quota_total != global_cap:
            ratio = global_cap / float(base_quota_total)
            scaled_floor = {
                cls: int(math.floor(class_quota[cls] * ratio))
                for cls in class_order
            }
            remaining = max(0, global_cap - sum(scaled_floor.values()))
            fractions = sorted(
                class_order,
                key=lambda cls: (class_quota[cls] * ratio - scaled_floor[cls], class_quota[cls]),
                reverse=True,
            )
            for cls in fractions:
                if remaining <= 0:
                    break
                scaled_floor[cls] += 1
                remaining -= 1
            class_quota = scaled_floor

        grouped: Dict[str, List[Dict[str, Any]]] = {cls: [] for cls in class_order}
        for obj in normalized_objects:
            cls = str(obj.get("cls", ""))
            if cls in grouped:
                grouped[cls].append(obj)

        class_counts = {cls: len(grouped[cls]) for cls in class_order}

        dynamic_quota = {
            cls: min(class_counts[cls], class_quota[cls])
            for cls in class_order
        }
        remaining_capacity = max(0, global_cap - sum(dynamic_quota.values()))
        overflow = {
            cls: max(0, class_counts[cls] - dynamic_quota[cls])
            for cls in class_order
        }

        while remaining_capacity > 0:
            candidate = max(class_order, key=lambda cls: (overflow[cls], class_counts[cls]))
            if overflow[candidate] <= 0:
                break
            dynamic_quota[candidate] += 1
            overflow[candidate] -= 1
            remaining_capacity -= 1

        def _rank_key(det: Dict[str, Any]) -> Tuple[float, float, int, int]:
            x1 = self._safe_int(det.get("top_left_x", 0))
            y1 = self._safe_int(det.get("top_left_y", 0))
            x2 = self._safe_int(det.get("bottom_right_x", 0))
            y2 = self._safe_int(det.get("bottom_right_y", 0))
            area = max(0, x2 - x1) * max(0, y2 - y1)
            conf = self._safe_float(det.get("_confidence", 0.0))
            return (-conf, -float(area), x1, y1)

        post_quota: List[Dict[str, Any]] = []
        dropped_by_class: Dict[str, int] = {cls: 0 for cls in class_order}
        for cls in class_order:
            ranked = sorted(grouped[cls], key=_rank_key)
            keep = ranked[: dynamic_quota[cls]]
            dropped_by_class[cls] = max(0, len(ranked) - len(keep))
            post_quota.extend(keep)

        post_quota_sorted = sorted(post_quota, key=_rank_key)
        capped = post_quota_sorted[:global_cap]
        dropped_global = max(0, len(post_quota_sorted) - len(capped))

        # Global cap sonrası düşenleri sınıfa yaz.
        if dropped_global > 0:
            for det in post_quota_sorted[global_cap:]:
                cls = str(det.get("cls", ""))
                if cls in dropped_by_class:
                    dropped_by_class[cls] += 1

        raw_count = len(normalized_objects)
        post_quota_count = len(post_quota_sorted)
        post_global_cap_count = len(capped)
        dropped_total = max(0, raw_count - post_global_cap_count)
        stats = {
            "frame_id": self._normalize_frame_key(frame_id),
            "raw_count": raw_count,
            "post_quota_count": post_quota_count,
            "post_global_cap_count": post_global_cap_count,
            "dropped_total": dropped_total,
            "dropped_count_by_class": dropped_by_class,
            "base_class_quota": class_quota,
            "dynamic_class_quota": dynamic_quota,
            "class_counts": class_counts,
            "dropped_after_quota": {
                cls: max(0, class_counts[cls] - dynamic_quota[cls])
                for cls in class_order
            },
        }

        if dropped_total > 0:
            self._record_clip_stats(stats)
            self.log.warn(
                "Payload clip applied: "
                f"frame_id={frame_id} raw_count={raw_count} "
                f"post_quota_count={post_quota_count} post_global_cap_count={post_global_cap_count} "
                f"dynamic_class_quota={dynamic_quota} dropped_count_by_class={dropped_by_class}"
            )

        return capped, stats

    @staticmethod
    def _build_safe_fallback_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
        tx = 0.0
        ty = 0.0
        tz = 0.0
        translations = payload.get("detected_translations", [])
        if isinstance(translations, list) and len(translations) > 0:
            first_trans = translations[0]
            if isinstance(first_trans, dict):
                tx = float(first_trans.get("translation_x", 0.0))
                ty = float(first_trans.get("translation_y", 0.0))
                tz = float(first_trans.get("translation_z", 0.0))

        return {
            "id": payload.get("id", "unknown"),
            "user": payload.get("user", Settings.TEAM_NAME),
            "frame": payload.get("frame", payload.get("id", "unknown")),
            "detected_objects": [],
            "detected_translations": [
                {
                    "translation_x": tx,
                    "translation_y": ty,
                    "translation_z": tz,
                }
            ],
            "detected_undefined_objects": [],
        }

    def consume_timeout_counters(self) -> Dict[str, int]:
        snapshot = self._timeout_counters
        self._timeout_counters = dict.fromkeys(snapshot, 0)
        return snapshot

    def consume_payload_guard_counters(self) -> Dict[str, int]:
        snapshot = self._payload_guard_counters
        self._payload_guard_counters = dict.fromkeys(snapshot, 0)
        return snapshot

    def _increment_timeout_counter(self, key: str) -> None:
        if key in self._timeout_counters:
            self._timeout_counters[key] += 1

    @staticmethod
    def _normalize_frame_key(frame_id: Any) -> str:
        if frame_id is None:
            return "unknown"
        return str(frame_id).strip() or "unknown"

    def _mark_seen_frame(self, frame_id: Any) -> bool:
        key = self._normalize_frame_key(frame_id)
        if key in self._seen_frames_lru:
            self._seen_frames_lru.move_to_end(key)
            return True
        self._touch_lru(self._seen_frames_lru, key)
        return False

    def _was_already_submitted(self, frame_key: str) -> bool:
        if frame_key in self._submitted_frames_lru:
            self._submitted_frames_lru.move_to_end(frame_key)
            return True
        return False

    def _mark_submitted(self, frame_key: str) -> None:
        self._touch_lru(self._submitted_frames_lru, frame_key)

    def _should_force_fallback(self, frame_key: str) -> bool:
        if frame_key in self._force_fallback_frames_lru:
            self._force_fallback_frames_lru.move_to_end(frame_key)
            return True
        return False

    def _mark_force_fallback(self, frame_key: str) -> None:
        self._touch_lru(self._force_fallback_frames_lru, frame_key)

    def _unmark_force_fallback(self, frame_key: str) -> None:
        self._force_fallback_frames_lru.pop(frame_key, None)

    def _touch_lru(self, lru: "OrderedDict[str, None]", key: str) -> None:
        lru[key] = None
        lru.move_to_end(key)
        max_size = max(1, int(getattr(Settings, "SEEN_FRAME_LRU_SIZE", 512)))
        while len(lru) > max_size:
            lru.popitem(last=False)

    def _record_clip_event(self, payload_clipped: bool) -> None:
        self._clip_ratio_window.append(1 if payload_clipped else 0)
        if len(self._clip_ratio_window) < self._clip_ratio_window.maxlen:
            return
        ratio = sum(self._clip_ratio_window) / float(len(self._clip_ratio_window))
        if ratio > 0.2:
            self.log.error(
                f"CRITICAL payload clip ratio high in last 100 frames: ratio={ratio:.2f}"
            )

    def _record_clip_stats(self, stats: Dict[str, Any]) -> None:
        frame_stats = {
            "frame_id": stats.get("frame_id"),
            "raw_count": int(stats.get("raw_count", 0)),
            "post_global_cap_count": int(stats.get("post_global_cap_count", 0)),
            "dropped_total": int(stats.get("dropped_total", 0)),
            "class_counts": dict(stats.get("class_counts", {})),
            "base_class_quota": dict(stats.get("base_class_quota", {})),
            "dynamic_class_quota": dict(stats.get("dynamic_class_quota", {})),
            "dropped_count_by_class": dict(stats.get("dropped_count_by_class", {})),
            "dropped_after_quota": dict(stats.get("dropped_after_quota", {})),
        }
        self._clip_frame_events.append(frame_stats)

        self._clip_aggregate_stats["frames_with_clip"] += 1
        self._clip_aggregate_stats["total_dropped"] += frame_stats["dropped_total"]
        self._clip_aggregate_stats["total_raw_objects"] += frame_stats["raw_count"]
        self._clip_aggregate_stats["total_post_cap_objects"] += frame_stats["post_global_cap_count"]
        for cls, count in frame_stats["dropped_count_by_class"].items():
            if cls in self._clip_aggregate_stats["dropped_by_class"]:
                self._clip_aggregate_stats["dropped_by_class"][cls] += int(count)

    def export_clip_tuning_report(self) -> Optional[str]:
        if not self._clip_frame_events:
            self.log.info("Payload clip tuning report skipped: no clipping observed.")
            return None

        aggregate = dict(self._clip_aggregate_stats)
        frames_with_clip = max(1, int(aggregate.get("frames_with_clip", 0)))
        aggregate["avg_dropped_per_clipped_frame"] = (
            float(aggregate.get("total_dropped", 0)) / float(frames_with_clip)
        )
        aggregate["avg_raw_per_clipped_frame"] = (
            float(aggregate.get("total_raw_objects", 0)) / float(frames_with_clip)
        )

        report = {
            "session_id": self._session_id,
            "generated_at_epoch": int(time.time()),
            "global_cap": max(0, self._safe_int(getattr(Settings, "RESULT_MAX_OBJECTS", 100))),
            "base_class_quota": dict(getattr(Settings, "RESULT_CLASS_QUOTA", {})),
            "aggregate": aggregate,
            "frames": self._clip_frame_events,
        }
        tag = f"clip_tuning_report_session_{self._session_id}"
        log_json_to_disk(report, direction="report", tag=tag)

        safe_tag = "".join(ch if str(ch).isalnum() or ch in "._-" else "_" for ch in tag)[:80] or "general"
        report_path = f"{Settings.LOG_DIR}/*_report_{safe_tag}.json"
        self.log.info(
            "Payload clip tuning report generated: "
            f"frames_with_clip={aggregate.get('frames_with_clip', 0)} "
            f"total_dropped={aggregate.get('total_dropped', 0)} path_pattern={report_path}"
        )
        return report_path

    def _build_idempotency_key(self, frame_key: str) -> str:
        prefix = str(getattr(Settings, "IDEMPOTENCY_KEY_PREFIX", "aia")).strip() or "aia"
        import uuid
        if not hasattr(self, "_run_uuid"):
            self._run_uuid = uuid.uuid4().hex[:8]
        return f"{prefix}:{self._session_id}:{self._run_uuid}:{frame_key}"

    def _timeout_tuple(self, read_timeout: float) -> Tuple[float, float]:
        connect_timeout = self._connect_timeout()
        read_timeout = max(0.1, float(read_timeout))
        return (connect_timeout, read_timeout)

    def _connect_timeout(self) -> float:
        raw = getattr(Settings, "REQUEST_CONNECT_TIMEOUT_SEC", Settings.REQUEST_TIMEOUT)
        return max(0.1, float(raw))

    def _read_timeout_frame_meta(self) -> float:
        raw = getattr(
            Settings,
            "REQUEST_READ_TIMEOUT_SEC_FRAME_META",
            Settings.REQUEST_TIMEOUT,
        )
        return max(0.1, float(raw))

    def _read_timeout_image(self) -> float:
        raw = getattr(
            Settings,
            "REQUEST_READ_TIMEOUT_SEC_IMAGE",
            Settings.REQUEST_TIMEOUT,
        )
        return max(0.1, float(raw))

    def _read_timeout_submit(self) -> float:
        raw = getattr(
            Settings,
            "REQUEST_READ_TIMEOUT_SEC_SUBMIT",
            Settings.REQUEST_TIMEOUT,
        )
        return max(0.1, float(raw))

    def _sleep_with_backoff(self, attempt: int) -> None:
        time.sleep(self._compute_backoff_delay(attempt))

    def _compute_backoff_delay(self, attempt: int) -> float:
        capped_attempt = max(1, int(attempt))
        base = float(getattr(Settings, "BACKOFF_BASE_SEC", Settings.RETRY_DELAY))
        max_delay = float(getattr(Settings, "BACKOFF_MAX_SEC", 5.0))
        jitter_ratio = float(getattr(Settings, "BACKOFF_JITTER_RATIO", 0.25))
        base = max(0.01, base)
        max_delay = max(base, max_delay)
        jitter_ratio = min(max(jitter_ratio, 0.0), 1.0)

        delay = min(max_delay, base * (2 ** (capped_attempt - 1)))
        jitter_min = max(0.0, 1.0 - jitter_ratio)
        jitter_max = 1.0 + jitter_ratio
        jittered = delay * random.uniform(jitter_min, jitter_max)
        return min(max_delay, max(0.0, jittered))

    @staticmethod
    def _clamp_bbox(
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        frame_w: int,
        frame_h: int,
    ) -> tuple:
        max_x = max(frame_w - 1, 0)
        max_y = max(frame_h - 1, 0)

        x1 = max(0, min(x1, max_x))
        y1 = max(0, min(y1, max_y))
        x2 = max(0, min(x2, max_x))
        y2 = max(0, min(y2, max_y))

        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        return x1, y1, x2, y2

    @staticmethod
    def _safe_int(val: Any) -> int:
        try:
            return int(float(val))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _safe_float(val: Any) -> float:
        try:
            return float(val)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _should_log_json(counter: int) -> bool:
        if not Settings.ENABLE_JSON_LOGGING:
            return False
        interval = max(1, int(Settings.JSON_LOG_EVERY_N_FRAMES))
        return counter % interval == 0
