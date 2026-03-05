"""TEKNOFEST Havacılıkta Yapay Zeka — Ana orkestrasyon.
Simülasyon: datasets/ içinden kare yükler. Yarışma: sunucudan frame alır, sonuç gönderir.
"""

import argparse
import os
import signal
import sys
import time
from collections import Counter, deque
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests
import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.settings import Settings  # noqa: E402
from src.detection import ObjectDetector  # noqa: E402
from src.localization import VisualOdometry  # noqa: E402
from src.movement import MovementEstimator  # noqa: E402
from src.competition_contract import (  # noqa: E402
    DataContractError,
    ErrorDecision,
    ErrorPolicy,
    FatalSystemError,
    RecoverableIOError,
)
from src.resilience import SessionResilienceController  # noqa: E402
from src.runtime_profile import apply_runtime_profile  # noqa: E402
from src.send_state import apply_send_result_status  # noqa: E402
from src.utils import Logger, Visualizer, log_json_to_disk, get_display_size  # noqa: E402
from src.utils import FrameContext  # noqa: E402
from src.utils import normalize_gps_health  # noqa: E402
from src.flow_policy import (  # noqa: E402
    DuplicateStormAction,
    FetchStrategy,
    FrameLifecycleState,
    decide_degrade_fetch_strategy,
    decide_duplicate_storm_action,
)
from src.task3_reference_policy import (  # noqa: E402
    canonicalize_task3_references,
)

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║     🛩️  TEKNOFEST 2026 - HAVACILIKTA YAPAY ZEKA YARIŞMASI    ║
║     ──────────────────────────────────────────────────────    ║
║     Nesne Tespiti (Görev 1) + Konum Kestirimi (Görev 2)     ║
║     Görüntü Eşleme (Görev 3)                                ║
╚══════════════════════════════════════════════════════════════╝
"""


def print_system_info(log: Logger, runtime_mode_label: str) -> None:
    print(BANNER)
    log.info(f"Working Directory : {PROJECT_ROOT}")
    log.info(f"Mode             : {runtime_mode_label}")
    log.info(f"Debug            : {'ON' if Settings.DEBUG else 'OFF'}")
    log.info(f"Model            : {Settings.MODEL_PATH}")
    log.info(f"Device           : {Settings.DEVICE}")
    log.info(f"FP16             : {'ON' if Settings.HALF_PRECISION else 'OFF'}")
    log.info(f"TTA              : {'ON' if Settings.AUGMENTED_INFERENCE else 'OFF'}")

    if torch.cuda.is_available():
        log.success(f"GPU              : {torch.cuda.get_device_name(0)}")
        mem_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        log.success(f"GPU Memory       : {mem_total:.1f} GB")
    else:
        log.warn("GPU              : NOT FOUND, running on CPU")


class FPSCounter:
    def __init__(self, report_interval: int = 10) -> None:
        self.report_interval = report_interval
        self.frame_count: int = 0
        self.start_time: float = time.time()
        self.log = Logger("FPS")

    def tick(self) -> Optional[float]:
        self.frame_count += 1
        if self.frame_count % self.report_interval == 0:
            elapsed = time.time() - self.start_time
            fps = self.frame_count / elapsed if elapsed > 0 else 0
            self.log.info(
                f"Frame: {self.frame_count} | FPS: {fps:.2f} | Elapsed: {elapsed:.1f}s"
            )
            return fps
        return None


def run_simulation(
    log: Logger,
    prefer_vid: bool = True,
    show: bool = False,
    save: bool = False,
    seed: Optional[int] = None,
    sequence: Optional[str] = None,
) -> None:
    from src.data_loader import DatasetLoader

    log.info("Initializing modules...")

    try:
        loader = DatasetLoader(prefer_vid=prefer_vid, seed=seed, sequence=sequence)
        if not loader.is_ready:
            log.error("Dataset loading failed, exiting.")
            return

        detector = ObjectDetector()
        odometry = VisualOdometry()
        movement = MovementEstimator()
        fps_counter = FPSCounter(report_interval=Settings.FPS_REPORT_INTERVAL)

        image_matcher = None
        if Settings.TASK3_ENABLED:
            from src.image_matcher import ImageMatcher

            image_matcher = ImageMatcher()
            loaded = image_matcher.load_references_from_directory()
            if loaded == 0:
                log.warn(
                    "Görev 3: Referans obje bulunamadı, Simülasyonda Görev 3 pasif"
                )
                image_matcher = None

        visualizer = Visualizer()
        if save:
            os.makedirs(Settings.DEBUG_OUTPUT_DIR, exist_ok=True)
            log.info(f"Saving frames to: {Settings.DEBUG_OUTPUT_DIR}")

        log.success("All modules initialized successfully")

    except (ImportError, RuntimeError, OSError, ValueError) as exc:
        log.error(f"Initialization error: {exc}")
        return

    running = True

    def signal_handler(sig, frame) -> None:
        nonlocal running
        running = False
        log.warn("Shutdown signal received, stopping loop...")

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    fullscreen_done: List[bool] = [False] if show else []

    try:
        for frame_info in loader:
            if not running:
                break

            if fps_counter.frame_count >= Settings.MAX_FRAMES:
                log.success(f"Max frame limit reached ({Settings.MAX_FRAMES})")
                break

            try:
                should_stop = _process_simulation_step(
                    log,
                    frame_info,
                    detector,
                    movement,
                    odometry,
                    image_matcher,
                    visualizer,
                    show,
                    save,
                    fullscreen_done=fullscreen_done,
                )
                if should_stop:
                    break
                fps_counter.tick()

            except Exception as exc:
                log.error(f"Frame {frame_info.get('frame_idx', '?')} error: {exc}")
                continue

    finally:
        log.info("Cleaning resources...")
        if show:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        if save:
            log.success(f"Frames saved: {Settings.DEBUG_OUTPUT_DIR}/")

        _print_summary(log, fps_counter)


def _process_simulation_step(
    log: Logger,
    frame_info: dict,
    detector: ObjectDetector,
    movement: MovementEstimator,
    odometry: VisualOdometry,
    image_matcher: Any,
    visualizer: Any,
    show: bool,
    save: bool,
    fullscreen_done: Optional[List[bool]] = None,
) -> bool:
    frame = frame_info["frame"]
    frame_idx = frame_info["frame_idx"]
    server_data = frame_info["server_data"]
    gps_health = frame_info["gps_health"]

    frame_ctx = FrameContext(frame)
    position = odometry.update(frame_ctx, server_data)
    current_z = position.get("z", 50.0) if position else 50.0
    detected_objects = detector.detect(frame, altitude=current_z)
    detected_objects = movement.annotate(detected_objects, frame_ctx=frame_ctx)

    if image_matcher is not None:
        _ = image_matcher.match(frame)

    _print_simulation_result(log, frame_idx, detected_objects, position, gps_health)

    if show or save:
        try:
            guardrail_stats = getattr(detector, "_last_guardrail_stats", None) or {}
            annotated = visualizer.draw_detections(
                frame,
                detected_objects,
                frame_id=str(frame_idx),
                position=position,
                save_to_disk=not save,
                guardrail_stats=guardrail_stats,
                gps_health=gps_health,
            )
        except Exception as exc:
            log.warn(f"Frame {frame_idx}: debug visualizer failed (fail-open): {exc}")
            annotated = frame.copy()

        mode_text = (
            "GPS"
            if gps_health == 1
            else ("Optical Flow" if gps_health == 0 else "GPS Unknown")
        )
        cv2.putText(
            annotated,
            f"Mode: {mode_text}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0) if gps_health == 1 else (0, 165, 255),
            2,
        )

        # GPS sağlıksız — duraklatma mesajı (SIMULATION_PAUSE_ON_GPS_LOSS aktifse)
        pause_on_gps_loss = getattr(Settings, "SIMULATION_PAUSE_ON_GPS_LOSS", False)
        if gps_health == 0 and pause_on_gps_loss and show:
            h_a, w_a = annotated.shape[:2]
            pause_text = "GPS sagliksiz — SPACE ile devam"
            cv2.putText(
                annotated,
                pause_text,
                (w_a // 2 - 180, h_a // 2 + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 165, 255),
                2,
            )

        if show:
            try:
                display_img = annotated
                max_w, max_h = get_display_size()
                h, w = display_img.shape[:2]
                if w > max_w or h > max_h:
                    scale = min(max_w / w, max_h / h)
                    display_img = cv2.resize(display_img, (int(w * scale), int(h * scale)))

                cv2.namedWindow("TEKNOFEST - Simulation", cv2.WINDOW_NORMAL)
                # Tam ekran (Show window seçildiğinde)
                if fullscreen_done and not fullscreen_done[0]:
                    try:
                        cv2.setWindowProperty(
                            "TEKNOFEST - Simulation",
                            cv2.WND_PROP_FULLSCREEN,
                            cv2.WINDOW_FULLSCREEN,
                        )
                        fullscreen_done[0] = True
                    except cv2.error:
                        pass
                cv2.imshow("TEKNOFEST - Simulation", display_img)

                # GPS sağlıksızken duraklat — SPACE ile devam
                if gps_health == 0 and pause_on_gps_loss:
                    while True:
                        key = cv2.waitKey(100) & 0xFF
                        if key in (ord("q"), 27):
                            log.info("Window closed by user (q/ESC)")
                            return True
                        if key == ord(" "):
                            break
                else:
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        log.info("Window closed by user (q/ESC)")
                        return True
            except cv2.error:
                pass

        if save:
            save_path = os.path.join(
                Settings.DEBUG_OUTPUT_DIR,
                f"frame_{frame_idx:04d}.jpg",
            )
            cv2.imwrite(save_path, annotated)

    return False


def _print_simulation_result(
    log: Logger,
    frame_idx: int,
    detected_objects: list,
    position: dict,
    gps_health: Optional[int],
) -> None:
    cls_counts = Counter(obj["cls"] for obj in detected_objects)
    tasit = cls_counts.get("0", 0)
    insan = cls_counts.get("1", 0)
    uap = cls_counts.get("2", 0)
    uai = cls_counts.get("3", 0)

    loc_mode = "GPS" if gps_health == 1 else ("OF" if gps_health == 0 else "UNKNOWN")
    pos_str = (
        f"x={position['x']:+.1f}m "
        f"y={position['y']:+.1f}m "
        f"z={position['z']:.0f}m"
    )

    log.success(
        f"Frame: {frame_idx:04d} | "
        f"Det: {len(detected_objects)} "
        f"({tasit} Vehicle, {insan} Human"
        f"{f', {uap} UAP' if uap else ''}"
        f"{f', {uai} UAI' if uai else ''}) | "
        f"Pos: {pos_str} ({loc_mode})"
    )


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_gps_health(frame_data: Dict[str, Any]) -> Optional[int]:
    value, _ = normalize_gps_health(
        frame_data.get("gps_health"),
        gps_health_status=frame_data.get("gps_health_status"),
    )
    return value


def _assert_camera_calibration_ready(log: Logger) -> None:
    if not bool(getattr(Settings, "CAMERA_CALIBRATION_GUARD_ENABLED", True)):
        return

    focal = _safe_float(getattr(Settings, "FOCAL_LENGTH_PX", 0.0))
    cx = _safe_float(getattr(Settings, "CAMERA_CX", -1.0), default=-1.0)
    cy = _safe_float(getattr(Settings, "CAMERA_CY", -1.0), default=-1.0)

    checks = {
        "FOCAL_LENGTH_PX": focal > 0.0 and np.isfinite(focal),
        "CAMERA_CX": cx >= 0.0 and np.isfinite(cx),
        "CAMERA_CY": cy >= 0.0 and np.isfinite(cy),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise DataContractError(
            "Camera calibration guard failed: "
            f"invalid params={failed} values="
            f"FOCAL_LENGTH_PX={focal}, CAMERA_CX={cx}, CAMERA_CY={cy}"
        )

    if focal == 800.0 and cx == 960.0 and cy == 540.0:
        log.warn(
            "Camera calibration guard: default camera parameters are active "
            "(FOCAL_LENGTH_PX=800.0, CAMERA_CX=960.0, CAMERA_CY=540.0)."
        )


def _validate_task3_references(
    log: Logger,
    server_refs: List[Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, int], str, str, bool]:
    return canonicalize_task3_references(
        log=log,
        references=server_refs,
        project_root=PROJECT_ROOT,
    )


def _apply_latency_compensation_if_needed(
    log: Logger,
    odometry: VisualOdometry,
    pending_result: Dict[str, Any],
    kpi_counters: Dict[str, Any],
) -> tuple[Dict[str, float], Dict[str, float]]:
    base_translation = dict(pending_result["detected_translation"])
    base_position = dict(pending_result.get("position", {}))

    if not Settings.LATENCY_COMP_ENABLED:
        return base_translation, base_position

    frame_data = pending_result.get("frame_data") or {}
    if _safe_gps_health(frame_data) != 0:
        return base_translation, base_position

    frame_fetch_monotonic = pending_result.get("frame_fetch_monotonic")
    if frame_fetch_monotonic is None:
        return base_translation, base_position

    dt_sec = max(0.0, time.monotonic() - _safe_float(frame_fetch_monotonic))
    max_dt_sec = max(0.0, _safe_float(Settings.LATENCY_COMP_MAX_MS)) / 1000.0
    max_delta_m = max(0.0, _safe_float(Settings.LATENCY_COMP_MAX_DELTA_M))

    projection_base = dict(pending_result.get("base_position", base_position))
    projected_position, applied_delta_m, used_dt_sec = (
        odometry.project_position_with_latency(
            position=projection_base,
            dt_sec=dt_sec,
            max_dt_sec=max_dt_sec,
            max_delta_m=max_delta_m,
        )
    )

    if applied_delta_m <= 0.0:
        return base_translation, base_position

    kpi_counters["compensation_apply_count"] = (
        int(kpi_counters.get("compensation_apply_count", 0)) + 1
    )
    kpi_counters["compensation_sum_delta_m"] = (
        _safe_float(kpi_counters.get("compensation_sum_delta_m", 0.0)) + applied_delta_m
    )
    apply_count = max(1, int(kpi_counters["compensation_apply_count"]))
    kpi_counters["compensation_avg_delta_m"] = (
        _safe_float(kpi_counters["compensation_sum_delta_m"]) / apply_count
    )
    kpi_counters["compensation_max_delta_m"] = max(
        _safe_float(kpi_counters.get("compensation_max_delta_m", 0.0)),
        applied_delta_m,
    )

    frame_id = pending_result.get("frame_id", "unknown")
    log.debug(
        f"Frame {frame_id}: latency compensation applied "
        f"(dt={dt_sec * 1000.0:.1f}ms, used={used_dt_sec * 1000.0:.1f}ms, delta={applied_delta_m:.3f}m)"
    )

    compensated_translation = {
        "translation_x": projected_position["x"],
        "translation_y": projected_position["y"],
        "translation_z": projected_position["z"],
    }
    return compensated_translation, projected_position


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _rolling_fps_from_durations(frame_cycle_window: deque) -> float:
    if not frame_cycle_window:
        return 0.0
    total = sum(frame_cycle_window)
    if total <= 1e-9:
        return 0.0
    return float(len(frame_cycle_window)) / float(total)


def _update_dynamic_json_log_interval(
    log: Logger,
    rolling_fps: float,
    kpi_counters: Dict[str, Any],
) -> None:
    if not bool(getattr(Settings, "DYNAMIC_JSON_LOG_INTERVAL_ENABLED", True)):
        return
    prev = max(1, int(getattr(Settings, "JSON_LOG_EVERY_N_FRAMES", 1)))
    slow = max(1, int(getattr(Settings, "DYNAMIC_JSON_LOG_SLOW_INTERVAL", 40)))
    medium = max(1, int(getattr(Settings, "DYNAMIC_JSON_LOG_MEDIUM_INTERVAL", 20)))
    fast = max(1, int(getattr(Settings, "DYNAMIC_JSON_LOG_FAST_INTERVAL", 10)))

    if rolling_fps < 1.0:
        target = slow
    elif rolling_fps < 2.0:
        target = medium
    else:
        target = fast

    if target != prev:
        Settings.JSON_LOG_EVERY_N_FRAMES = target
        log.info(
            f"event=json_log_interval_adjusted old={prev} new={target} rolling_fps={rolling_fps:.2f}"
        )
    kpi_counters["json_log_interval"] = int(Settings.JSON_LOG_EVERY_N_FRAMES)


def _maybe_toggle_low_fps_guard(
    log: Logger,
    rolling_fps: float,
    guard_state: Dict[str, Any],
    kpi_counters: Dict[str, Any],
) -> None:
    if not bool(getattr(Settings, "LOW_FPS_GUARD_ENABLED", True)):
        return

    active = bool(guard_state.get("active", False))
    low_threshold = float(getattr(Settings, "LOW_FPS_GUARD_THRESHOLD", 1.0))
    recover_threshold = float(getattr(Settings, "LOW_FPS_GUARD_RECOVERY_THRESHOLD", 1.4))
    recover_streak_needed = max(
        1, int(getattr(Settings, "LOW_FPS_GUARD_RECOVERY_STREAK", 12))
    )

    if not active and rolling_fps > 0.0 and rolling_fps < low_threshold:
        guard_state["active"] = True
        guard_state["recovery_streak"] = 0

        Settings.SAHI_ENABLED = not bool(getattr(Settings, "PROTECTIVE_DISABLE_SAHI", True))
        Settings.INFERENCE_SIZE = min(
            int(Settings.INFERENCE_SIZE),
            max(256, int(getattr(Settings, "PROTECTIVE_INFERENCE_SIZE", 960))),
        )
        Settings.MAX_DETECTIONS = min(
            int(Settings.MAX_DETECTIONS),
            max(1, int(getattr(Settings, "PROTECTIVE_MAX_DETECTIONS", 180))),
        )
        Settings.CONFIDENCE_THRESHOLD = max(
            float(Settings.CONFIDENCE_THRESHOLD),
            float(getattr(Settings, "PROTECTIVE_CONFIDENCE_THRESHOLD", 0.50)),
        )
        Settings.AUGMENTED_INFERENCE = False
        Settings.DEGRADE_SEND_INTERVAL_FRAMES = max(
            int(Settings.DEGRADE_SEND_INTERVAL_FRAMES),
            max(1, int(getattr(Settings, "PROTECTIVE_DEGRADE_SEND_INTERVAL_FRAMES", 8))),
        )
        Settings.JSON_LOG_EVERY_N_FRAMES = max(
            int(Settings.JSON_LOG_EVERY_N_FRAMES),
            max(1, int(getattr(Settings, "PROTECTIVE_LOG_INTERVAL", 25))),
        )

        kpi_counters["fps_guard_activations"] = (
            int(kpi_counters.get("fps_guard_activations", 0)) + 1
        )
        log.warn(
            f"event=fps_guard_activated rolling_fps={rolling_fps:.2f} "
            f"sahi={'on' if Settings.SAHI_ENABLED else 'off'} "
            f"imgsz={Settings.INFERENCE_SIZE} max_det={Settings.MAX_DETECTIONS}"
        )
        return

    if not active:
        return

    if rolling_fps >= recover_threshold:
        guard_state["recovery_streak"] = int(guard_state.get("recovery_streak", 0)) + 1
    else:
        guard_state["recovery_streak"] = 0

    if int(guard_state["recovery_streak"]) < recover_streak_needed:
        return

    guard_state["active"] = False
    guard_state["recovery_streak"] = 0
    Settings.SAHI_ENABLED = bool(guard_state["orig_sahi"])
    Settings.INFERENCE_SIZE = int(guard_state["orig_inference_size"])
    Settings.MAX_DETECTIONS = int(guard_state["orig_max_det"])
    Settings.CONFIDENCE_THRESHOLD = float(guard_state["orig_conf"])
    Settings.AUGMENTED_INFERENCE = bool(
        guard_state.get("orig_augmented", Settings.AUGMENTED_INFERENCE)
    )
    Settings.JSON_LOG_EVERY_N_FRAMES = int(guard_state["orig_json_interval"])
    Settings.DEGRADE_SEND_INTERVAL_FRAMES = int(guard_state["orig_degrade_interval"])
    kpi_counters["fps_guard_recoveries"] = (
        int(kpi_counters.get("fps_guard_recoveries", 0)) + 1
    )
    log.info(f"event=fps_guard_recovered rolling_fps={rolling_fps:.2f}")


def _run_periodic_gpu_maintenance(
    log: Logger,
    processed_frames: int,
    kpi_counters: Dict[str, Any],
) -> None:
    interval = max(1, int(getattr(Settings, "GPU_CLEANUP_INTERVAL", 200)))
    if processed_frames <= 0 or processed_frames % interval != 0:
        return

    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated(0) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(0) / (1024 ** 2)
        kpi_counters["gpu_memory_allocated_mb"] = round(float(allocated), 2)
        kpi_counters["gpu_memory_reserved_mb"] = round(float(reserved), 2)
        log.info(
            "event=gpu_maintenance "
            f"processed_frames={processed_frames} allocated_mb={allocated:.1f} reserved_mb={reserved:.1f}"
        )
    else:
        log.info(f"event=gpu_maintenance processed_frames={processed_frames} mode=cpu")

    kpi_counters["gpu_maintenance_runs"] = int(
        kpi_counters.get("gpu_maintenance_runs", 0)
    ) + 1


def run_competition(log: Logger) -> None:
    from src.network import NetworkManager

    log.info("Initializing modules...")

    try:
        network = NetworkManager(simulation_mode=False)
        detector = ObjectDetector()
        odometry = VisualOdometry()
        movement = MovementEstimator()
        fps_counter = FPSCounter(report_interval=Settings.FPS_REPORT_INTERVAL)

        image_matcher = None
        if Settings.TASK3_ENABLED:
            from src.image_matcher import ImageMatcher

            image_matcher = ImageMatcher()
            loaded = image_matcher.load_references_from_directory()
            if loaded == 0:
                log.warn(
                    "Görev 3: Referans obje bulunamadı, detected_undefined_objects boş gönderilecek"
                )

        visualizer: Optional[Visualizer] = Visualizer() if Settings.DEBUG else None

        log.success("All modules initialized successfully")

    except (ImportError, RuntimeError, OSError, ValueError) as exc:
        log.error(f"Initialization error: {exc}")
        log.error("System startup failed, exiting.")
        return

    try:
        _assert_camera_calibration_ready(log)
    except DataContractError as exc:
        log.error(f"Calibration guard failed: {exc}")
        return

    if not network.start_session():
        log.error("Server session start failed")
        return
    if hasattr(network, "assert_contract_ready"):
        try:
            network.assert_contract_ready()
        except DataContractError as exc:
            log.error(f"Payload schema self-check failed: {exc}")
            return

    reference_validation_stats: Dict[str, int] = {
        "total": 0,
        "valid": 0,
        "duplicate": 0,
        "quarantined": 0,
    }
    id_integrity_mode = "normal"
    id_integrity_reason_code = "no_server_references"

    server_refs = network.get_task3_references()
    if image_matcher is not None and server_refs:
        (
            canonical_refs,
            reference_validation_stats,
            id_integrity_mode,
            id_integrity_reason_code,
            duplicate_critical,
        ) = _validate_task3_references(log=log, server_refs=server_refs)
        if canonical_refs:
            loaded = image_matcher.load_references(canonical_refs)
            if loaded == 0:
                log.warn(
                    "Görev 3: Doğrulanan referanslar yüklense de matcher aktifleşmedi"
                )
            else:
                log.info(
                    f"Görev 3: Sunucudan {loaded} canonical referans obje yüklendi"
                )
            if duplicate_critical:
                log.warn(
                    "Görev 3: ID integrity kritik eşiği aşıldı, ancak matcher aktif tutuluyor "
                    "(task3_disable_overridden=True)"
                )
        else:
            image_matcher = None
            log.warn("Görev 3: Geçerli referans kalmadı, kontrollü pasif mod")

    running = True
    transient_failures = 0
    transient_budget = max(10, Settings.MAX_RETRIES * 5)
    ack_failures = 0
    ack_failure_budget = max(20, Settings.MAX_RETRIES * 10)
    consecutive_permanent_rejects = 0
    PERMANENT_REJECT_ABORT_THRESHOLD = 5
    consecutive_duplicate_frames = 0
    CONSECUTIVE_DUPLICATE_ABORT_THRESHOLD = max(
        1, int(getattr(Settings, "DUPLICATE_STORM_THRESHOLD", 5))
    )
    duplicate_storm_action_cfg = str(
        getattr(Settings, "DUPLICATE_STORM_ACTION", "terminate_session")
    )
    resilience = SessionResilienceController(log=log)
    error_policy = ErrorPolicy(
        retry_budget=max(5, Settings.MAX_RETRIES * 2),
        degrade_budget=3,
    )
    frame_state = FrameLifecycleState.IDLE
    current_frame_id = "none"
    degrade_replay_state: Dict[str, Any] = {
        "objects": [],
        "age": 10**9,
    }
    degrade_fallback_window: deque = deque(
        maxlen=max(5, int(getattr(Settings, "DEGRADE_FALLBACK_RATIO_WINDOW", 40)))
    )
    frame_cycle_window: deque = deque(
        maxlen=max(5, int(getattr(Settings, "LOW_FPS_GUARD_WINDOW", 20)))
    )
    low_fps_guard_state: Dict[str, Any] = {
        "active": False,
        "recovery_streak": 0,
        "orig_sahi": bool(Settings.SAHI_ENABLED),
        "orig_inference_size": int(Settings.INFERENCE_SIZE),
        "orig_max_det": int(Settings.MAX_DETECTIONS),
        "orig_conf": float(Settings.CONFIDENCE_THRESHOLD),
        "orig_augmented": bool(Settings.AUGMENTED_INFERENCE),
        "orig_json_interval": int(Settings.JSON_LOG_EVERY_N_FRAMES),
        "orig_degrade_interval": int(Settings.DEGRADE_SEND_INTERVAL_FRAMES),
    }

    valid_transitions = {
        FrameLifecycleState.IDLE: {
            FrameLifecycleState.FETCHED,
            FrameLifecycleState.TERMINAL,
        },
        FrameLifecycleState.FETCHED: {
            FrameLifecycleState.PROCESSED,
            FrameLifecycleState.TERMINAL,
        },
        FrameLifecycleState.PROCESSED: {
            FrameLifecycleState.SUBMITTING,
            FrameLifecycleState.TERMINAL,
        },
        FrameLifecycleState.SUBMITTING: {
            FrameLifecycleState.ACKED,
            FrameLifecycleState.TERMINAL,
        },
        FrameLifecycleState.ACKED: {
            FrameLifecycleState.IDLE,
            FrameLifecycleState.TERMINAL,
        },
        FrameLifecycleState.TERMINAL: {FrameLifecycleState.TERMINAL},
    }

    def _transition_frame_state(
        new_state: FrameLifecycleState,
        reason_code: str,
        frame_id: Optional[str] = None,
    ) -> None:
        nonlocal frame_state, current_frame_id
        if new_state not in valid_transitions.get(frame_state, set()):
            raise DataContractError(
                f"Invalid frame state transition: {frame_state.value} -> {new_state.value}"
            )

        from_state = frame_state
        frame_state = new_state
        if frame_id is not None:
            current_frame_id = str(frame_id)
        kpi_counters["frame_state"] = frame_state.value
        kpi_counters["frame_state_transitions"] = (
            int(kpi_counters.get("frame_state_transitions", 0)) + 1
        )
        log.info(
            f"event=frame_state_transition from={from_state.value} to={new_state.value} "
            f"frame_id={current_frame_id} reason_code={reason_code}"
        )

    kpi_counters: Dict[str, Any] = {
        "send_ok": 0,
        "send_fail": 0,
        "send_fallback_ok": 0,
        "send_permanent_reject": 0,
        "payload_preflight_reject_count": 0,
        "payload_clipped_count": 0,
        "mode_gps": 0,
        "mode_of": 0,
        "mode_unknown": 0,
        "degrade_frames": 0,
        "degrade_replayed_detection_frames": 0,
        "degrade_empty_payload_count": 0,
        "frame_duplicate_drop": 0,
        "timeout_fetch": 0,
        "timeout_image": 0,
        "timeout_submit": 0,
        "consecutive_duplicate_abort": 0,
        "duplicate_storm_events": 0,
        "frame_state": frame_state.value,
        "frame_state_transitions": 0,
        "state_machine_errors": 0,
        "compensation_apply_count": 0,
        "compensation_sum_delta_m": 0.0,
        "compensation_avg_delta_m": 0.0,
        "compensation_max_delta_m": 0.0,
        "error_decision_retry": 0,
        "error_decision_degrade": 0,
        "error_decision_stop": 0,
        "error_unknown_count": 0,
        "rolling_fps": 0.0,
        "json_log_interval": int(Settings.JSON_LOG_EVERY_N_FRAMES),
        "fps_guard_activations": 0,
        "fps_guard_recoveries": 0,
        "gpu_maintenance_runs": 0,
        "reference_validation_stats": reference_validation_stats,
        "id_integrity_mode": id_integrity_mode,
        "id_integrity_reason_code": id_integrity_reason_code,
    }
    pending_result: Optional[Dict] = None

    def signal_handler(sig, frame) -> None:
        nonlocal running
        running = False
        log.warn("Shutdown signal received (Ctrl+C)")

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    import concurrent.futures

    try:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        fetch_future = None
        submit_future = None

        while running:
            try:
                abort_reason = resilience.should_abort(
                    has_pending_result=(pending_result is not None or submit_future is not None)
                )
                if abort_reason:
                    log.error(f"Resilience abort: {abort_reason}")
                    break

                if fps_counter.frame_count >= Settings.MAX_FRAMES:
                    log.success(
                        f"Max frame count reached ({Settings.MAX_FRAMES}), session complete"
                    )
                    break

                # Bekleyen sonuç varsa önce gönder (fetch ile çakışmayı önle)
                if pending_result is not None and submit_future is None:
                    if frame_state == FrameLifecycleState.PROCESSED:
                        _transition_frame_state(
                            FrameLifecycleState.SUBMITTING,
                            reason_code="begin_submit",
                            frame_id=pending_result.get("frame_id", "unknown"),
                        )
                    elif frame_state != FrameLifecycleState.SUBMITTING:
                        raise DataContractError(
                            f"Pending result exists but frame_state={frame_state.value}"
                        )

                    submit_future = executor.submit(
                        _submit_competition_step,
                        log,
                        network,
                        resilience,
                        odometry,
                        kpi_counters,
                        pending_result,
                        ack_failures,
                        ack_failure_budget,
                        consecutive_permanent_rejects,
                        PERMANENT_REJECT_ABORT_THRESHOLD,
                    )

                if submit_future is not None and submit_future.done():
                    result = submit_future.result()
                    pending_result = result[0]
                    ack_failures = result[1]
                    action_result = result[2]
                    consecutive_permanent_rejects = result[3]
                    submit_future = None
                    if action_result == "break":
                        _transition_frame_state(
                            FrameLifecycleState.TERMINAL,
                            reason_code="submit_terminal_break",
                        )
                        break
                    elif action_result == "continue":
                        continue

                    if not (
                        isinstance(action_result, tuple) and len(action_result) == 2
                    ):
                        kpi_counters["state_machine_errors"] += 1
                        log.error(
                            f"Unexpected action_result type: {type(action_result)}, skipping frame"
                        )
                        if pending_result is None and frame_state == FrameLifecycleState.SUBMITTING:
                            _transition_frame_state(
                                FrameLifecycleState.ACKED,
                                reason_code="unexpected_action_pending_cleared",
                            )
                            _transition_frame_state(
                                FrameLifecycleState.IDLE,
                                reason_code="reset_to_idle_after_unexpected_action",
                            )
                        continue
                    _, success_info = action_result
                    _transition_frame_state(
                        FrameLifecycleState.ACKED,
                        reason_code="ack_received",
                        frame_id=success_info.get("frame_id", "unknown"),
                    )

                    gps_health = _safe_gps_health(success_info["frame_data"])

                    if gps_health == 1:
                        kpi_counters["mode_gps"] += 1
                    elif gps_health == 0:
                        kpi_counters["mode_of"] += 1
                    else:
                        kpi_counters["mode_unknown"] += 1

                    if (
                        Settings.DEBUG
                        and visualizer is not None
                        and success_info["frame_for_debug"] is not None
                    ):
                        try:
                            debug_interval = max(
                                1, int(getattr(Settings, "COMPETITION_DEBUG_DRAW_INTERVAL", 1))
                            )
                            if (fps_counter.frame_count + 1) % debug_interval == 0:
                                visualizer.draw_detections(
                                    success_info["frame_for_debug"],
                                    success_info["detected_objects"],
                                    frame_id=str(success_info["frame_id"]),
                                    position=success_info["position"],
                                    guardrail_stats=success_info.get("guardrail_stats"),
                                )
                        except Exception as exc:
                            log.warn(
                                f"Frame {success_info['frame_id']}: debug draw failed (fail-open): {exc}"
                            )

                    frame_fetch_monotonic = success_info.get("frame_fetch_monotonic")
                    if frame_fetch_monotonic is not None:
                        cycle_sec = max(
                            1e-3,
                            time.monotonic() - _safe_float(frame_fetch_monotonic, default=0.0),
                        )
                        frame_cycle_window.append(cycle_sec)
                        rolling_fps = _rolling_fps_from_durations(frame_cycle_window)
                        kpi_counters["rolling_fps"] = round(rolling_fps, 4)
                        _update_dynamic_json_log_interval(
                            log=log,
                            rolling_fps=rolling_fps,
                            kpi_counters=kpi_counters,
                        )
                        _maybe_toggle_low_fps_guard(
                            log=log,
                            rolling_fps=rolling_fps,
                            guard_state=low_fps_guard_state,
                            kpi_counters=kpi_counters,
                        )

                    fps_counter.tick()
                    _run_periodic_gpu_maintenance(
                        log=log,
                        processed_frames=fps_counter.frame_count,
                        kpi_counters=kpi_counters,
                    )
                    _transition_frame_state(
                        FrameLifecycleState.IDLE,
                        reason_code="ready_for_next_fetch",
                        frame_id=success_info.get("frame_id", "unknown"),
                    )

                    interval = max(1, int(Settings.COMPETITION_RESULT_LOG_INTERVAL))
                    if fps_counter.frame_count % interval == 0:
                        _print_competition_result(
                            log=log,
                            frame_id=success_info["frame_id"],
                            detected_objects=success_info["detected_objects"],
                            send_status="SUCCESS",
                            position=success_info["position"],
                            gps_health=gps_health,
                        )

                    if Settings.LOOP_DELAY > 0:
                        time.sleep(Settings.LOOP_DELAY)

                elif submit_future is not None and not submit_future.done():
                    time.sleep(0.01)
                    continue

                elif pending_result is None:
                    if frame_state != FrameLifecycleState.IDLE:
                        raise DataContractError(
                            f"Fetch requested while frame_state={frame_state.value}"
                        )
                    # Yeni frame al
                    if fetch_future is None:
                        fetch_future = executor.submit(
                            _fetch_competition_step,
                            log,
                            network,
                            detector,
                            movement,
                            odometry,
                            image_matcher,
                            resilience,
                            kpi_counters,
                            transient_failures,
                            transient_budget,
                            degrade_replay_state,
                            degrade_fallback_window,
                        )

                    if fetch_future.done():
                        fetch_res, tf_new, action, is_dup = fetch_future.result()
                        fetch_future = None
                        transient_failures = tf_new
                        if action == "continue":
                            delay_val = (
                                min(max(0.2, Settings.RETRY_DELAY), 5.0)
                                if fetch_res is None
                                else 0
                            )
                            time.sleep(delay_val)
                            continue
                        if action == "break":
                            _transition_frame_state(
                                FrameLifecycleState.TERMINAL,
                                reason_code="fetch_terminal_break",
                            )
                            break

                        if is_dup:
                            consecutive_duplicate_frames += 1
                            log.warn(
                                f"event=duplicate_frame_detected streak={consecutive_duplicate_frames} "
                                f"threshold={CONSECUTIVE_DUPLICATE_ABORT_THRESHOLD} action_cfg={duplicate_storm_action_cfg}"
                            )
                            duplicate_action = decide_duplicate_storm_action(
                                consecutive_duplicates=consecutive_duplicate_frames,
                                threshold=CONSECUTIVE_DUPLICATE_ABORT_THRESHOLD,
                                configured_action=duplicate_storm_action_cfg,
                            )
                            if duplicate_action == DuplicateStormAction.TERMINATE_SESSION:
                                kpi_counters["duplicate_storm_events"] += 1
                                kpi_counters["consecutive_duplicate_abort"] = (
                                    consecutive_duplicate_frames
                                )
                                log.error(
                                    "event=duplicate_storm_terminal_action "
                                    f"streak={consecutive_duplicate_frames} action=terminate_session"
                                )
                                _transition_frame_state(
                                    FrameLifecycleState.TERMINAL,
                                    reason_code="duplicate_storm_terminal",
                                )
                                break
                        else:
                            consecutive_duplicate_frames = 0

                        pending_result = fetch_res
                        if pending_result is not None:
                            fetched_frame_id = str(pending_result.get("frame_id", "unknown"))
                            _transition_frame_state(
                                FrameLifecycleState.FETCHED,
                                reason_code="frame_metadata_fetched",
                                frame_id=fetched_frame_id,
                            )
                            _transition_frame_state(
                                FrameLifecycleState.PROCESSED,
                                reason_code="frame_processed",
                                frame_id=fetched_frame_id,
                            )
                    else:
                        time.sleep(0.01)
                        continue

            except KeyboardInterrupt:
                log.warn("Interrupted by user")
                break
            except (requests.Timeout, requests.ConnectionError) as exc:
                decision = error_policy.decide_on_error(RecoverableIOError(str(exc)))
                log.warn(
                    f"event=error_decision decision={decision.value} type={type(exc).__name__}"
                )
                if decision == ErrorDecision.RETRY:
                    kpi_counters["error_decision_retry"] += 1
                    time.sleep(0.2)
                    continue
                if decision == ErrorDecision.DEGRADE:
                    kpi_counters["error_decision_degrade"] += 1
                    time.sleep(0.2)
                    continue
                kpi_counters["error_decision_stop"] += 1
                break
            except (ValueError, KeyError, TypeError, DataContractError) as exc:
                if isinstance(exc, DataContractError):
                    kpi_counters["state_machine_errors"] += 1
                decision = error_policy.decide_on_error(DataContractError(str(exc)))
                log.error(
                    f"event=error_decision decision={decision.value} type={type(exc).__name__}"
                )
                if decision == ErrorDecision.DEGRADE:
                    kpi_counters["error_decision_degrade"] += 1
                    time.sleep(0.2)
                    continue
                kpi_counters["error_decision_stop"] += 1
                break
            except (RuntimeError, OSError, MemoryError) as exc:
                decision = error_policy.decide_on_error(FatalSystemError(str(exc)))
                log.error(
                    f"event=error_decision decision={decision.value} type={type(exc).__name__}"
                )
                kpi_counters["error_decision_stop"] += 1
                break
            except Exception as exc:
                decision = error_policy.decide_on_error(exc)
                kpi_counters["error_unknown_count"] += 1
                log.error(
                    f"event=error_decision decision={decision.value} type={type(exc).__name__} error={exc}"
                )
                if "pytest" in sys.modules and getattr(sys, "last_type", None) is None:
                    raise
                if decision == ErrorDecision.DEGRADE:
                    kpi_counters["error_decision_degrade"] += 1
                    time.sleep(0.2)
                    continue
                kpi_counters["error_decision_stop"] += 1
                break

    finally:
        resilience_stats = resilience.finalize()
        log.info("Cleaning resources...")
        if Settings.DEBUG and visualizer is not None:
            cv2.destroyAllWindows()
            cv2.waitKey(1)

        _print_summary(
            log,
            fps_counter,
            kpi_counters=kpi_counters,
            resilience_stats={
                "breaker_open_count": resilience_stats.breaker_open_count,
                "degrade_entries": resilience_stats.degrade_entries,
                "degrade_frames": resilience_stats.degrade_frames,
                "recovered_count": resilience_stats.recovered_count,
                "transient_wall_time_sec": resilience_stats.transient_wall_time_sec,
            },
        )
        Settings.SAHI_ENABLED = bool(low_fps_guard_state["orig_sahi"])
        Settings.INFERENCE_SIZE = int(low_fps_guard_state["orig_inference_size"])
        Settings.MAX_DETECTIONS = int(low_fps_guard_state["orig_max_det"])
        Settings.CONFIDENCE_THRESHOLD = float(low_fps_guard_state["orig_conf"])
        Settings.AUGMENTED_INFERENCE = bool(low_fps_guard_state["orig_augmented"])
        Settings.JSON_LOG_EVERY_N_FRAMES = int(low_fps_guard_state["orig_json_interval"])
        Settings.DEGRADE_SEND_INTERVAL_FRAMES = int(
            low_fps_guard_state["orig_degrade_interval"]
        )


def _print_competition_result(
    log: Logger,
    frame_id,
    detected_objects: list,
    send_status: str,
    position: dict,
    gps_health: Optional[int],
) -> None:
    mode = "GPS" if gps_health == 1 else ("OF" if gps_health == 0 else "UNKNOWN")

    def _safe_float(val) -> float:
        try:
            return float(val)
        except (TypeError, ValueError):
            return 0.0

    x = _safe_float(position.get("x", 0.0))
    y = _safe_float(position.get("y", 0.0))
    z = _safe_float(position.get("z", 0.0))

    send_status_text = (
        "OK" if send_status in {"acked", "fallback_acked", "SUCCESS"} else "FAIL"
    )
    log.info(
        f"Frame: {frame_id} | Obj: {len(detected_objects)} | "
        f"Send: {send_status_text} ({send_status}) | Mode: {mode} | "
        f"Pos: x={x:+.1f} y={y:+.1f} z={z:.1f}"
    )


def _print_summary(
    log: Logger,
    fps_counter: FPSCounter,
    kpi_counters: Optional[Dict[str, Any]] = None,
    resilience_stats: Optional[Dict[str, float]] = None,
) -> None:
    log.info("-" * 50)
    log.info(f"Total processed frames: {fps_counter.frame_count}")
    elapsed = time.time() - fps_counter.start_time
    avg_fps = 0.0
    if elapsed > 0:
        avg_fps = fps_counter.frame_count / elapsed
        log.info(f"Average FPS: {avg_fps:.2f}")
    log.info(f"Total elapsed: {elapsed:.1f}s")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        log.info("GPU cache cleared")

    def val_str(val):
        return f"{val:.0f}" if isinstance(val, (float, int)) else str(val)

    if kpi_counters is not None:
        log.info(
            "KPI: "
            f"Send OK={kpi_counters.get('send_ok', 0)} | "
            f"Send FAIL={kpi_counters.get('send_fail', 0)} | "
            f"Fallback ACK={kpi_counters.get('send_fallback_ok', 0)} | "
            f"Permanent Reject={kpi_counters.get('send_permanent_reject', 0)} | "
            f"Preflight Reject={kpi_counters.get('payload_preflight_reject_count', 0)} | "
            f"Payload Clipped={kpi_counters.get('payload_clipped_count', 0)} | "
            f"Mode OF={kpi_counters.get('mode_of', 0)} | "
            f"Mode Unknown={kpi_counters.get('mode_unknown', 0)} | "
            f"Degrade Frames={kpi_counters.get('degrade_frames', 0)} | "
            f"Degrade Replay={kpi_counters.get('degrade_replayed_detection_frames', 0)} | "
            f"Degrade Empty={kpi_counters.get('degrade_empty_payload_count', 0)} | "
            f"DupDrop={kpi_counters.get('frame_duplicate_drop', 0)} | "
            f"DupStorm={kpi_counters.get('duplicate_storm_events', 0)} | "
            f"State={kpi_counters.get('frame_state', 'IDLE')} | "
            f"StateErr={kpi_counters.get('state_machine_errors', 0)} | "
            f"RollingFPS={_safe_float(kpi_counters.get('rolling_fps', 0.0)):.2f} | "
            f"JSONInt={kpi_counters.get('json_log_interval', Settings.JSON_LOG_EVERY_N_FRAMES)} | "
            f"FPSGuard(A/R)="
            f"{kpi_counters.get('fps_guard_activations', 0)}/"
            f"{kpi_counters.get('fps_guard_recoveries', 0)} | "
            f"GPUMaint={kpi_counters.get('gpu_maintenance_runs', 0)} | "
            f"Timeouts(fetch/image/submit)="
            f"{kpi_counters.get('timeout_fetch', 0)}/"
            f"{kpi_counters.get('timeout_image', 0)}/"
            f"{kpi_counters.get('timeout_submit', 0)}"
        )
        send_ok = int(kpi_counters.get("send_ok", 0))
        send_fail = int(kpi_counters.get("send_fail", 0))
        processed = max(1, int(fps_counter.frame_count))
        frame_loss_ratio = max(0.0, 1.0 - (float(send_ok) / float(processed)))
        fallback_ratio = float(kpi_counters.get("send_fallback_ok", 0)) / float(
            max(1, send_ok)
        )
        reject_ratio = float(kpi_counters.get("send_permanent_reject", 0)) / float(
            max(1, send_ok + send_fail)
        )
        log.info(
            "Derived Metrics: "
            f"FrameLossRatio={frame_loss_ratio:.4f} | "
            f"FallbackRatio={fallback_ratio:.4f} | "
            f"SubmitRejectRatio={reject_ratio:.4f}"
        )
        log.info(
            "KPI Compensation: "
            f"Apply Count={kpi_counters.get('compensation_apply_count', 0)} | "
            f"Avg Delta={_safe_float(kpi_counters.get('compensation_avg_delta_m', 0.0)):.3f}m | "
            f"Max Delta={_safe_float(kpi_counters.get('compensation_max_delta_m', 0.0)):.3f}m"
        )
        log.info(
            f"Payload Size   : Max {val_str(kpi_counters.get('max_payload_bytes'))} bytes "
            f"(Avg: {val_str(kpi_counters.get('avg_payload_bytes'))} bytes)"
        )
        ref_stats = kpi_counters.get("reference_validation_stats", {})
        if isinstance(ref_stats, dict):
            log.info(
                "Task3 Ref Integrity: "
                f"Mode={kpi_counters.get('id_integrity_mode', 'unknown')} | "
                f"Reason={kpi_counters.get('id_integrity_reason_code', 'unknown')} | "
                f"Stats(total/valid/duplicate/quarantined)="
                f"{int(ref_stats.get('total', 0))}/"
                f"{int(ref_stats.get('valid', 0))}/"
                f"{int(ref_stats.get('duplicate', 0))}/"
                f"{int(ref_stats.get('quarantined', 0))}"
            )

    if resilience_stats is not None:
        log.info(
            "Resilience: "
            f"Breaker Open Count={int(resilience_stats.get('breaker_open_count', 0))} | "
            f"Degrade Entries={int(resilience_stats.get('degrade_entries', 0))} | "
            f"Recovery Count={int(resilience_stats.get('recovered_count', 0))} | "
            f"Transient Wall Time={float(resilience_stats.get('transient_wall_time_sec', 0.0)):.1f}s"
        )

    report_payload: Dict[str, Any] = {
        "processed_frames": int(fps_counter.frame_count),
        "elapsed_sec": float(round(elapsed, 4)),
        "average_fps": float(round(avg_fps, 4)) if elapsed > 0 else 0.0,
        "kpi_counters": dict(kpi_counters or {}),
        "resilience_stats": dict(resilience_stats or {}),
        "baseline_metrics": {
            "frame_loss_ratio": max(
                0.0,
                1.0
                - (
                    float((kpi_counters or {}).get("send_ok", 0))
                    / float(max(1, int(fps_counter.frame_count)))
                ),
            ),
            "submit_reject_ratio": float(
                (kpi_counters or {}).get("send_permanent_reject", 0)
            )
            / float(
                max(
                    1,
                    int((kpi_counters or {}).get("send_ok", 0))
                    + int((kpi_counters or {}).get("send_fail", 0)),
                )
            ),
            "fallback_ratio": float((kpi_counters or {}).get("send_fallback_ok", 0))
            / float(max(1, int((kpi_counters or {}).get("send_ok", 0)))),
            "average_fps": float(round(avg_fps, 4)) if elapsed > 0 else 0.0,
        },
    }
    try:
        log_json_to_disk(report_payload, direction="metrics", tag="run_summary")
    except Exception as exc:
        log.warn(f"Metrics summary write skipped: {exc}")

    log.success("System shutdown complete")


def _ask_choice(prompt: str, options: dict, default: Optional[str] = None) -> str:
    print()
    if default and default in options:
        print(f"{prompt} (Default: [{default}] {options[default]})")
    else:
        print(prompt)
    for key, desc in options.items():
        print(f"  [{key}] {desc}")
    print()

    while True:
        choice = input("  Selection: ").strip()
        if not choice and default in options:
            return default
        if choice in options:
            return choice
        print(f"  Invalid selection, choose one of: {', '.join(options.keys())}")


def show_interactive_menu() -> dict:
    from src.data_loader import get_available_sequences

    print("\n" + "=" * 56)
    print("  RUN MODE SELECTION")
    print("=" * 56)

    mode = _ask_choice(
        "  Choose run mode:",
        {
            "1": "Competition (server)",
            "2": "Simulation VID (sequential frames / video)",
            "3": "Simulation DET (single images)",
        },
        default="2"
    )

    if mode == "1":
        return {"mode": "competition", "prefer_vid": True, "show": False, "save": False}

    prefer_vid = mode == "2"
    sequence = None

    if prefer_vid:
        seqs = get_available_sequences()
        if seqs:
            seq_options = {"0": "Autoselect largest/first"}
            seq_keys = list(seqs.keys())
            for i, k in enumerate(seq_keys, 1):
                info = seqs[k]
                desc = f"{k} ({info['type']}, {info['count']} items)"
                seq_options[str(i)] = desc
            
            seq_choice = _ask_choice("  Available sequences/videos:", seq_options, default="0")
            if seq_choice != "0":
                sequence = seq_keys[int(seq_choice) - 1]

    output = _ask_choice(
        "  How do you want outputs?",
        {
            "1": "Terminal only",
            "2": "Show window",
            "3": "Save images",
            "4": "Show window + Save images",
        },
        default="2"
    )

    show = output in ("2", "4")
    save = output in ("3", "4")

    return {"mode": "simulate", "prefer_vid": prefer_vid, "show": show, "save": save, "sequence": sequence}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TEKNOFEST AIA Runtime")
    parser.add_argument(
        "--mode",
        choices=[
            Settings.COMPETITION_RUNTIME_MODE,
            Settings.VISUAL_VALIDATION_RUNTIME_MODE,
            "simulate_vid",
            "simulate_det",
        ],
        default=Settings.DEFAULT_RUNTIME_MODE,
        help=(
            "Run mode. competition=competition workflow, "
            "visual_validation=human-checkable validation mode (default). "
            "simulate_vid/simulate_det are backward-compatible aliases."
        ),
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enable interactive menu flow",
    )
    parser.add_argument(
        "--deterministic-profile",
        choices=["off", "balanced", "max"],
        default="balanced",
        help="Runtime determinism profile",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Competition server base URL override (or use AIA_BASE_URL env var)",
    )
    parser.add_argument(
        "--team-name",
        type=str,
        default=None,
        help="Team name override for outbound payloads (or use AIA_TEAM_NAME env var)",
    )
    parser.add_argument("--show", action="store_true", help="Show simulation window")
    parser.add_argument("--save", action="store_true", help="Save simulation images")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Deterministik simülasyon için rastgele seed",
    )
    parser.add_argument(
        "--sequence",
        type=str,
        default=None,
        help="VID modunda seçilecek sekans adı (örn. uav0000123)",
    )
    return parser.parse_args()


def apply_runtime_overrides(args: argparse.Namespace, log: Logger) -> None:
    base_url_cli = (args.base_url or "").strip()
    base_url_env = (os.getenv("AIA_BASE_URL", "") or "").strip()
    team_name_cli = (args.team_name or "").strip()
    team_name_env = (os.getenv("AIA_TEAM_NAME", "") or "").strip()

    base_url_override = base_url_cli or base_url_env
    team_name_override = team_name_cli or team_name_env

    if base_url_override:
        Settings.BASE_URL = base_url_override.rstrip("/")
        src = "CLI --base-url" if base_url_cli else "ENV AIA_BASE_URL"
        log.info(f"Runtime override: BASE_URL <- {Settings.BASE_URL} ({src})")

    if team_name_override:
        Settings.TEAM_NAME = team_name_override
        src = "CLI --team-name" if team_name_cli else "ENV AIA_TEAM_NAME"
        log.info(f"Runtime override: TEAM_NAME <- {Settings.TEAM_NAME} ({src})")


def main() -> None:
    log = Logger("Main")
    args = parse_args()
    apply_runtime_overrides(args, log)

    requested_profile = args.deterministic_profile
    effective_profile = requested_profile
    if (
        not args.interactive
        and args.mode == Settings.COMPETITION_RUNTIME_MODE
        and requested_profile != "max"
    ):
        log.warn(
            "Competition mode requires deterministic-profile=max; "
            f"overriding requested profile '{requested_profile}' -> 'max'"
        )
        effective_profile = "max"

    apply_runtime_profile(effective_profile, requested_profile=requested_profile)

    print(BANNER)

    if len(sys.argv) == 1 or args.interactive:
        choices = show_interactive_menu()
    else:
        if args.mode == Settings.COMPETITION_RUNTIME_MODE:
            choices = {
                "mode": "competition",
                "prefer_vid": True,
                "show": False,
                "save": False,
            }
        elif args.mode in {Settings.VISUAL_VALIDATION_RUNTIME_MODE, "simulate_vid"}:
            choices = {
                "mode": "simulate",
                "prefer_vid": True,
                "show": (
                    True
                    if args.mode == Settings.VISUAL_VALIDATION_RUNTIME_MODE
                    and not args.save
                    else args.show
                ),
                "save": args.save,
            }
        else:
            choices = {
                "mode": "simulate",
                "prefer_vid": False,
                "show": args.show,
                "save": args.save,
            }

    simulate = choices["mode"] == "simulate"
    if not simulate and effective_profile != "max":
        log.warn(
            "Competition mode requires deterministic-profile=max; "
            f"overriding requested profile '{requested_profile}' -> 'max'"
        )
        effective_profile = "max"
        apply_runtime_profile(effective_profile, requested_profile=requested_profile)

    runtime_mode_label = "VISUAL_VALIDATION" if simulate else "COMPETITION"
    print_system_info(log, runtime_mode_label=runtime_mode_label)

    if simulate:
        run_simulation(
            log,
            prefer_vid=choices.get("prefer_vid", True),
            show=choices.get("show", False),
            save=choices.get("save", False),
            seed=args.seed,
            sequence=choices.get("sequence", args.sequence),
        )
    else:
        run_competition(log)


def _fetch_competition_step(
    log: Logger,
    network: Any,
    detector: Any,
    movement: Any,
    odometry: Any,
    image_matcher: Any,
    resilience: Any,
    kpi_counters: dict,
    transient_failures: int,
    transient_budget: int,
    degrade_replay_state: Dict[str, Any],
    degrade_fallback_window: deque,
):
    from src.network import FrameFetchStatus
    import time

    if not resilience.before_fetch():
        cooldown_left = resilience.open_cooldown_remaining()
        wait_s = min(max(0.2, Settings.RETRY_DELAY), max(0.2, cooldown_left))
        log.warn(
            f"Circuit breaker OPEN; waiting cooldown ({cooldown_left:.1f}s remaining, sleep={wait_s:.1f}s)"
        )
        return None, transient_failures, "continue", False

    fetch_result = network.get_frame()
    timeout_snapshot = network.consume_timeout_counters()
    kpi_counters["timeout_fetch"] += timeout_snapshot.get("fetch", 0)
    kpi_counters["timeout_image"] += timeout_snapshot.get("image", 0)
    kpi_counters["timeout_submit"] += timeout_snapshot.get("submit", 0)

    if fetch_result.status == FrameFetchStatus.END_OF_STREAM:
        log.info("End of stream confirmed by server (204)")
        return None, transient_failures, "break", False

    if fetch_result.status == FrameFetchStatus.TRANSIENT_ERROR:
        transient_failures += 1
        resilience.on_fetch_transient()
        delay = min(5.0, Settings.RETRY_DELAY * (2 ** min(transient_failures, 4)))
        log.warn(
            f"Transient frame fetch failure {transient_failures}/{transient_budget}; retrying in {delay:.1f}s"
        )
        if transient_failures >= transient_budget:
            log.warn(
                "Transient failure budget reached; session stays alive under wall-clock circuit breaker policy"
            )
        time.sleep(delay)
        return None, transient_failures, "continue", False

    if fetch_result.status == FrameFetchStatus.FATAL_ERROR:
        log.error(
            f"Fatal frame fetch error: {fetch_result.error_type} (http={fetch_result.http_status})"
        )
        return None, transient_failures, "break", False

    frame_data = fetch_result.frame_data or {}
    transient_failures = 0
    degrade_mode = Settings.DEGRADE_FETCH_ONLY_ENABLED and resilience.is_degraded()
    frame_id = frame_data.get("frame_id", "unknown")
    frame_fetch_monotonic = time.monotonic()

    if degrade_replay_state.get("objects"):
        degrade_replay_state["age"] = int(degrade_replay_state.get("age", 0)) + 1

    if fetch_result.is_duplicate:
        kpi_counters["frame_duplicate_drop"] += 1
        log.warn(
            f"Frame {frame_id}: duplicate metadata detected. Processing normally per specification idempotency."
        )

    frame = None
    use_fallback = False
    fallback_reason_code = "frame_download_failed"

    degrade_seq = 0
    heavy_every = max(1, int(Settings.DEGRADE_SEND_INTERVAL_FRAMES))
    if degrade_mode:
        kpi_counters["degrade_frames"] += 1
        degrade_seq = resilience.record_degraded_frame()

    fallback_ratio = 0.0
    if len(degrade_fallback_window) > 0:
        fallback_ratio = float(sum(degrade_fallback_window)) / float(
            len(degrade_fallback_window)
        )
    force_recovery_heavy = (
        degrade_mode
        and len(degrade_fallback_window) >= max(5, degrade_fallback_window.maxlen // 2)
        and fallback_ratio >= float(getattr(Settings, "DEGRADE_FALLBACK_RATIO_HIGH", 0.75))
    )
    if force_recovery_heavy:
        log.warn(
            f"event=degrade_recovery_force_heavy frame_id={frame_id} "
            f"fallback_ratio={fallback_ratio:.2f}"
        )

    fetch_decision = decide_degrade_fetch_strategy(
        is_degraded=degrade_mode,
        degrade_seq=degrade_seq,
        heavy_every=heavy_every,
        force_full_frame=force_recovery_heavy,
    )
    if degrade_mode:
        log.info(
            "event=degrade_fetch_decision "
            f"frame_id={frame_id} strategy={fetch_decision.strategy.value} "
            f"reason_code={fetch_decision.reason_code} slot={fetch_decision.degrade_seq}/{heavy_every}"
        )

    if fetch_decision.strategy == FetchStrategy.FULL_FRAME:
        frame = network.download_image(frame_data)
        timeout_snapshot = network.consume_timeout_counters()
        kpi_counters["timeout_fetch"] += timeout_snapshot.get("fetch", 0)
        kpi_counters["timeout_image"] += timeout_snapshot.get("image", 0)
        kpi_counters["timeout_submit"] += timeout_snapshot.get("submit", 0)
        if frame is None:
            use_fallback = True
            fallback_reason_code = "frame_download_failed"
            if degrade_mode:
                log.warn(
                    f"Frame {frame_id}: degrade heavy pass image download failed, sending fallback result"
                )
            else:
                log.warn(
                    f"Frame {frame_id}: image download failed, sending fallback result"
                )
        elif degrade_mode:
            log.info(
                f"Frame {frame_id}: degrade heavy pass (every {heavy_every} frames)"
            )
    else:
        if degrade_mode:
            use_fallback = True
            fallback_reason_code = fetch_decision.reason_code
            log.info(
                f"Frame {frame_id}: degraded fetch-only fallback (slot {degrade_seq}/{heavy_every})"
            )
        else:
            frame = network.download_image(frame_data)
            timeout_snapshot = network.consume_timeout_counters()
            kpi_counters["timeout_fetch"] += timeout_snapshot.get("fetch", 0)
            kpi_counters["timeout_image"] += timeout_snapshot.get("image", 0)
            kpi_counters["timeout_submit"] += timeout_snapshot.get("submit", 0)
            if frame is None:
                use_fallback = True
                log.warn(
                    f"Frame {frame_id}: image download failed, sending fallback result"
                )

    if use_fallback:
        gps_health = _safe_gps_health(frame_data)
        if gps_health == 0:
            if hasattr(odometry, "predict_without_measurement"):
                last_position = odometry.predict_without_measurement(
                    reason_code=fallback_reason_code,
                    gps_health=0,
                )
            else:
                last_position = odometry.get_last_of_position()
        else:
            last_position = odometry.get_position()
        runtime_meta = (
            odometry.get_runtime_meta()
            if hasattr(odometry, "get_runtime_meta")
            else {
                "update_mode": "fallback",
                "state_source": "last_known",
                "quality_flag": "degraded" if gps_health != 1 else "nominal",
                "reason_code": fallback_reason_code,
            }
        )
        replay_objects: List[Dict[str, Any]] = []
        replay_used = False
        if bool(getattr(Settings, "DEGRADE_REPLAY_ENABLED", True)):
            max_age = max(0, int(getattr(Settings, "DEGRADE_REPLAY_MAX_AGE_FRAMES", 6)))
            max_objects = max(0, int(getattr(Settings, "DEGRADE_REPLAY_MAX_OBJECTS", 40)))
            cached_objects = degrade_replay_state.get("objects") or []
            cache_age = int(degrade_replay_state.get("age", 10**9))
            if cached_objects and cache_age <= max_age:
                replay_objects = [dict(obj) for obj in cached_objects[:max_objects]]
                replay_used = True
                kpi_counters["degrade_replayed_detection_frames"] = int(
                    kpi_counters.get("degrade_replayed_detection_frames", 0)
                ) + 1
                log.info(
                    f"event=degrade_detection_replay frame_id={frame_id} "
                    f"object_count={len(replay_objects)} cache_age={cache_age}"
                )
            else:
                kpi_counters["degrade_empty_payload_count"] = int(
                    kpi_counters.get("degrade_empty_payload_count", 0)
                ) + 1
        else:
            kpi_counters["degrade_empty_payload_count"] = int(
                kpi_counters.get("degrade_empty_payload_count", 0)
            ) + 1

        pending_result = {
            "frame_id": frame_id,
            "frame_data": frame_data,
            "detected_objects": replay_objects,
            "frame": None,
            "position": last_position,
            "degraded": degrade_mode,
            "guardrail_stats": {},
            "detected_translation": {
                "translation_x": last_position["x"],
                "translation_y": last_position["y"],
                "translation_z": last_position["z"],
            },
            "localization_runtime": runtime_meta,
            "base_position": dict(last_position),
            "frame_fetch_monotonic": frame_fetch_monotonic,
            "frame_shape": None,
            "detected_undefined_objects": [],
            "is_duplicate": fetch_result.is_duplicate,
            "used_detection_replay": replay_used,
        }
    else:
        frame_ctx = FrameContext(frame)
        detect_profile = "light" if degrade_mode else "default"
        try:
            detected_objects = detector.detect(frame, runtime_profile=detect_profile)
        except TypeError:
            detected_objects = detector.detect(frame)
        detected_objects = movement.annotate(detected_objects, frame_ctx=frame_ctx)
        if detected_objects:
            max_objects = max(1, int(getattr(Settings, "DEGRADE_REPLAY_MAX_OBJECTS", 40)))
            degrade_replay_state["objects"] = [
                dict(obj) for obj in detected_objects[:max_objects]
            ]
            degrade_replay_state["age"] = 0
        undefined_objects = []
        if image_matcher is not None:
            undefined_objects = image_matcher.match(frame)
        position = odometry.update(frame_ctx, frame_data)
        runtime_meta = (
            odometry.get_runtime_meta()
            if hasattr(odometry, "get_runtime_meta")
            else {
                "update_mode": "corrected",
                "state_source": "unknown",
                "quality_flag": "nominal",
                "reason_code": "legacy_odometry",
            }
        )
        guardrail_stats = getattr(detector, "_last_guardrail_stats", None) or {}
        pending_result = {
            "frame_id": frame_id,
            "frame_data": frame_data,
            "detected_objects": detected_objects,
            "frame": frame,
            "position": position,
            "guardrail_stats": guardrail_stats,
            "degraded": degrade_mode,
            "detected_translation": {
                "translation_x": position["x"],
                "translation_y": position["y"],
                "translation_z": position["z"],
            },
            "localization_runtime": runtime_meta,
            "base_position": dict(position),
            "frame_fetch_monotonic": frame_fetch_monotonic,
            "frame_shape": frame.shape,
            "detected_undefined_objects": undefined_objects,
            "is_duplicate": fetch_result.is_duplicate,
        }

    if degrade_mode:
        degrade_fallback_window.append(1 if use_fallback else 0)
        if len(degrade_fallback_window) > 0:
            kpi_counters["degrade_fallback_ratio"] = round(
                float(sum(degrade_fallback_window))
                / float(len(degrade_fallback_window)),
                4,
            )

    return pending_result, transient_failures, "process", fetch_result.is_duplicate


def _submit_competition_step(
    log: Logger,
    network: Any,
    resilience: Any,
    odometry: VisualOdometry,
    kpi_counters: dict,
    pending_result: dict,
    ack_failures: int,
    ack_failure_budget: int,
    consecutive_permanent_rejects: int,
    permanent_reject_abort_threshold: int,
):
    import time

    required_keys = {
        "frame_id",
        "frame_data",
        "detected_objects",
        "detected_translation",
        "frame_shape",
    }
    missing = required_keys - set(pending_result.keys())
    if missing:
        raise DataContractError(f"pending_result missing keys: {sorted(list(missing))}")

    frame_id = pending_result["frame_id"]
    frame_data = pending_result["frame_data"]
    detected_objects = pending_result["detected_objects"]
    detected_translation, position_for_success = _apply_latency_compensation_if_needed(
        log=log,
        odometry=odometry,
        pending_result=pending_result,
        kpi_counters=kpi_counters,
    )

    send_status = network.send_result(
        frame_id,
        detected_objects,
        detected_translation,
        frame_data=frame_data,
        frame_shape=pending_result["frame_shape"],
        degrade=bool(pending_result.get("degraded", False)),
        detected_undefined_objects=pending_result.get("detected_undefined_objects"),
    )

    timeout_snapshot = network.consume_timeout_counters()
    kpi_counters["timeout_fetch"] += timeout_snapshot.get("fetch", 0)
    kpi_counters["timeout_image"] += timeout_snapshot.get("image", 0)
    kpi_counters["timeout_submit"] += timeout_snapshot.get("submit", 0)

    guard_snapshot = network.consume_payload_guard_counters()
    kpi_counters["payload_preflight_reject_count"] += guard_snapshot.get(
        "preflight_reject", 0
    )
    kpi_counters["payload_clipped_count"] += guard_snapshot.get("payload_clipped", 0)

    pending_result_snapshot = dict(pending_result)

    pending_result, should_abort_session, success_cycle = apply_send_result_status(
        send_status=send_status,
        pending_result=pending_result,
        kpi_counters=kpi_counters,
    )

    if pending_result is None and not success_cycle:
        status_value = str(getattr(send_status, "value", send_status))
        if status_value == "permanent_rejected":
            retry_limit = max(0, int(getattr(Settings, "PERMANENT_REJECT_RETRY_LIMIT", 1)))
            retry_count = int(pending_result_snapshot.get("permanent_reject_retry_count", 0))
            if retry_count < retry_limit:
                pending_result_snapshot["permanent_reject_retry_count"] = retry_count + 1
                pending_result_snapshot["degraded"] = True
                log.warn(
                    f"Frame {frame_id}: permanent rejected; controlled retry "
                    f"({retry_count + 1}/{retry_limit}) with safe payload path"
                )
                time.sleep(min(2.0, Settings.RETRY_DELAY))
                return (
                    pending_result_snapshot,
                    ack_failures,
                    "continue",
                    consecutive_permanent_rejects,
                )

            consecutive_permanent_rejects += 1
            pending_result_snapshot["degraded"] = True
            pending_result_snapshot["permanent_reject_exhausted_count"] = (
                int(pending_result_snapshot.get("permanent_reject_exhausted_count", 0))
                + 1
            )
            log.error(
                "event=permanent_reject_terminal_decision "
                f"frame_id={frame_id} streak={consecutive_permanent_rejects} "
                f"threshold={permanent_reject_abort_threshold}"
            )
            if consecutive_permanent_rejects >= permanent_reject_abort_threshold:
                log.error(
                    "Consecutive permanent reject threshold reached, aborting session"
                )
                return None, ack_failures, "break", consecutive_permanent_rejects

            ack_failures += 1
            resilience.on_ack_failure()
            delay = min(5.0, Settings.RETRY_DELAY * (2 ** min(ack_failures, 4)))
            time.sleep(delay)
            return (
                pending_result_snapshot,
                ack_failures,
                "continue",
                consecutive_permanent_rejects,
            )

        # Failsafe: ACK alınamayan döngüde frame'i düşürme; aynı frame ile devam et.
        ack_failures += 1
        resilience.on_ack_failure()
        time.sleep(min(2.0, Settings.RETRY_DELAY))
        return pending_result_snapshot, ack_failures, "continue", consecutive_permanent_rejects

    if success_cycle:
        ack_failures = 0
        consecutive_permanent_rejects = 0
        resilience.on_success_cycle()
        pending_result_snapshot["position"] = position_for_success
        success_info = {
            "frame_for_debug": pending_result_snapshot.get("frame"),
            "detected_objects": detected_objects,
            "position": pending_result_snapshot.get("position"),
            "guardrail_stats": pending_result_snapshot.get("guardrail_stats", {}),
            "frame_id": frame_id,
            "frame_data": frame_data,
            "frame_fetch_monotonic": pending_result_snapshot.get("frame_fetch_monotonic"),
            "degraded": bool(pending_result_snapshot.get("degraded", False)),
        }
        return (
            pending_result,
            ack_failures,
            ("process", success_info),
            consecutive_permanent_rejects,
        )
    else:
        ack_failures += 1
        resilience.on_ack_failure()
        delay = min(5.0, Settings.RETRY_DELAY * (2 ** min(ack_failures, 4)))
        log.warn(
            f"Frame {frame_id}: result send failed ({send_status}), "
            f"waiting ACK ({ack_failures}/{ack_failure_budget}); retrying in {delay:.1f}s"
        )
        if ack_failures >= ack_failure_budget:
            log.warn(
                "ACK failure budget reached; session stays alive "
                "under wall-clock circuit breaker policy"
            )

        time.sleep(delay)
        return pending_result, ack_failures, "continue", consecutive_permanent_rejects


if __name__ == "__main__":
    main()
