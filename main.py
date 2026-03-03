"""TEKNOFEST Havacılıkta Yapay Zeka — Ana orkestrasyon.
Simülasyon: datasets/ içinden kare yükler. Yarışma: sunucudan frame alır, sonuç gönderir."""

import argparse
import base64
import os
import signal
import sys
import time
from collections import Counter
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
from src.utils import Logger, Visualizer  # noqa: E402
from src.frame_context import FrameContext  # noqa: E402

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║     🛩️  TEKNOFEST 2026 - HAVACILIKTA YAPAY ZEKA YARIŞMASI    ║
║     ──────────────────────────────────────────────────────    ║
║     Nesne Tespiti (Görev 1) + Konum Kestirimi (Görev 2)     ║
║     Görüntü Eşleme (Görev 3)                                ║
╚══════════════════════════════════════════════════════════════╝
"""


def print_system_info(log: Logger, simulate: bool = False) -> None:
    print(BANNER)
    log.info(f"Working Directory : {PROJECT_ROOT}")
    log.info(f"Mode             : {'SIMULATION' if simulate else 'COMPETITION'}")
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
                log.warn("Görev 3: Referans obje bulunamadı, Simülasyonda Görev 3 pasif")
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
) -> bool:
    frame = frame_info["frame"]
    frame_idx = frame_info["frame_idx"]
    server_data = frame_info["server_data"]
    gps_health = frame_info["gps_health"]

    frame_ctx = FrameContext(frame)
    detected_objects = detector.detect(frame)
    detected_objects = movement.annotate(detected_objects, frame_ctx=frame_ctx)
    position = odometry.update(frame_ctx, server_data)

    if image_matcher is not None:
        _ = image_matcher.match(frame)

    _print_simulation_result(log, frame_idx, detected_objects, position, gps_health)

    if show or save:
        annotated = visualizer.draw_detections(
            frame,
            detected_objects,
            frame_id=str(frame_idx),
            position=position,
            save_to_disk=not save,
        )

        mode_text = "GPS" if gps_health == 1 else "Optical Flow"
        cv2.putText(
            annotated,
            f"Mode: {mode_text}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0) if gps_health else (0, 165, 255),
            2,
        )

        if show:
            try:
                cv2.imshow("TEKNOFEST - Simulation", annotated)
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
    gps_health: int,
) -> None:
    cls_counts = Counter(obj["cls"] for obj in detected_objects)
    tasit = cls_counts.get("0", 0)
    insan = cls_counts.get("1", 0)
    uap = cls_counts.get("2", 0)
    uai = cls_counts.get("3", 0)

    loc_mode = "GPS" if gps_health == 1 else "OF"
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


def _safe_gps_health(frame_data: Dict[str, Any]) -> int:
    try:
        return int(float(frame_data.get("gps_health", 0)))
    except (TypeError, ValueError):
        return 0


def _mark_runtime_meta_input_quality(
    runtime_meta: Optional[Dict[str, Any]],
    frame_data: Dict[str, Any],
) -> Dict[str, Any]:
    meta = dict(runtime_meta or {})
    if frame_data.get("gps_health_fallback_used"):
        meta["quality_flag"] = "degraded_input"
        meta["reason_code"] = str(frame_data.get("gps_health_source", "fallback_ttl"))
        meta["health_source"] = str(frame_data.get("gps_health_source", "unknown"))
    return meta


def _smooth_mode_transition(
    raw_gps_health: int,
    mode_state: Dict[str, Any],
    stability_frames: int = 2,
) -> int:
    stable = mode_state.get("stable")
    candidate = mode_state.get("candidate")
    candidate_count = int(mode_state.get("candidate_count", 0))

    if stable is None:
        mode_state["stable"] = int(raw_gps_health)
        mode_state["candidate"] = None
        mode_state["candidate_count"] = 0
        return int(raw_gps_health)

    raw = int(raw_gps_health)
    if raw == int(stable):
        mode_state["candidate"] = None
        mode_state["candidate_count"] = 0
        return int(stable)

    if candidate == raw:
        candidate_count += 1
    else:
        candidate = raw
        candidate_count = 1

    mode_state["candidate"] = candidate
    mode_state["candidate_count"] = candidate_count

    if candidate_count >= max(1, int(stability_frames)):
        mode_state["stable"] = raw
        mode_state["candidate"] = None
        mode_state["candidate_count"] = 0

    return int(mode_state.get("stable", raw))


def _normalize_task3_object_id(raw_object_id: Any) -> Optional[int]:
    if isinstance(raw_object_id, bool) or raw_object_id is None:
        return None

    if isinstance(raw_object_id, int):
        object_id = raw_object_id
    elif isinstance(raw_object_id, float):
        if not raw_object_id.is_integer():
            return None
        object_id = int(raw_object_id)
    elif isinstance(raw_object_id, str):
        stripped = raw_object_id.strip()
        if not stripped:
            return None
        try:
            object_id = int(stripped)
        except ValueError:
            return None
    else:
        return None

    if object_id < 0:
        return None
    return object_id


def _build_task3_reference_source(
    ref_data: Dict[str, Any],
    object_id: int,
) -> Tuple[Optional[Dict[str, Any]], str]:
    label = str(ref_data.get("label", f"ref_{object_id}"))

    if ref_data.get("path"):
        path = str(ref_data["path"])
        if not os.path.isabs(path):
            path = os.path.join(PROJECT_ROOT, path)
        return {"object_id": object_id, "path": path, "label": label}, "path"

    image_base64 = ref_data.get("image_base64")
    if image_base64:
        try:
            arr = np.frombuffer(base64.b64decode(image_base64), dtype=np.uint8)
            image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if image is None:
                return None, "base64_decode_failed"
            return {"object_id": object_id, "image": image, "label": label}, "image_base64"
        except (ValueError, TypeError):
            return None, "base64_decode_failed"

    image = ref_data.get("image")
    if image is not None:
        return {"object_id": object_id, "image": image, "label": label}, "image"

    return None, "missing_image_source"


def _validate_task3_references(
    log: Logger,
    server_refs: List[Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, int], str, str, bool]:
    stats: Dict[str, int] = {
        "total": len(server_refs),
        "valid": 0,
        "duplicate": 0,
        "quarantined": 0,
    }
    canonical_refs: List[Dict[str, Any]] = []
    seen_object_ids: Dict[int, Dict[str, Any]] = {}

    for idx, raw_ref in enumerate(server_refs):
        if not isinstance(raw_ref, dict):
            stats["quarantined"] += 1
            log.warn(
                f"event=task3_ref_quarantined reason=invalid_record_type index={idx}"
            )
            continue

        object_id = _normalize_task3_object_id(raw_ref.get("object_id"))
        if object_id is None:
            stats["quarantined"] += 1
            log.warn(
                f"event=task3_ref_quarantined reason=invalid_object_id index={idx} raw_object_id={raw_ref.get('object_id')}"
            )
            continue

        source_ref, source_kind = _build_task3_reference_source(raw_ref, object_id)
        if source_ref is None:
            stats["quarantined"] += 1
            log.warn(
                f"event=task3_ref_quarantined reason={source_kind} object_id={object_id} index={idx}"
            )
            continue

        if object_id in seen_object_ids:
            stats["duplicate"] += 1
            stats["quarantined"] += 1
            first_source = seen_object_ids[object_id].get("_source_kind", "unknown")
            log.warn(
                f"event=task3_ref_duplicate_detected object_id={object_id} first_source={first_source} duplicate_source={source_kind} index={idx}"
            )
            log.warn(
                f"event=task3_ref_quarantined reason=duplicate_object_id object_id={object_id} index={idx}"
            )
            continue

        source_ref["_source_kind"] = source_kind
        seen_object_ids[object_id] = source_ref
        canonical_refs.append(source_ref)
        stats["valid"] += 1

    for ref in canonical_refs:
        ref.pop("_source_kind", None)

    duplicate_ratio = (
        float(stats["duplicate"]) / float(stats["total"])
        if stats["total"] > 0
        else 0.0
    )
    duplicate_critical = (
        stats["duplicate"] >= int(Settings.TASK3_DUPLICATE_DEGRADE_MIN_COUNT)
        and duplicate_ratio >= float(Settings.TASK3_DUPLICATE_DEGRADE_RATIO)
    )

    if duplicate_critical:
        id_integrity_mode = "degraded"
        id_integrity_reason_code = "duplicate_ratio_critical"
    elif stats["duplicate"] > 0:
        id_integrity_mode = "degraded"
        id_integrity_reason_code = "duplicate_detected_safe_degrade"
    elif stats["quarantined"] > 0:
        id_integrity_mode = "degraded"
        id_integrity_reason_code = "reference_quarantined_non_duplicate"
    else:
        id_integrity_mode = "normal"
        id_integrity_reason_code = "ok"

    log.info(
        f"event=task3_ref_validation_summary total={stats['total']} valid={stats['valid']} duplicate={stats['duplicate']} quarantined={stats['quarantined']}"
    )
    log.warn(
        f"event=task3_id_integrity_mode mode={id_integrity_mode} reason_code={id_integrity_reason_code} duplicate_ratio={duplicate_ratio:.3f}"
    )

    return (
        canonical_refs,
        stats,
        id_integrity_mode,
        id_integrity_reason_code,
        duplicate_critical,
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
    projected_position, applied_delta_m, used_dt_sec = odometry.project_position_with_latency(
        position=projection_base,
        dt_sec=dt_sec,
        max_dt_sec=max_dt_sec,
        max_delta_m=max_delta_m,
    )

    if applied_delta_m <= 0.0:
        return base_translation, base_position

    kpi_counters["compensation_apply_count"] = int(kpi_counters.get("compensation_apply_count", 0)) + 1
    kpi_counters["compensation_sum_delta_m"] = _safe_float(
        kpi_counters.get("compensation_sum_delta_m", 0.0)
    ) + applied_delta_m
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
                log.warn("Görev 3: Referans obje bulunamadı, detected_undefined_objects boş gönderilecek")

        visualizer: Optional[Visualizer] = Visualizer() if Settings.DEBUG else None

        log.success("All modules initialized successfully")

    except (ImportError, RuntimeError, OSError, ValueError) as exc:
        log.error(f"Initialization error: {exc}")
        log.error("System startup failed, exiting.")
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
        if duplicate_critical:
            image_matcher = None
            log.warn(
                "Görev 3: ID integrity kritik eşiği aşıldı, kontrollü pasif moda geçildi "
                "(task3_disable_reason=duplicate_ratio_critical)"
            )
        elif canonical_refs:
            loaded = image_matcher.load_references(canonical_refs)
            if loaded == 0:
                log.warn("Görev 3: Doğrulanan referanslar yüklense de matcher aktifleşmedi")
            else:
                log.info(
                    f"Görev 3: Sunucudan {loaded} canonical referans obje yüklendi"
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
    CONSECUTIVE_DUPLICATE_ABORT_THRESHOLD = 5
    resilience = SessionResilienceController(log=log)
    error_policy = ErrorPolicy(
        retry_budget=max(5, Settings.MAX_RETRIES * 2),
        degrade_budget=3,
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
        "degrade_frames": 0,
        "frame_duplicate_drop": 0,
        "timeout_fetch": 0,
        "timeout_image": 0,
        "timeout_submit": 0,
        "forced_health_fallback_count": 0,
        "consecutive_duplicate_abort": 0,
        "compensation_apply_count": 0,
        "compensation_sum_delta_m": 0.0,
        "compensation_avg_delta_m": 0.0,
        "compensation_max_delta_m": 0.0,
        "error_decision_retry": 0,
        "error_decision_degrade": 0,
        "error_decision_stop": 0,
        "error_unknown_count": 0,
        "reference_validation_stats": reference_validation_stats,
        "id_integrity_mode": id_integrity_mode,
        "id_integrity_reason_code": id_integrity_reason_code,
    }
    pending_result: Optional[Dict] = None
    mode_transition_state: Dict[str, Any] = {}

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
                abort_reason = resilience.should_abort()
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
                    submit_future = executor.submit(
                        _submit_competition_step,
                        log, network, resilience, odometry, kpi_counters, pending_result,
                        ack_failures, ack_failure_budget,
                        consecutive_permanent_rejects, PERMANENT_REJECT_ABORT_THRESHOLD,
                    )

                if submit_future is not None and submit_future.done():
                    result = submit_future.result()
                    pending_result = result[0]
                    ack_failures = result[1]
                    action_result = result[2]
                    consecutive_permanent_rejects = result[3]
                    submit_future = None
                    if action_result == "break":
                        break
                    elif action_result == "continue":
                        continue

                    if not (isinstance(action_result, tuple) and len(action_result) == 2):
                        log.error(
                            f"Unexpected action_result type: {type(action_result)}, skipping frame"
                        )
                        continue
                    _, success_info = action_result

                    gps_health = _safe_gps_health(success_info["frame_data"])
                    gps_health = _smooth_mode_transition(
                        gps_health,
                        mode_transition_state,
                        stability_frames=2,
                    )

                    if gps_health == 1:
                        kpi_counters["mode_gps"] += 1
                    else:
                        kpi_counters["mode_of"] += 1

                    if Settings.DEBUG and visualizer is not None and success_info["frame_for_debug"] is not None:
                        visualizer.draw_detections(
                            success_info["frame_for_debug"],
                            success_info["detected_objects"],
                            frame_id=str(success_info["frame_id"]),
                            position=success_info["position"],
                        )

                    fps_counter.tick()

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
                    # Yeni frame al
                    if fetch_future is None:
                        fetch_future = executor.submit(
                            _fetch_competition_step,
                            log, network, detector, movement, odometry, image_matcher,
                            resilience, kpi_counters,                             transient_failures, transient_budget
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
                        elif action == "break":
                            break
                        if is_dup:
                            consecutive_duplicate_frames += 1
                            if consecutive_duplicate_frames >= CONSECUTIVE_DUPLICATE_ABORT_THRESHOLD:
                                kpi_counters["consecutive_duplicate_abort"] = consecutive_duplicate_frames
                                log.error(
                                    f"Ardışık {consecutive_duplicate_frames} duplicate frame, "
                                    "oturum sonlandırılıyor"
                                )
                                break
                        else:
                            consecutive_duplicate_frames = 0
                        pending_result = fetch_res
                    else:
                        time.sleep(0.01)
                        continue

            except KeyboardInterrupt:
                log.warn("Interrupted by user")
                break
            except (requests.Timeout, requests.ConnectionError) as exc:
                decision = error_policy.decide_on_error(
                    RecoverableIOError(str(exc))
                )
                log.warn(f"event=error_decision decision={decision.value} type={type(exc).__name__}")
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
            except (ValueError, KeyError, TypeError) as exc:
                decision = error_policy.decide_on_error(
                    DataContractError(str(exc))
                )
                log.error(f"event=error_decision decision={decision.value} type={type(exc).__name__}")
                if decision == ErrorDecision.DEGRADE:
                    kpi_counters["error_decision_degrade"] += 1
                    time.sleep(0.2)
                    continue
                kpi_counters["error_decision_stop"] += 1
                break
            except (RuntimeError, OSError, MemoryError) as exc:
                decision = error_policy.decide_on_error(
                    FatalSystemError(str(exc))
                )
                log.error(f"event=error_decision decision={decision.value} type={type(exc).__name__}")
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


def _print_competition_result(
    log: Logger,
    frame_id,
    detected_objects: list,
    send_status: str,
    position: dict,
    gps_health: int,
) -> None:
    mode = "GPS" if gps_health == 1 else "OF"

    def _safe_float(val) -> float:
        try:
            return float(val)
        except (TypeError, ValueError):
            return 0.0

    x = _safe_float(position.get("x", 0.0))
    y = _safe_float(position.get("y", 0.0))
    z = _safe_float(position.get("z", 0.0))

    send_status_text = "OK" if send_status in {"acked", "fallback_acked", "SUCCESS"} else "FAIL"
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
            f"Forced Health Fallback={kpi_counters.get('forced_health_fallback_count', 0)} | "
            f"Mode OF={kpi_counters.get('mode_of', 0)} | "
            f"Degrade Frames={kpi_counters.get('degrade_frames', 0)} | "
            f"DupDrop={kpi_counters.get('frame_duplicate_drop', 0)} | "
            f"Timeouts(fetch/image/submit)="
            f"{kpi_counters.get('timeout_fetch', 0)}/"
            f"{kpi_counters.get('timeout_image', 0)}/"
            f"{kpi_counters.get('timeout_submit', 0)}"
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

    log.success("System shutdown complete")


def _ask_choice(prompt: str, options: dict) -> str:
    print()
    print(prompt)
    for key, desc in options.items():
        print(f"  [{key}] {desc}")
    print()

    while True:
        choice = input("  Selection: ").strip()
        if choice in options:
            return choice
        print(f"  Invalid selection, choose one of: {', '.join(options.keys())}")


def show_interactive_menu() -> dict:
    print("\n" + "=" * 56)
    print("  RUN MODE SELECTION")
    print("=" * 56)

    mode = _ask_choice(
        "  Choose run mode:",
        {
            "1": "Competition (server)",
            "2": "Simulation VID (sequential frames)",
            "3": "Simulation DET (single images)",
        },
    )

    if mode == "1":
        return {"mode": "competition", "prefer_vid": True, "show": False, "save": False}

    prefer_vid = mode == "2"

    output = _ask_choice(
        "  How do you want outputs?",
        {
            "1": "Terminal only",
            "2": "Show window",
            "3": "Save images",
            "4": "Show window + Save images",
        },
    )

    show = output in ("2", "4")
    save = output in ("3", "4")

    return {"mode": "simulate", "prefer_vid": prefer_vid, "show": show, "save": save}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TEKNOFEST AIA Runtime")
    parser.add_argument(
        "--mode",
        choices=["competition", "simulate_vid", "simulate_det"],
        default="competition",
        help="Run mode (default: competition)",
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
    parser.add_argument("--seed", type=int, default=None, help="Deterministik simülasyon için rastgele seed")
    parser.add_argument("--sequence", type=str, default=None, help="VID modunda seçilecek sekans adı (örn. uav0000123)")
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
    if args.mode == "competition" and requested_profile != "max":
        log.warn(
            "Competition mode requires deterministic-profile=max; "
            f"overriding requested profile '{requested_profile}' -> 'max'"
        )
        effective_profile = "max"

    apply_runtime_profile(effective_profile, requested_profile=requested_profile)

    print(BANNER)

    if args.interactive:
        choices = show_interactive_menu()
    else:
        if args.mode == "competition":
            choices = {
                "mode": "competition",
                "prefer_vid": True,
                "show": False,
                "save": False,
            }
        elif args.mode == "simulate_vid":
            choices = {
                "mode": "simulate",
                "prefer_vid": True,
                "show": args.show,
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
    print_system_info(log, simulate=simulate)

    if simulate:
        run_simulation(
            log,
            prefer_vid=choices["prefer_vid"],
            show=choices["show"],
            save=choices["save"],
            seed=args.seed,
            sequence=args.sequence,
        )
    else:
        run_competition(log)


def _fetch_competition_step(
    log: Logger, network: Any, detector: Any, movement: Any, odometry: Any, image_matcher: Any,
    resilience: Any, kpi_counters: dict, transient_failures: int, transient_budget: int
):
    from src.network import FrameFetchStatus
    import time

    if not resilience.before_fetch():
        cooldown_left = resilience.open_cooldown_remaining()
        wait_s = min(max(0.2, Settings.RETRY_DELAY), max(0.2, cooldown_left))
        log.warn(f"Circuit breaker OPEN; waiting cooldown ({cooldown_left:.1f}s remaining, sleep={wait_s:.1f}s)")
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
        log.warn(f"Transient frame fetch failure {transient_failures}/{transient_budget}; retrying in {delay:.1f}s")
        if transient_failures >= transient_budget:
            log.warn("Transient failure budget reached; session stays alive under wall-clock circuit breaker policy")
        time.sleep(delay)
        return None, transient_failures, "continue", False

    if fetch_result.status == FrameFetchStatus.FATAL_ERROR:
        log.error(f"Fatal frame fetch error: {fetch_result.error_type} (http={fetch_result.http_status})")
        return None, transient_failures, "break", False

    frame_data = fetch_result.frame_data or {}
    transient_failures = 0
    degrade_mode = Settings.DEGRADE_FETCH_ONLY_ENABLED and resilience.is_degraded()
    frame_id = frame_data.get("frame_id", "unknown")
    frame_fetch_monotonic = time.monotonic()

    if fetch_result.is_duplicate:
        kpi_counters["frame_duplicate_drop"] += 1
        log.warn(f"Frame {frame_id}: duplicate metadata detected. Processing normally per specification idempotency.")

    frame = None
    use_fallback = False

    if degrade_mode:
        kpi_counters["degrade_frames"] += 1
        degrade_seq = resilience.record_degraded_frame()
        heavy_every = max(1, int(Settings.DEGRADE_SEND_INTERVAL_FRAMES))
        should_try_heavy = (degrade_seq % heavy_every) == 0
        if should_try_heavy:
            frame = network.download_image(frame_data)
            timeout_snapshot = network.consume_timeout_counters()
            kpi_counters["timeout_fetch"] += timeout_snapshot.get("fetch", 0)
            kpi_counters["timeout_image"] += timeout_snapshot.get("image", 0)
            kpi_counters["timeout_submit"] += timeout_snapshot.get("submit", 0)
            if frame is None:
                use_fallback = True
                log.warn(f"Frame {frame_id}: degrade heavy pass image download failed, sending fallback result")
            else:
                log.info(f"Frame {frame_id}: degrade heavy pass (every {heavy_every} frames)")
        else:
            use_fallback = True
            log.info(f"Frame {frame_id}: degraded fetch-only fallback (slot {degrade_seq}/{heavy_every})")
    else:
        frame = network.download_image(frame_data)
        timeout_snapshot = network.consume_timeout_counters()
        kpi_counters["timeout_fetch"] += timeout_snapshot.get("fetch", 0)
        kpi_counters["timeout_image"] += timeout_snapshot.get("image", 0)
        kpi_counters["timeout_submit"] += timeout_snapshot.get("submit", 0)
        if frame is None:
            use_fallback = True
            log.warn(f"Frame {frame_id}: image download failed, sending fallback result")

    if use_fallback:
        gps_health = int(float(frame_data.get("gps_health", 0)))
        if gps_health == 0:
            if hasattr(odometry, "predict_without_measurement"):
                last_position = odometry.predict_without_measurement(
                    reason_code="frame_download_failed",
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
                "quality_flag": "degraded" if gps_health == 0 else "nominal",
                "reason_code": "frame_download_failed",
            }
        )
        runtime_meta = _mark_runtime_meta_input_quality(runtime_meta, frame_data)
        pending_result = {
            "frame_id": frame_id, "frame_data": frame_data, "detected_objects": [],
            "frame": None, "position": last_position,
            "degraded": degrade_mode, "pending_ttl": 1 if degrade_mode else None,
            "detected_translation": {
                "translation_x": last_position["x"],
                "translation_y": last_position["y"],
                "translation_z": last_position["z"],
            },
            "localization_runtime": runtime_meta,
            "base_position": dict(last_position),
            "frame_fetch_monotonic": frame_fetch_monotonic,
            "frame_shape": None, "detected_undefined_objects": [],
            "is_duplicate": fetch_result.is_duplicate,
        }
    else:
        frame_ctx = FrameContext(frame)
        detected_objects = detector.detect(frame)
        detected_objects = movement.annotate(detected_objects, frame_ctx=frame_ctx)
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
        runtime_meta = _mark_runtime_meta_input_quality(runtime_meta, frame_data)
        pending_result = {
            "frame_id": frame_id, "frame_data": frame_data, "detected_objects": detected_objects,
            "frame": frame, "position": position, "degraded": degrade_mode,
            "pending_ttl": 1 if degrade_mode else None,
            "detected_translation": {
                "translation_x": position["x"],
                "translation_y": position["y"],
                "translation_z": position["z"],
            },
            "localization_runtime": runtime_meta,
            "base_position": dict(position),
            "frame_fetch_monotonic": frame_fetch_monotonic,
            "frame_shape": frame.shape, "detected_undefined_objects": undefined_objects,
            "is_duplicate": fetch_result.is_duplicate,
        }

    if frame_data.get("gps_health_fallback_used"):
        kpi_counters["forced_health_fallback_count"] += 1

    return pending_result, transient_failures, "process", fetch_result.is_duplicate


def _submit_competition_step(
    log: Logger, network: Any, resilience: Any, odometry: VisualOdometry, kpi_counters: dict, pending_result: dict,
    ack_failures: int, ack_failure_budget: int,
    consecutive_permanent_rejects: int, permanent_reject_abort_threshold: int,
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
        raise DataContractError(
            f"pending_result missing keys: {sorted(list(missing))}"
        )

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
        frame_id, detected_objects, detected_translation,
        frame_data=frame_data, frame_shape=pending_result["frame_shape"],
        degrade=bool(pending_result.get("degraded", False)),
        detected_undefined_objects=pending_result.get("detected_undefined_objects"),
    )

    timeout_snapshot = network.consume_timeout_counters()
    kpi_counters["timeout_fetch"] += timeout_snapshot.get("fetch", 0)
    kpi_counters["timeout_image"] += timeout_snapshot.get("image", 0)
    kpi_counters["timeout_submit"] += timeout_snapshot.get("submit", 0)

    guard_snapshot = network.consume_payload_guard_counters()
    kpi_counters["payload_preflight_reject_count"] += guard_snapshot.get("preflight_reject", 0)
    kpi_counters["payload_clipped_count"] += guard_snapshot.get("payload_clipped", 0)

    pending_result_snapshot = dict(pending_result)

    pending_result, should_abort_session, success_cycle = apply_send_result_status(
        send_status=send_status, pending_result=pending_result, kpi_counters=kpi_counters,
    )

    if pending_result is None and not success_cycle:
        consecutive_permanent_rejects += 1
        log.warn(
            f"Frame {frame_id}: permanent rejected, frame dropped "
            f"({consecutive_permanent_rejects}/{permanent_reject_abort_threshold})"
        )
        if consecutive_permanent_rejects >= permanent_reject_abort_threshold:
            log.error("Consecutive permanent reject threshold reached, aborting session")
            return None, ack_failures, "break", consecutive_permanent_rejects
        return None, ack_failures, "continue", consecutive_permanent_rejects

    if success_cycle:
        ack_failures = 0
        consecutive_permanent_rejects = 0
        resilience.on_success_cycle()
        pending_result_snapshot["position"] = position_for_success
        success_info = {
            "frame_for_debug": pending_result_snapshot.get("frame"),
            "detected_objects": detected_objects,
            "position": pending_result_snapshot.get("position"),
            "frame_id": frame_id,
            "frame_data": frame_data
        }
        return pending_result, ack_failures, ("process", success_info), consecutive_permanent_rejects
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

        if pending_result is not None:
            pending_ttl = pending_result.get("pending_ttl")
            if pending_ttl is not None:
                pending_ttl = int(pending_ttl) - 1
                pending_result["pending_ttl"] = pending_ttl
                if pending_ttl <= 0:
                    log.warn(
                        f"Frame {frame_id}: stale degraded pending result dropped after TTL"
                    )
                    pending_result = None

        time.sleep(delay)
        return pending_result, ack_failures, "continue", consecutive_permanent_rejects


if __name__ == "__main__":
    main()
