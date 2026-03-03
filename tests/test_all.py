"""Tüm birim testleri."""

import copy
import json
import time
import unittest
from unittest.mock import Mock, patch, mock_open

import numpy as np
import pytest

from config.settings import Settings

# ─── Soft imports (environment-dependent) ────────────────────────────────────
try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

try:
    from src.network import (
        NetworkManager,
        FrameFetchResult,
        FrameFetchStatus,
        SendResultStatus,
    )
except Exception:  # pragma: no cover
    NetworkManager = None
    FrameFetchResult = None
    FrameFetchStatus = None
    SendResultStatus = None

try:
    from src.detection import ObjectDetector
except Exception:  # pragma: no cover
    ObjectDetector = None

try:
    from src.image_matcher import ImageMatcher
except Exception:  # pragma: no cover
    ImageMatcher = None

try:
    from src.movement import MovementEstimator
except Exception:  # pragma: no cover
    MovementEstimator = None

try:
    import main as main_module
except Exception:  # pragma: no cover
    main_module = None

from src.send_state import apply_send_result_status
from src.resilience import ResilienceState, SessionResilienceController
from src.runtime_profile import apply_runtime_profile
from src.competition_contract import (
    DataContractError,
    ErrorDecision,
    ErrorPolicy,
    FatalSystemError,
    RecoverableIOError,
)
from src.payload_schema import CompetitionPayloadSchema
from src.utils import Logger, log_json_to_disk, _sanitize_log_component, _prune_old_logs
from main import run_simulation


@unittest.skipUnless(main_module is not None, "main runtime missing")
class TestRuntimeModeSelection:
    def test_parse_args_default_mode_is_visual_validation(self):
        with patch("sys.argv", ["main.py"]):
            args = main_module.parse_args()
        assert args.mode == Settings.DEFAULT_RUNTIME_MODE

    def test_visual_validation_mode_runs_simulation_path(self):
        with patch.object(main_module, "parse_args") as parse_args_mock, patch.object(
            main_module, "run_simulation"
        ) as run_simulation_mock, patch.object(
            main_module, "run_competition"
        ) as run_competition_mock, patch.object(
            main_module, "apply_runtime_profile"
        ), patch.object(
            main_module, "apply_runtime_overrides"
        ), patch.object(
            main_module, "print_system_info"
        ), patch(
            "main.print", return_value=None
        ):
            parse_args_mock.return_value = type(
                "Args",
                (),
                {
                    "mode": "visual_validation",
                    "interactive": False,
                    "deterministic_profile": "balanced",
                    "base_url": None,
                    "team_name": None,
                    "show": False,
                    "save": False,
                    "seed": None,
                    "sequence": None,
                },
            )()

            main_module.main()

        assert run_simulation_mock.called
        assert not run_competition_mock.called

    def test_interactive_competition_enforces_max_profile(self):
        with patch.object(main_module, "parse_args") as parse_args_mock, patch.object(
            main_module, "show_interactive_menu"
        ) as show_interactive_menu_mock, patch.object(
            main_module, "run_simulation"
        ) as run_simulation_mock, patch.object(
            main_module, "run_competition"
        ) as run_competition_mock, patch.object(
            main_module, "apply_runtime_profile"
        ) as apply_runtime_profile_mock, patch.object(
            main_module, "apply_runtime_overrides"
        ), patch.object(
            main_module, "print_system_info"
        ), patch(
            "main.print", return_value=None
        ):
            parse_args_mock.return_value = type(
                "Args",
                (),
                {
                    "mode": "visual_validation",
                    "interactive": True,
                    "deterministic_profile": "balanced",
                    "base_url": None,
                    "team_name": None,
                    "show": False,
                    "save": False,
                    "seed": None,
                    "sequence": None,
                },
            )()
            show_interactive_menu_mock.return_value = {
                "mode": "competition",
                "prefer_vid": True,
                "show": False,
                "save": False,
            }

            main_module.main()

        assert run_competition_mock.called
        assert not run_simulation_mock.called
        assert apply_runtime_profile_mock.call_count == 2
        assert apply_runtime_profile_mock.call_args_list[-1].kwargs == {
            "requested_profile": "balanced"
        }
        assert apply_runtime_profile_mock.call_args_list[-1].args == ("max",)


class TestLogger:
    @patch("builtins.print")
    def test_logger_info(self, mock_print):
        Logger("TestModule").info("Test message")
        assert mock_print.called

    @patch("builtins.print")
    def test_logger_debug(self, mock_print):
        logger = Logger("TestModule")
        Settings.DEBUG = True
        logger.debug("show")
        assert mock_print.called
        mock_print.reset_mock()
        Settings.DEBUG = False
        logger.debug("hide")
        assert not mock_print.called
        Settings.DEBUG = True

    @patch("builtins.print")
    def test_logger_error(self, mock_print):
        Logger("TestModule").error("Error")
        assert mock_print.called

    @patch("builtins.print")
    def test_logger_warn(self, mock_print):
        Logger("TestModule").warn("Warn")
        assert mock_print.called

    @patch("builtins.print")
    def test_logger_success(self, mock_print):
        Logger("TestModule").success("OK")
        assert mock_print.called


class TestSanitizeAndLogs:
    def test_sanitize_log_component(self):
        assert _sanitize_log_component("valid_name") == "valid_name"
        assert _sanitize_log_component("invalid?name!") == "invalid_name_"
        assert _sanitize_log_component("") == "general"

    @patch("os.makedirs")
    @patch("builtins.open", new_callable=mock_open)
    @patch("src.utils._prune_old_logs")
    def test_log_json_to_disk(self, mock_prune, mock_file, mock_makedirs):
        data = {"key": "value"}
        Settings.LOG_DIR = "/fake/dir"
        log_json_to_disk(data, direction="test_dir", tag="test_tag")
        mock_makedirs.assert_called_with("/fake/dir", exist_ok=True)
        mock_file.assert_called_once()
        mock_prune.assert_called_once_with("/fake/dir")
        written = "".join(c.args[0] for c in mock_file().write.call_args_list)
        assert json.loads(written) == data

    @patch("os.listdir")
    @patch("os.path.getmtime")
    @patch("os.remove")
    def test_prune_old_logs(self, mock_remove, mock_getmtime, mock_listdir):
        mock_listdir.return_value = ["log1.json", "log2.json", "log3.json"]
        mock_getmtime.side_effect = [1.0, 3.0, 2.0]
        Settings.LOG_MAX_FILES = 1
        _prune_old_logs("/fake/dir")
        assert mock_remove.call_count == 2


class TestRuntimeProfile:
    def test_off(self):
        orig_tta = Settings.AUGMENTED_INFERENCE
        orig_fp16 = Settings.HALF_PRECISION
        apply_runtime_profile("off")
        assert Settings.AUGMENTED_INFERENCE == orig_tta
        assert Settings.HALF_PRECISION == orig_fp16

    def test_balanced(self):
        Settings.AUGMENTED_INFERENCE = True
        Settings.HALF_PRECISION = True
        apply_runtime_profile("balanced")
        assert Settings.AUGMENTED_INFERENCE is False
        assert Settings.HALF_PRECISION is True

    def test_max(self):
        Settings.AUGMENTED_INFERENCE = True
        Settings.HALF_PRECISION = True
        apply_runtime_profile("max")
        assert Settings.AUGMENTED_INFERENCE is False
        assert Settings.HALF_PRECISION is False

    def test_invalid(self):
        with pytest.raises(ValueError):
            apply_runtime_profile("invalid_profile")


# =============================================================================
#  §3  SEND STATE TESTS
# =============================================================================


class TestSendState:
    @staticmethod
    def _counters():
        return {
            "send_ok": 0,
            "send_fallback_ok": 0,
            "send_fail": 0,
            "send_permanent_reject": 0,
        }

    def test_acked(self):
        c = self._counters()
        p, abort, ok = apply_send_result_status("acked", {"x": 1}, c)
        assert p is None and not abort and ok and c["send_ok"] == 1

    def test_fallback_acked(self):
        c = self._counters()

        class FS:
            value = "fallback_acked"

        p, abort, ok = apply_send_result_status(FS(), {"x": 1}, c)
        assert p is None and not abort and ok
        assert c["send_ok"] == 1 and c["send_fallback_ok"] == 1

    def test_permanent_rejected(self):
        c = self._counters()
        p, abort, ok = apply_send_result_status("permanent_rejected", {"x": 1}, c)
        assert p is None and not abort and not ok
        assert c["send_fail"] == 1 and c["send_permanent_reject"] == 1

    def test_other_error(self):
        c = self._counters()
        p, abort, ok = apply_send_result_status("transient_error", {"x": 1}, c)
        assert p == {"x": 1} and not abort and not ok
        assert c["send_fail"] == 1 and c["send_permanent_reject"] == 0


@unittest.skipUnless(ObjectDetector is not None, "detection deps missing")
class TestEdgeMarginRatio(unittest.TestCase):
    def test_is_touching_edge_uses_ratio(self):
        from src.detection import ObjectDetector

        orig = Settings.EDGE_MARGIN_RATIO
        Settings.EDGE_MARGIN_RATIO = 0.01
        try:
            self.assertTrue(
                ObjectDetector._is_touching_edge((5, 50, 100, 150), 1000, 500)
            )
            self.assertFalse(
                ObjectDetector._is_touching_edge((15, 50, 100, 150), 1000, 500)
            )
        finally:
            Settings.EDGE_MARGIN_RATIO = orig


class TestMainAckStateMachine:
    @staticmethod
    def _counters():
        return {
            "send_ok": 0,
            "send_fail": 0,
            "send_fallback_ok": 0,
            "send_permanent_reject": 0,
        }

    def test_retryable_failure_keeps_pending(self):
        pending = {"frame_id": "f-1"}
        c = self._counters()
        p, abort, ok = apply_send_result_status("retryable_failure", pending, c)
        assert p is pending and not abort and not ok and c["send_fail"] == 1

    def test_fallback_acked_clears_pending(self):
        c = self._counters()
        p, abort, ok = apply_send_result_status(
            "fallback_acked", {"frame_id": "f-2"}, c
        )
        assert p is None and not abort and ok
        assert c["send_ok"] == 1 and c["send_fallback_ok"] == 1

    def test_permanent_rejected_drops_frame(self):
        c = self._counters()
        pending = {"frame_id": "f-3"}
        p, abort, ok = apply_send_result_status("permanent_rejected", pending, c)
        assert p is None and not abort and not ok
        assert c["send_fail"] == 1 and c["send_permanent_reject"] == 1


@unittest.skipUnless(main_module is not None, "main runtime missing")
class TestTask3ReferenceValidation(unittest.TestCase):
    def test_unique_ids_all_validated(self):
        refs = [
            {"object_id": 1, "image": np.zeros((12, 12, 3), dtype=np.uint8)},
            {"object_id": 2, "image": np.zeros((12, 12, 3), dtype=np.uint8)},
        ]
        canonical, stats, mode, reason, disabled = (
            main_module._validate_task3_references(main_module.Logger("Test"), refs)
        )
        self.assertEqual(len(canonical), 2)
        self.assertEqual(
            stats, {"total": 2, "valid": 2, "duplicate": 0, "quarantined": 0}
        )
        self.assertEqual(mode, "normal")
        self.assertEqual(reason, "ok")
        self.assertFalse(disabled)

    def test_duplicate_id_second_is_quarantined(self):
        refs = [
            {"object_id": 7, "image": np.zeros((12, 12, 3), dtype=np.uint8)},
            {"object_id": "7", "image": np.ones((12, 12, 3), dtype=np.uint8)},
        ]
        canonical, stats, mode, reason, disabled = (
            main_module._validate_task3_references(main_module.Logger("Test"), refs)
        )
        self.assertEqual(len(canonical), 1)
        self.assertEqual(canonical[0]["object_id"], 7)
        self.assertEqual(stats["duplicate"], 1)
        self.assertEqual(stats["quarantined"], 1)
        self.assertEqual(mode, "degraded")
        self.assertEqual(reason, "duplicate_detected_safe_degrade")
        self.assertFalse(disabled)

    def test_invalid_object_id_rejected(self):
        refs = [
            {"object_id": None, "image": np.zeros((12, 12, 3), dtype=np.uint8)},
            {"object_id": "abc", "image": np.zeros((12, 12, 3), dtype=np.uint8)},
        ]
        canonical, stats, mode, reason, disabled = (
            main_module._validate_task3_references(main_module.Logger("Test"), refs)
        )
        self.assertEqual(canonical, [])
        self.assertEqual(stats["valid"], 0)
        self.assertEqual(stats["quarantined"], 2)
        self.assertEqual(mode, "degraded")
        self.assertEqual(reason, "reference_quarantined_non_duplicate")
        self.assertFalse(disabled)


@unittest.skipUnless(ImageMatcher is not None, "image matcher deps missing")
class TestImageMatcherIdIntegrity(unittest.TestCase):
    def setUp(self):
        self.matcher = ImageMatcher()
        self.matcher.detector = Mock()
        self.matcher.detector.detectAndCompute.return_value = (
            [object(), object(), object(), object(), object()],
            np.ones((5, 32), dtype=np.uint8),
        )

    def test_load_references_unique_ids(self):
        refs = [
            {"object_id": 1, "image": np.zeros((16, 16, 3), dtype=np.uint8)},
            {"object_id": 2, "image": np.zeros((16, 16, 3), dtype=np.uint8)},
        ]
        loaded = self.matcher.load_references(refs)
        self.assertEqual(loaded, 2)
        self.assertEqual(self.matcher.reference_count, 2)
        self.assertEqual(self.matcher.last_load_stats["valid"], 2)
        self.assertEqual(self.matcher.id_lifecycle_states[1], "loaded")
        self.assertEqual(self.matcher.id_lifecycle_states[2], "loaded")

    def test_load_references_duplicate_id_quarantined(self):
        refs = [
            {"object_id": 3, "image": np.zeros((16, 16, 3), dtype=np.uint8)},
            {"object_id": 3, "image": np.ones((16, 16, 3), dtype=np.uint8)},
        ]
        loaded = self.matcher.load_references(refs)
        self.assertEqual(loaded, 1)
        self.assertEqual(self.matcher.reference_count, 1)
        self.assertEqual(self.matcher.last_load_stats["duplicate"], 1)
        self.assertEqual(self.matcher.last_load_stats["quarantined"], 1)

    def test_match_output_never_contains_duplicate_id(self):
        refs = [
            {"object_id": 5, "image": np.zeros((16, 16, 3), dtype=np.uint8)},
            {"object_id": 5, "image": np.ones((16, 16, 3), dtype=np.uint8)},
        ]
        self.matcher.load_references(refs)
        with patch.object(
            self.matcher,
            "_match_reference",
            return_value=(1.0, 1.0, 6.0, 6.0),
        ):
            out = self.matcher.match(np.zeros((32, 32, 3), dtype=np.uint8))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["object_id"], 5)
        self.assertEqual(self.matcher.id_lifecycle_states[5], "matched")


class _StubLog:
    def __init__(self):
        self.lines = []

    def info(self, msg):
        self.lines.append(("info", msg))

    def warn(self, msg):
        self.lines.append(("warn", msg))

    def error(self, msg):
        self.lines.append(("error", msg))


class TestSessionResilience:
    def _ctrl(self):
        return SessionResilienceController(_StubLog())

    def _setup_settings(self):
        Settings.CB_TRANSIENT_WINDOW_SEC = 2.0
        Settings.CB_TRANSIENT_MAX_EVENTS = 3
        Settings.CB_OPEN_COOLDOWN_SEC = 0.2
        Settings.CB_MAX_OPEN_CYCLES = 2
        Settings.CB_SESSION_MAX_TRANSIENT_SEC = 0.4

    def test_transient_window_opens_breaker(self):
        self._setup_settings()
        c = self._ctrl()
        c.on_fetch_transient()
        c.on_fetch_transient()
        assert c.state == ResilienceState.DEGRADED
        c.on_fetch_transient()
        assert c.state == ResilienceState.OPEN
        assert c.stats.breaker_open_count == 1

    def test_open_to_half_open_to_normal_recovery(self):
        self._setup_settings()
        c = self._ctrl()
        c.on_fetch_transient()
        c.on_fetch_transient()
        c.on_fetch_transient()
        assert c.state == ResilienceState.OPEN
        assert not c.before_fetch()
        time.sleep(0.25)
        assert c.before_fetch()
        assert c.state == ResilienceState.DEGRADED
        c.on_success_cycle()
        assert c.state == ResilienceState.NORMAL
        assert c.stats.recovered_count >= 1

    def test_session_wall_clock_abort(self):
        self._setup_settings()
        c = self._ctrl()
        c.on_ack_failure()
        assert c.state == ResilienceState.DEGRADED
        time.sleep(0.45)
        reason = c.should_abort()
        assert reason is not None
        assert "Transient wall time" in reason or "aborting" in reason.lower()

    def test_breaker_open_cycles_abort(self):
        self._setup_settings()
        c = self._ctrl()
        assert c.should_abort() is None
        c.on_ack_failure()
        c._non_normal_since = time.monotonic() - 1.0
        c.state = ResilienceState.DEGRADED
        reason = c.should_abort()
        assert reason is not None


@patch("src.data_loader.DatasetLoader")
@patch("main.ObjectDetector")
@patch("main.VisualOdometry")
@patch("main.MovementEstimator")
@patch("main.Logger")
@patch("main.Visualizer")
@patch("src.image_matcher.ImageMatcher")
def test_run_simulation_stops_on_max_frames(
    MockImageMatcher,
    MockVisualizer,
    MockLogger,
    MockEstimator,
    MockOdometry,
    MockDetector,
    MockDatasetLoader,
):
    mock_loader = MockDatasetLoader.return_value
    mock_loader.is_ready = True
    mock_loader.__iter__.return_value = [
        {"frame": None, "frame_idx": 0, "server_data": {}, "gps_health": 1},
        {"frame": None, "frame_idx": 1, "server_data": {}, "gps_health": 1},
    ]
    MockDetector.return_value.detect.return_value = []
    MockEstimator.return_value.annotate.return_value = []
    MockOdometry.return_value.update.return_value = {"x": 0.0, "y": 0.0, "z": 0.0}
    MockImageMatcher.return_value.match.return_value = []

    with patch("config.settings.Settings.MAX_FRAMES", 1):
        run_simulation(MockLogger(), prefer_vid=False, show=False, save=False)

    assert mock_loader.__iter__.called
    assert MockDetector.return_value.detect.call_count == 1


@unittest.skipUnless(
    cv2 is not None and MovementEstimator is not None, "opencv/runtime deps missing"
)
class TestMovementCompensation(unittest.TestCase):
    def setUp(self):
        self._orig = {
            "MOTION_COMP_ENABLED": Settings.MOTION_COMP_ENABLED,
            "MOVEMENT_MIN_HISTORY": Settings.MOVEMENT_MIN_HISTORY,
            "MOVEMENT_THRESHOLD_PX": Settings.MOVEMENT_THRESHOLD_PX,
            "MOVEMENT_MATCH_DISTANCE_PX": Settings.MOVEMENT_MATCH_DISTANCE_PX,
            "MOTION_COMP_MIN_FEATURES": Settings.MOTION_COMP_MIN_FEATURES,
        }
        Settings.MOVEMENT_MIN_HISTORY = 2
        Settings.MOVEMENT_THRESHOLD_PX = 8.0
        Settings.MOVEMENT_MATCH_DISTANCE_PX = 200.0
        Settings.MOTION_COMP_MIN_FEATURES = 20

    def tearDown(self):
        for key, value in self._orig.items():
            setattr(Settings, key, value)

    def _frame(self, shift_x: int = 0):
        rng = np.random.default_rng(42)
        base = np.zeros((320, 320, 3), dtype=np.uint8)
        for _ in range(600):
            x = int(rng.integers(5, 315))
            y = int(rng.integers(5, 315))
            cv2.circle(base, (x, y), 1, (255, 255, 255), -1)
        m = np.float32([[1, 0, shift_x], [0, 1, 0]])
        return cv2.warpAffine(base, m, (320, 320))

    def _vehicle(self, x1, y1, x2, y2):
        return [
            {
                "cls": "0",
                "top_left_x": x1,
                "top_left_y": y1,
                "bottom_right_x": x2,
                "bottom_right_y": y2,
                "landing_status": "-1",
            }
        ]

    def test_stationary_vehicle_with_camera_pan_marked_static(self):
        Settings.MOTION_COMP_ENABLED = True
        est = MovementEstimator()
        est.annotate(self._vehicle(100, 100, 140, 140), frame_ctx=self._frame(0))
        out = est.annotate(self._vehicle(112, 100, 152, 140), frame_ctx=self._frame(12))
        self.assertEqual(out[0]["motion_status"], "0")

    def test_actual_motion_preserved_with_compensation(self):
        Settings.MOTION_COMP_ENABLED = True
        est = MovementEstimator()
        est.annotate(self._vehicle(100, 100, 140, 140), frame_ctx=self._frame(0))
        out = est.annotate(self._vehicle(124, 100, 164, 140), frame_ctx=self._frame(12))
        self.assertEqual(out[0]["motion_status"], "1")

    def test_low_feature_fallback_no_crash(self):
        Settings.MOTION_COMP_ENABLED = True
        est = MovementEstimator()
        blank = np.zeros((320, 320, 3), dtype=np.uint8)
        est.annotate(self._vehicle(100, 100, 140, 140), frame_ctx=blank)
        out = est.annotate(self._vehicle(108, 100, 148, 140), frame_ctx=blank)
        self.assertIn(out[0]["motion_status"], {"0", "1", "-1"})

    def test_first_frame_warmup_not_moving(self):
        Settings.MOTION_COMP_ENABLED = True
        est = MovementEstimator()
        out = est.annotate(self._vehicle(100, 100, 140, 140), frame_ctx=self._frame(0))
        self.assertEqual(out[0]["motion_status"], "0")


@unittest.skipUnless(
    NetworkManager is not None and FrameFetchStatus is not None, "network deps missing"
)
class TestFrameDedup(unittest.TestCase):
    def setUp(self):
        self._orig = {
            "MAX_RETRIES": Settings.MAX_RETRIES,
            "SEEN_FRAME_LRU_SIZE": Settings.SEEN_FRAME_LRU_SIZE,
        }
        Settings.MAX_RETRIES = 1
        Settings.SEEN_FRAME_LRU_SIZE = 2

    def tearDown(self):
        for k, v in self._orig.items():
            setattr(Settings, k, v)

    def test_duplicate_frame_id_is_marked(self):
        mgr = NetworkManager(base_url="http://test", simulation_mode=False)
        r1, r2 = Mock(status_code=200), Mock(status_code=200)
        r1.json.return_value = {"id": "frame-1", "image_url": "/a.jpg"}
        r2.json.return_value = {"id": "frame-1", "image_url": "/a.jpg"}
        mgr.session.get = Mock(side_effect=[r1, r2])
        first = mgr.get_frame()
        second = mgr.get_frame()
        self.assertEqual(first.status, FrameFetchStatus.OK)
        self.assertFalse(first.is_duplicate)
        self.assertTrue(second.is_duplicate)

    def test_seen_frame_lru_evicts_oldest(self):
        mgr = NetworkManager(base_url="http://test", simulation_mode=False)
        self.assertFalse(mgr._mark_seen_frame("A"))
        self.assertFalse(mgr._mark_seen_frame("B"))
        self.assertFalse(mgr._mark_seen_frame("C"))
        self.assertFalse(mgr._mark_seen_frame("A"))


@unittest.skipUnless(
    NetworkManager is not None and SendResultStatus is not None, "network deps missing"
)
class TestIdempotencySubmit(unittest.TestCase):
    def setUp(self):
        self._orig = {
            "MAX_RETRIES": Settings.MAX_RETRIES,
            "IDEMPOTENCY_KEY_PREFIX": Settings.IDEMPOTENCY_KEY_PREFIX,
        }
        Settings.MAX_RETRIES = 1
        Settings.IDEMPOTENCY_KEY_PREFIX = "aia"

    def tearDown(self):
        for k, v in self._orig.items():
            setattr(Settings, k, v)

    def test_idempotency_header_is_sent(self):
        mgr = NetworkManager(base_url="http://test", simulation_mode=False)
        mgr.session.post = Mock(return_value=Mock(status_code=200))
        ok = mgr.send_result(
            frame_id="frame-7",
            detected_objects=[],
            detected_translation={
                "translation_x": 0,
                "translation_y": 0,
                "translation_z": 0,
            },
            frame_data={"id": "frame-7", "url": "/f/7"},
            frame_shape=None,
        )
        self.assertEqual(ok, SendResultStatus.ACKED)
        key = mgr.session.post.call_args.kwargs["headers"]["Idempotency-Key"]
        self.assertTrue(key.startswith("aia:"))
        self.assertTrue(key.endswith(":frame-7"))

    def test_second_submit_same_frame_is_blocked(self):
        mgr = NetworkManager(base_url="http://test", simulation_mode=False)
        mgr.session.post = Mock(return_value=Mock(status_code=200))
        kw = dict(
            frame_id="frame-9",
            detected_objects=[],
            detected_translation={
                "translation_x": 0,
                "translation_y": 0,
                "translation_z": 0,
            },
            frame_data={"id": "frame-9", "url": "/f/9"},
            frame_shape=None,
        )
        first = mgr.send_result(**kw)
        second = mgr.send_result(**kw)

        self.assertEqual(first, SendResultStatus.ACKED)
        self.assertEqual(second, SendResultStatus.ACKED)

        self.assertEqual(mgr.session.post.call_count, 2)


@unittest.skipUnless(
    requests is not None and NetworkManager is not None, "network deps missing"
)
class TestNetworkTimeouts(unittest.TestCase):
    def setUp(self):
        self._orig = {
            "MAX_RETRIES": Settings.MAX_RETRIES,
            "REQUEST_TIMEOUT": Settings.REQUEST_TIMEOUT,
            "REQUEST_CONNECT_TIMEOUT_SEC": Settings.REQUEST_CONNECT_TIMEOUT_SEC,
            "REQUEST_READ_TIMEOUT_SEC_FRAME_META": Settings.REQUEST_READ_TIMEOUT_SEC_FRAME_META,
            "REQUEST_READ_TIMEOUT_SEC_IMAGE": Settings.REQUEST_READ_TIMEOUT_SEC_IMAGE,
            "REQUEST_READ_TIMEOUT_SEC_SUBMIT": Settings.REQUEST_READ_TIMEOUT_SEC_SUBMIT,
            "BACKOFF_BASE_SEC": Settings.BACKOFF_BASE_SEC,
            "BACKOFF_MAX_SEC": Settings.BACKOFF_MAX_SEC,
            "BACKOFF_JITTER_RATIO": Settings.BACKOFF_JITTER_RATIO,
        }
        Settings.MAX_RETRIES = 1
        Settings.REQUEST_TIMEOUT = 5
        Settings.REQUEST_CONNECT_TIMEOUT_SEC = 1.5
        Settings.REQUEST_READ_TIMEOUT_SEC_FRAME_META = 2.5
        Settings.REQUEST_READ_TIMEOUT_SEC_IMAGE = 4.0
        Settings.REQUEST_READ_TIMEOUT_SEC_SUBMIT = 3.5
        Settings.BACKOFF_BASE_SEC = 0.4
        Settings.BACKOFF_MAX_SEC = 5.0
        Settings.BACKOFF_JITTER_RATIO = 0.25

    def tearDown(self):
        for k, v in self._orig.items():
            setattr(Settings, k, v)

    def test_timeout_tuple_is_used_per_endpoint(self):
        mgr = NetworkManager(base_url="http://test", simulation_mode=False)
        frame_resp, image_resp, submit_resp = (
            Mock(status_code=204),
            Mock(status_code=500),
            Mock(status_code=200),
        )
        get_calls = []

        def fake_get(url, **kwargs):
            get_calls.append((url, kwargs))
            return (
                frame_resp if url.endswith(Settings.ENDPOINT_NEXT_FRAME) else image_resp
            )

        mgr.session.get = fake_get
        mgr.session.post = Mock(return_value=submit_resp)
        mgr.get_frame()
        mgr.download_image({"frame_url": "/frame.jpg"})
        mgr.send_result(
            frame_id="f1",
            detected_objects=[],
            detected_translation={
                "translation_x": 0,
                "translation_y": 0,
                "translation_z": 0,
            },
            frame_data={"id": "f1", "url": "/frame/1"},
            frame_shape=None,
        )
        expected_frame = (
            Settings.REQUEST_CONNECT_TIMEOUT_SEC,
            Settings.REQUEST_READ_TIMEOUT_SEC_FRAME_META,
        )
        expected_image = (
            Settings.REQUEST_CONNECT_TIMEOUT_SEC,
            Settings.REQUEST_READ_TIMEOUT_SEC_IMAGE,
        )
        expected_submit = (
            Settings.REQUEST_CONNECT_TIMEOUT_SEC,
            Settings.REQUEST_READ_TIMEOUT_SEC_SUBMIT,
        )
        self.assertEqual(get_calls[0][1]["timeout"], expected_frame)
        self.assertEqual(get_calls[1][1]["timeout"], expected_image)
        self.assertEqual(mgr.session.post.call_args.kwargs["timeout"], expected_submit)

    def test_backoff_delay_stays_in_expected_bounds(self):
        mgr = NetworkManager(base_url="http://test", simulation_mode=False)
        vals = [mgr._compute_backoff_delay(4) for _ in range(200)]
        assert all(2.4 <= v <= 4.0 for v in vals)
        large = [mgr._compute_backoff_delay(12) for _ in range(200)]
        assert all(3.75 <= v <= 5.0 for v in large)

    def test_timeout_counters_only_increment_on_timeout(self):
        mgr = NetworkManager(base_url="http://test", simulation_mode=False)
        mgr.session.get = Mock(side_effect=requests.Timeout("x"))
        result = mgr.get_frame()
        self.assertEqual(result.status.value, "transient_error")
        counts = mgr.consume_timeout_counters()
        self.assertEqual(counts["fetch"], 1)
        self.assertEqual(counts["image"], 0)


class _Response:
    def __init__(self, status_code):
        self.status_code = status_code


@unittest.skipUnless(NetworkManager is not None, "network deps missing")
class TestNetworkPayloadGuard(unittest.TestCase):
    def setUp(self):
        self._orig = {
            "RESULT_MAX_OBJECTS": Settings.RESULT_MAX_OBJECTS,
            "RESULT_CLASS_QUOTA": dict(Settings.RESULT_CLASS_QUOTA),
            "MAX_RETRIES": Settings.MAX_RETRIES,
        }
        Settings.RESULT_MAX_OBJECTS = 100
        Settings.RESULT_CLASS_QUOTA = {"0": 40, "1": 40, "2": 10, "3": 10}
        Settings.MAX_RETRIES = 3
        self.net = NetworkManager(base_url="http://localhost", simulation_mode=False)
        self.net._sleep_with_backoff = lambda attempt: None

    def tearDown(self):
        for k, v in self._orig.items():
            setattr(Settings, k, v)

    @staticmethod
    def _obj(cls, conf, x, y):
        return {
            "cls": cls,
            "landing_status": "-1",
            "motion_status": "0",
            "top_left_x": x,
            "top_left_y": y,
            "bottom_right_x": x + 10,
            "bottom_right_y": y + 10,
            "_confidence": conf,
        }

    def test_limit_and_class_quota_are_enforced(self):
        objs = []
        for i in range(60):
            objs.append(self._obj("0", 0.99 - i * 0.001, i, i))
            objs.append(self._obj("1", 0.98 - i * 0.001, i, i + 1))
        for i in range(20):
            objs.append(self._obj("2", 0.97 - i * 0.001, i, i + 2))
            objs.append(self._obj("3", 0.96 - i * 0.001, i, i + 3))
        capped, stats = self.net._apply_object_caps(objs, frame_id="f-1")
        self.assertEqual(len(capped), 100)
        cc = {"0": 0, "1": 0, "2": 0, "3": 0}
        for d in capped:
            cc[d["cls"]] += 1
        self.assertLessEqual(cc["0"], 40)
        self.assertLessEqual(cc["1"], 40)
        self.assertGreater(stats["dropped_total"], 0)

    def test_capping_is_deterministic(self):
        src = [
            self._obj("0", 0.90, 5, 5),
            self._obj("0", 0.80, 6, 6),
            self._obj("1", 0.95, 4, 8),
            self._obj("2", 0.70, 1, 2),
            self._obj("3", 0.60, 3, 1),
        ]
        a, _ = self.net._apply_object_caps(src + src, frame_id="f-2")
        b, _ = self.net._apply_object_caps(
            list(reversed(copy.deepcopy(src + src))), frame_id="f-2"
        )
        self.assertEqual(a, b)

    def test_preflight_invalid_payload_forces_fallback(self):
        payload, rej, clip = self.net._preflight_validate_and_normalize_payload(
            payload={"id": 1, "user": "u"},
            frame_shape=None,
            frame_id="f-3",
        )
        self.assertTrue(rej)
        self.assertFalse(clip)

    def test_4xx_then_fallback_200_returns_fallback_acked(self):
        self.net.session = Mock()
        self.net.session.post = Mock(side_effect=[_Response(400), _Response(200)])
        status = self.net.send_result(
            frame_id="f-4",
            detected_objects=[{"cls": "0", "confidence": 0.9}],
            detected_translation={
                "translation_x": 1,
                "translation_y": 2,
                "translation_z": 3,
            },
            frame_data={"id": "f-4", "user": "u", "url": "frame-url"},
            frame_shape=(1080, 1920, 3),
        )
        self.assertEqual(status, SendResultStatus.FALLBACK_ACKED)

    def test_4xx_then_fallback_4xx_returns_permanent_rejected(self):
        self.net.session = Mock()
        self.net.session.post = Mock(side_effect=[_Response(422), _Response(400)])
        status = self.net.send_result(
            frame_id="f-5",
            detected_objects=[{"cls": "0", "confidence": 0.9}],
            detected_translation={
                "translation_x": 1,
                "translation_y": 2,
                "translation_z": 3,
            },
            frame_data={"id": "f-5", "user": "u", "url": "frame-url"},
            frame_shape=(1080, 1920, 3),
        )
        self.assertEqual(status, SendResultStatus.PERMANENT_REJECTED)

    def test_5xx_retries_exhausted_returns_retryable_failure(self):
        self.net.session = Mock()
        self.net.session.post = Mock(side_effect=[_Response(500)] * 3)
        status = self.net.send_result(
            frame_id="f-6",
            detected_objects=[{"cls": "0", "confidence": 0.9}],
            detected_translation={
                "translation_x": 1,
                "translation_y": 2,
                "translation_z": 3,
            },
            frame_data={"id": "f-6", "user": "u", "url": "frame-url"},
            frame_shape=(1080, 1920, 3),
        )
        self.assertEqual(status, SendResultStatus.RETRYABLE_FAILURE)


class TestCompetitionPayloadSchema(unittest.TestCase):
    def test_legacy_motion_alias_is_normalized_to_canonical(self):
        obj = {
            "cls": "0",
            "landing_status": "1",
            "movement_status": "0",
            "top_left_x": 1,
            "top_left_y": 2,
            "bottom_right_x": 20,
            "bottom_right_y": 30,
        }
        out, alias_count = CompetitionPayloadSchema.canonicalize_objects(
            [obj], frame_shape=(100, 100)
        )
        self.assertEqual(alias_count, 1)
        self.assertEqual(len(out), 1)
        self.assertIn("motion_status", out[0])
        self.assertNotIn("movement_status", out[0])
        self.assertEqual(out[0]["motion_status"], 0)

    def test_settings_self_check_requires_canonical_motion_field(self):
        old = Settings.MOTION_FIELD_NAME
        try:
            Settings.MOTION_FIELD_NAME = "movement_status"
            with self.assertRaises(DataContractError):
                CompetitionPayloadSchema.self_check()
        finally:
            Settings.MOTION_FIELD_NAME = old


class TestErrorPolicy(unittest.TestCase):
    def test_recoverable_error_then_degrade_after_budget(self):
        p = ErrorPolicy(retry_budget=1, degrade_budget=3)
        self.assertEqual(
            p.decide_on_error(RecoverableIOError("x")),
            ErrorDecision.RETRY,
        )
        self.assertEqual(
            p.decide_on_error(RecoverableIOError("x2")),
            ErrorDecision.DEGRADE,
        )

    def test_data_contract_error_stops_after_degrade_budget(self):
        p = ErrorPolicy(retry_budget=5, degrade_budget=1)
        self.assertEqual(
            p.decide_on_error(DataContractError("bad")),
            ErrorDecision.DEGRADE,
        )
        self.assertEqual(
            p.decide_on_error(DataContractError("bad2")),
            ErrorDecision.STOP,
        )

    def test_fatal_error_is_always_stop(self):
        p = ErrorPolicy()
        self.assertEqual(
            p.decide_on_error(FatalSystemError("fatal")),
            ErrorDecision.STOP,
        )


class _DummyDetector:
    detect_calls = 0

    def detect(self, frame):
        _DummyDetector.detect_calls += 1
        return []


class _DummyMovement:
    def annotate(self, detections, frame_ctx=None):
        return detections


class _DummyOdometry:
    def update(self, frame, frame_data):
        return {"x": 0.0, "y": 0.0, "z": 0.0}

    @staticmethod
    def project_position_with_latency(position, dt_sec, max_dt_sec, max_delta_m):
        return dict(position), 0.0, 0.0


class _FakeNetwork:
    frame_results = []
    timeout_snapshots = []
    download_calls = 0
    send_calls = 0

    def __init__(self, base_url=None, simulation_mode=None):
        pass

    @classmethod
    def reset(cls):
        cls.frame_results = []
        cls.timeout_snapshots = []
        cls.download_calls = 0
        cls.send_calls = 0
        _DummyDetector.detect_calls = 0

    def start_session(self):
        return True

    def get_task3_references(self):
        return []

    def get_frame(self):
        return self.frame_results.pop(0)

    def consume_timeout_counters(self):
        if self.timeout_snapshots:
            return self.timeout_snapshots.pop(0)
        return {"fetch": 0, "image": 0, "submit": 0}

    def consume_payload_guard_counters(self):
        return {"preflight_reject": 0, "payload_clipped": 0}

    def download_image(self, frame_data):
        _FakeNetwork.download_calls += 1
        return np.zeros((8, 8, 3), dtype=np.uint8)

    def send_result(
        self,
        frame_id,
        detected_objects,
        detected_translation,
        frame_data=None,
        frame_shape=None,
        degrade=False,
        detected_undefined_objects=None,
    ):
        _FakeNetwork.send_calls += 1
        return SendResultStatus.ACKED


class _Task3RefNetwork(_FakeNetwork):
    refs = []

    def get_task3_references(self):
        return list(self.refs)


@unittest.skipUnless(
    np is not None
    and FrameFetchResult is not None
    and FrameFetchStatus is not None
    and SendResultStatus is not None
    and main_module is not None,
    "runtime deps missing",
)
class TestCompetitionLoopHardening(unittest.TestCase):
    def setUp(self):
        self._orig = {
            "DEBUG": Settings.DEBUG,
            "MAX_FRAMES": Settings.MAX_FRAMES,
            "LOOP_DELAY": Settings.LOOP_DELAY,
            "FPS_REPORT_INTERVAL": Settings.FPS_REPORT_INTERVAL,
            "DEGRADE_FETCH_ONLY_ENABLED": Settings.DEGRADE_FETCH_ONLY_ENABLED,
            "CB_SESSION_MAX_TRANSIENT_SEC": Settings.CB_SESSION_MAX_TRANSIENT_SEC,
            "CB_TRANSIENT_WINDOW_SEC": Settings.CB_TRANSIENT_WINDOW_SEC,
            "CB_TRANSIENT_MAX_EVENTS": Settings.CB_TRANSIENT_MAX_EVENTS,
            "CB_OPEN_COOLDOWN_SEC": Settings.CB_OPEN_COOLDOWN_SEC,
            "CB_MAX_OPEN_CYCLES": Settings.CB_MAX_OPEN_CYCLES,
        }
        Settings.DEBUG = False
        Settings.MAX_FRAMES = 50
        Settings.LOOP_DELAY = 0.0
        Settings.FPS_REPORT_INTERVAL = 99999
        Settings.DEGRADE_FETCH_ONLY_ENABLED = False
        Settings.CB_SESSION_MAX_TRANSIENT_SEC = 300.0
        Settings.CB_TRANSIENT_WINDOW_SEC = 60.0
        Settings.CB_TRANSIENT_MAX_EVENTS = 100
        Settings.CB_OPEN_COOLDOWN_SEC = 0.01
        Settings.CB_MAX_OPEN_CYCLES = 100
        _FakeNetwork.reset()
        self.summary_calls = []

    def tearDown(self):
        for k, v in self._orig.items():
            setattr(Settings, k, v)

    def _summary_cb(self, *a, **kw):
        self.summary_calls.append(kw)

    def test_duplicate_frame_dropped_before_processing(self):
        fd = {"frame_id": "f1", "frame_url": "/f1.jpg", "gps_health": 1}
        _FakeNetwork.frame_results = [
            FrameFetchResult(
                status=FrameFetchStatus.OK, frame_data=fd, is_duplicate=False
            ),
            FrameFetchResult(
                status=FrameFetchStatus.OK, frame_data=fd, is_duplicate=True
            ),
            FrameFetchResult(status=FrameFetchStatus.END_OF_STREAM),
        ]
        _FakeNetwork.timeout_snapshots = [{"fetch": 0, "image": 0, "submit": 0}] * 8
        with patch("src.network.NetworkManager", _FakeNetwork), patch.object(
            main_module, "ObjectDetector", _DummyDetector
        ), patch.object(main_module, "MovementEstimator", _DummyMovement), patch.object(
            main_module, "VisualOdometry", _DummyOdometry
        ), patch.object(
            main_module, "_print_summary", side_effect=self._summary_cb
        ):
            main_module.run_competition(main_module.Logger("Test"))
        self.assertEqual(_FakeNetwork.download_calls, 2)
        self.assertEqual(_FakeNetwork.send_calls, 2)
        self.assertEqual(_DummyDetector.detect_calls, 2)
        self.assertEqual(
            self.summary_calls[-1]["kpi_counters"]["frame_duplicate_drop"], 1
        )

    def test_transient_fetch_timeout_recovers(self):
        fd2 = {"frame_id": "f2", "frame_url": "/f2.jpg", "gps_health": 1}
        _FakeNetwork.frame_results = [
            FrameFetchResult(
                status=FrameFetchStatus.TRANSIENT_ERROR, error_type="retries_exhausted"
            ),
            FrameFetchResult(
                status=FrameFetchStatus.OK, frame_data=fd2, is_duplicate=False
            ),
            FrameFetchResult(status=FrameFetchStatus.END_OF_STREAM),
        ]
        _FakeNetwork.timeout_snapshots = [
            {"fetch": 1, "image": 0, "submit": 0},
            {"fetch": 0, "image": 0, "submit": 0},
            {"fetch": 0, "image": 0, "submit": 0},
            {"fetch": 0, "image": 0, "submit": 0},
            {"fetch": 0, "image": 0, "submit": 0},
        ]
        with patch("src.network.NetworkManager", _FakeNetwork), patch.object(
            main_module, "ObjectDetector", _DummyDetector
        ), patch.object(main_module, "MovementEstimator", _DummyMovement), patch.object(
            main_module, "VisualOdometry", _DummyOdometry
        ), patch.object(
            main_module, "_print_summary", side_effect=self._summary_cb
        ):
            main_module.run_competition(main_module.Logger("Test"))
        self.assertEqual(_FakeNetwork.send_calls, 1)
        self.assertEqual(self.summary_calls[-1]["kpi_counters"]["timeout_fetch"], 1)

    def test_action_result_unexpected_type_handled_safely(self):
        fd1 = {"frame_id": "f1", "frame_url": "/f1.jpg", "gps_health": 1}
        fd2 = {"frame_id": "f2", "frame_url": "/f2.jpg", "gps_health": 1}
        _FakeNetwork.frame_results = [
            FrameFetchResult(
                status=FrameFetchStatus.OK, frame_data=fd1, is_duplicate=False
            ),
            FrameFetchResult(
                status=FrameFetchStatus.OK, frame_data=fd2, is_duplicate=False
            ),
            FrameFetchResult(status=FrameFetchStatus.END_OF_STREAM),
        ]
        _FakeNetwork.timeout_snapshots = [{"fetch": 0, "image": 0, "submit": 0}] * 10
        call_count = [0]

        def mock_submit(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return (None, False, 12345, 0)  # unexpected action_result
            success_info = {
                "frame_data": {"gps_health": 1},
                "frame_id": "f2",
                "detected_objects": [],
                "frame_for_debug": None,
                "position": (0.0, 0.0, 0.0),
            }
            return (None, False, (None, success_info), 0)

        with patch("src.network.NetworkManager", _FakeNetwork), patch.object(
            main_module, "ObjectDetector", _DummyDetector
        ), patch.object(main_module, "MovementEstimator", _DummyMovement), patch.object(
            main_module, "VisualOdometry", _DummyOdometry
        ), patch.object(
            main_module, "_submit_competition_step", side_effect=mock_submit
        ), patch.object(
            main_module, "_print_summary", side_effect=self._summary_cb
        ):
            main_module.run_competition(main_module.Logger("Test"))
        self.assertEqual(call_count[0], 2)

    def test_task3_server_duplicates_only_canonical_passes_matcher(self):
        _Task3RefNetwork.reset()
        _Task3RefNetwork.frame_results = [
            FrameFetchResult(status=FrameFetchStatus.END_OF_STREAM),
        ]
        _Task3RefNetwork.timeout_snapshots = [{"fetch": 0, "image": 0, "submit": 0}]
        _Task3RefNetwork.refs = [
            {"object_id": 11, "image": np.zeros((8, 8, 3), dtype=np.uint8)},
            {"object_id": 11, "image": np.ones((8, 8, 3), dtype=np.uint8)},
        ]

        with patch("src.network.NetworkManager", _Task3RefNetwork), patch.object(
            main_module, "ObjectDetector", _DummyDetector
        ), patch.object(main_module, "MovementEstimator", _DummyMovement), patch.object(
            main_module, "VisualOdometry", _DummyOdometry
        ), patch(
            "src.image_matcher.ImageMatcher"
        ) as mock_matcher_cls, patch.object(
            main_module, "_print_summary", side_effect=self._summary_cb
        ):
            matcher = mock_matcher_cls.return_value
            matcher.load_references_from_directory.return_value = 0
            matcher.load_references.side_effect = lambda refs: len(refs)
            matcher.match.return_value = []

            main_module.run_competition(main_module.Logger("Test"))

        canonical_refs = matcher.load_references.call_args[0][0]
        self.assertEqual(len(canonical_refs), 1)
        self.assertEqual(canonical_refs[0]["object_id"], 11)
        kpi = self.summary_calls[-1]["kpi_counters"]
        self.assertEqual(kpi["reference_validation_stats"]["duplicate"], 1)
        self.assertEqual(kpi["id_integrity_mode"], "degraded")
        self.assertEqual(
            kpi["id_integrity_reason_code"], "duplicate_detected_safe_degrade"
        )


class _LatencyNet:
    def __init__(self):
        self.last_translation = None

    def send_result(self, frame_id, detected_objects, detected_translation, **kwargs):
        self.last_translation = dict(detected_translation)
        return "acked"

    @staticmethod
    def consume_timeout_counters():
        return {"fetch": 0, "image": 0, "submit": 0}

    @staticmethod
    def consume_payload_guard_counters():
        return {"preflight_reject": 0, "payload_clipped": 0}


class _LatencyResilience:
    @staticmethod
    def on_success_cycle():
        return None

    @staticmethod
    def on_ack_failure():
        return None


class _LatencyOdometry:
    def __init__(self):
        self.project_calls = []

    def project_position_with_latency(self, position, dt_sec, max_dt_sec, max_delta_m):
        self.project_calls.append(
            {
                "position": dict(position),
                "dt_sec": dt_sec,
                "max_dt_sec": max_dt_sec,
                "max_delta_m": max_delta_m,
            }
        )
        dt_used = min(max(0.0, dt_sec), max_dt_sec)
        delta = min(max_delta_m, dt_used * 10.0)
        projected = {
            "x": position["x"] + delta,
            "y": position["y"],
            "z": position["z"],
        }
        return projected, delta, dt_used


@unittest.skipUnless(main_module is not None, "main runtime missing")
class TestGps0LatencyCompensation(unittest.TestCase):
    def setUp(self):
        self._orig = {
            "LATENCY_COMP_ENABLED": Settings.LATENCY_COMP_ENABLED,
            "LATENCY_COMP_MAX_MS": Settings.LATENCY_COMP_MAX_MS,
            "LATENCY_COMP_MAX_DELTA_M": Settings.LATENCY_COMP_MAX_DELTA_M,
            "LATENCY_COMP_EMA_ALPHA": Settings.LATENCY_COMP_EMA_ALPHA,
        }
        Settings.LATENCY_COMP_MAX_MS = 250.0
        Settings.LATENCY_COMP_MAX_DELTA_M = 5.0
        Settings.LATENCY_COMP_EMA_ALPHA = 0.5

    def tearDown(self):
        for key, value in self._orig.items():
            setattr(Settings, key, value)

    @staticmethod
    def _kpi():
        return {
            "send_ok": 0,
            "send_fail": 0,
            "send_fallback_ok": 0,
            "send_permanent_reject": 0,
            "timeout_fetch": 0,
            "timeout_image": 0,
            "timeout_submit": 0,
            "payload_preflight_reject_count": 0,
            "payload_clipped_count": 0,
            "compensation_apply_count": 0,
            "compensation_sum_delta_m": 0.0,
            "compensation_avg_delta_m": 0.0,
            "compensation_max_delta_m": 0.0,
        }

    @staticmethod
    def _pending(gps_health):
        return {
            "frame_id": "f-lat-1",
            "frame_data": {"frame_id": "f-lat-1", "gps_health": gps_health},
            "detected_objects": [],
            "detected_translation": {
                "translation_x": 10.0,
                "translation_y": 0.0,
                "translation_z": 2.0,
            },
            "position": {"x": 10.0, "y": 0.0, "z": 2.0},
            "base_position": {"x": 10.0, "y": 0.0, "z": 2.0},
            "frame_fetch_monotonic": 10.0,
            "frame_shape": None,
            "degraded": False,
            "detected_undefined_objects": [],
        }

    def test_feature_disabled_keeps_existing_behavior(self):
        Settings.LATENCY_COMP_ENABLED = False
        net = _LatencyNet()
        odo = _LatencyOdometry()
        pending = self._pending(gps_health=0)
        kpi = self._kpi()

        with patch("time.monotonic", return_value=10.12):
            main_module._submit_competition_step(
                Logger("Test"),
                net,
                _LatencyResilience(),
                odo,
                kpi,
                pending,
                0,
                5,
                0,
                5,
            )

        self.assertEqual(net.last_translation["translation_x"], 10.0)
        self.assertEqual(len(odo.project_calls), 0)
        self.assertEqual(kpi["compensation_apply_count"], 0)

    def test_gps_health_1_does_not_apply_compensation(self):
        Settings.LATENCY_COMP_ENABLED = True
        net = _LatencyNet()
        odo = _LatencyOdometry()
        pending = self._pending(gps_health=1)
        kpi = self._kpi()

        with patch("time.monotonic", return_value=10.12):
            main_module._submit_competition_step(
                Logger("Test"),
                net,
                _LatencyResilience(),
                odo,
                kpi,
                pending,
                0,
                5,
                0,
                5,
            )

        self.assertEqual(net.last_translation["translation_x"], 10.0)
        self.assertEqual(len(odo.project_calls), 0)
        self.assertEqual(kpi["compensation_apply_count"], 0)

    def test_gps_health_0_uses_runtime_dt_for_compensation(self):
        Settings.LATENCY_COMP_ENABLED = True
        net = _LatencyNet()
        odo = _LatencyOdometry()
        pending = self._pending(gps_health=0)
        kpi = self._kpi()

        with patch("time.monotonic", return_value=10.12):
            main_module._submit_competition_step(
                Logger("Test"),
                net,
                _LatencyResilience(),
                odo,
                kpi,
                pending,
                0,
                5,
                0,
                5,
            )

        self.assertEqual(len(odo.project_calls), 1)
        self.assertAlmostEqual(odo.project_calls[0]["dt_sec"], 0.12, places=5)
        self.assertAlmostEqual(net.last_translation["translation_x"], 11.2, places=5)
        self.assertEqual(kpi["compensation_apply_count"], 1)
        self.assertAlmostEqual(kpi["compensation_avg_delta_m"], 1.2, places=5)
        self.assertAlmostEqual(kpi["compensation_max_delta_m"], 1.2, places=5)

    def test_compensation_respects_dt_and_delta_clamp_limits(self):
        Settings.LATENCY_COMP_ENABLED = True
        Settings.LATENCY_COMP_MAX_MS = 50.0
        Settings.LATENCY_COMP_MAX_DELTA_M = 0.3
        net = _LatencyNet()
        odo = _LatencyOdometry()
        pending = self._pending(gps_health=0)
        pending["frame_fetch_monotonic"] = 20.0
        kpi = self._kpi()

        with patch("time.monotonic", return_value=20.2):
            main_module._submit_competition_step(
                Logger("Test"),
                net,
                _LatencyResilience(),
                odo,
                kpi,
                pending,
                0,
                5,
                0,
                5,
            )

        self.assertAlmostEqual(odo.project_calls[0]["max_dt_sec"], 0.05, places=6)
        self.assertAlmostEqual(net.last_translation["translation_x"], 10.3, places=6)
        self.assertEqual(kpi["compensation_apply_count"], 1)
        self.assertAlmostEqual(kpi["compensation_max_delta_m"], 0.3, places=6)


class TestLatencyCompensatorHelper(unittest.TestCase):
    def test_velocity_is_derived_from_position_diff_and_ema(self):
        from src.localization import LatencyCompensator

        comp = LatencyCompensator(ema_alpha=0.5)
        comp.update_velocity({"x": 0.0, "y": 0.0, "z": 0.0}, sample_monotonic=1.0)
        comp.update_velocity({"x": 2.0, "y": 0.0, "z": 0.0}, sample_monotonic=2.0)
        projected, delta_m, _ = comp.project_position(
            position={"x": 2.0, "y": 0.0, "z": 0.0},
            dt_sec=0.2,
            max_dt_sec=1.0,
            max_delta_m=10.0,
        )

        self.assertAlmostEqual(projected["x"], 2.2, places=6)
        self.assertAlmostEqual(delta_m, 0.2, places=6)


class TestVisualOdometryPredictOnly(unittest.TestCase):
    def test_predict_without_measurement_updates_runtime_meta(self):
        from src.localization import VisualOdometry

        odom = VisualOdometry()
        odom.position = {"x": 5.0, "y": 0.0, "z": 1.0}
        odom._last_update_monotonic = 10.0  # test bootstrap

        with patch("time.monotonic", return_value=10.1):
            pos = odom.predict_without_measurement(
                reason_code="frame_download_failed",
                gps_health=0,
            )

        self.assertIn("x", pos)
        self.assertIn("y", pos)
        self.assertIn("z", pos)
        meta = odom.get_runtime_meta()
        self.assertEqual(meta["update_mode"], "predict-only")
        self.assertEqual(meta["state_source"], "vision_predict")
        self.assertEqual(meta["quality_flag"], "degraded")
        self.assertEqual(meta["reason_code"], "frame_download_failed")
