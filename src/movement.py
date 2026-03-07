"""Taşıt hareketlilik (motion_status): 1=hareketli, 0=sabit, -1=taşıt değil.
Merkez takibi + kamera kayması kompanzasyonu ile yer değiştirme hesaplanır."""

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from src.utils import FrameContext

import cv2
import numpy as np
from config.settings import Settings


@dataclass
class _Track:
    history: Deque[Tuple[float, float, float, float]] = field(default_factory=deque)
    missed: int = 0
    last_status: str = "0"
    last_box: Optional[Tuple[float, float, float, float]] = None


class MovementEstimator:
    """Merkez takibi ile taşıt hareket durumu (1=hareketli, 0=sabit, -1=taşıt değil)."""

    def __init__(self) -> None:
        self._tracks: Dict[int, _Track] = {}
        self._next_track_id: int = 1
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_points: Optional[np.ndarray] = None
        self._flow_inv_scale: float = 1.0
        self._cam_shift_hist: Deque[Tuple[float, float]] = deque(
            maxlen=Settings.MOVEMENT_WINDOW_FRAMES
        )
        self._cam_total_x: float = 0.0
        self._cam_total_y: float = 0.0
        self._frame_width: int = Settings.MOVEMENT_THRESHOLD_REF_WIDTH
        self._is_frozen_frame: bool = False
        self._frame_diff: float = float("inf")

    def annotate(self, detections: List[Dict], frame_ctx: Optional["FrameContext"] = None) -> List[Dict]:
        if frame_ctx is not None:
            if isinstance(frame_ctx, np.ndarray):
                self._frame_width = frame_ctx.shape[1]
            else:
                self._frame_width = frame_ctx.frame.shape[1]

        cam_dx = cam_dy = 0.0
        if Settings.MOTION_COMP_ENABLED and frame_ctx is not None:
            cam_dx, cam_dy = self._estimate_camera_shift(frame_ctx)
        self._cam_shift_hist.append((cam_dx, cam_dy))
        self._cam_total_x += cam_dx
        self._cam_total_y += cam_dy

        _CAM_TOTAL_CAP = 1e6
        self._cam_total_x = max(-_CAM_TOTAL_CAP, min(_CAM_TOTAL_CAP, self._cam_total_x))
        self._cam_total_y = max(-_CAM_TOTAL_CAP, min(_CAM_TOTAL_CAP, self._cam_total_y))

        self._is_frozen_frame = self._frame_diff < Settings.FROZEN_FRAME_DIFF_THRESHOLD

        vehicles: List[Tuple[int, Dict]] = []
        for idx, det in enumerate(detections):
            cls_val = det.get("cls_int", det.get("cls"))
            if cls_val == Settings.CLASS_TASIT or (isinstance(cls_val, str) and cls_val == "0"):
                vehicles.append((idx, det))
            else:
                det["motion_status"] = "-1"

        if not vehicles:
            self._age_tracks(set())
            return detections

        centers = {idx: self._center(det) for idx, det in vehicles}
        assignments = self._match(vehicles)
        matched_track_ids = set(assignments.values())
        self._age_tracks(matched_track_ids)

        for idx, det in vehicles:
            track_id = assignments.get(idx)
            if track_id is None:
                track_id = self._create_track(centers[idx])
            track = self._tracks[track_id]
            cx, cy = centers[idx]

            from src.postprocess import _bbox
            track.last_box = _bbox(det)

            if not self._is_frozen_frame:
                track.history.append((cx, cy, self._cam_total_x, self._cam_total_y))
            track.missed = 0

            status = self._status(track.history, track.last_status)
            track.last_status = status
            det["motion_status"] = status

        return detections

    def _status(
        self,
        history: Deque[Tuple[float, float, float, float]],
        previous_status: str,
    ) -> str:
        """Sliding window: kamera-kompanze edilmiş yer değiştirme > threshold → hareketli"""
        n = len(history)
        scale = self._frame_width / Settings.MOVEMENT_THRESHOLD_REF_WIDTH
        threshold = Settings.MOVEMENT_THRESHOLD_PX * scale

        # Adaptif eşik: büyük kamera pan/tilt'ta eşiği artır
        if getattr(Settings, "MOVEMENT_ADAPTIVE_PAN_ENABLED", False) and self._cam_shift_hist:
            total_mag = sum(abs(dx) + abs(dy) for dx, dy in self._cam_shift_hist)
            avg_mag = total_mag / len(self._cam_shift_hist)
            pan_px = float(getattr(Settings, "MOVEMENT_ADAPTIVE_PAN_PX", 15.0))
            if avg_mag > pan_px:
                factor = float(getattr(Settings, "MOVEMENT_ADAPTIVE_PAN_FACTOR", 1.35))
                threshold = threshold * factor

        early_ratio = max(0.1, float(getattr(Settings, "MOVEMENT_EARLY_SIGNAL_RATIO", 0.75)))
        hysteresis_ratio = max(0.1, float(getattr(Settings, "MOVEMENT_HYSTERESIS_RATIO", 0.65)))
        move_on_threshold = threshold
        move_off_threshold = threshold * hysteresis_ratio

        if n < Settings.MOVEMENT_MIN_HISTORY:
            if n >= 2:
                x0, y0, cam0_x, cam0_y = history[0]
                x1, y1, cam1_x, cam1_y = history[-1]
                rel_dx = (x1 - x0) - (cam1_x - cam0_x)
                rel_dy = (y1 - y0) - (cam1_y - cam0_y)
                dist = (rel_dx * rel_dx + rel_dy * rel_dy) ** 0.5
                if dist >= threshold * early_ratio:
                    return "1"
                if previous_status == "1" and dist >= move_off_threshold:
                    return "1"
            return "0"

        step = max(1, Settings.MOVEMENT_MIN_HISTORY - 1)
        max_dist = 0.0

        for i in range(n - step):
            j = i + step
            x0, y0, cam0_x, cam0_y = history[i]
            x1, y1, cam1_x, cam1_y = history[j]

            rel_dx = (x1 - x0) - (cam1_x - cam0_x)
            rel_dy = (y1 - y0) - (cam1_y - cam0_y)
            dist = (rel_dx * rel_dx + rel_dy * rel_dy) ** 0.5
            if dist > max_dist:
                max_dist = dist

        if previous_status == "1":
            return "1" if max_dist >= move_off_threshold else "0"
        return "1" if max_dist >= move_on_threshold else "0"

    def _match(self, vehicles: List[Tuple[int, Dict]]) -> Dict[int, int]:
        assignments: Dict[int, int] = {}
        if not self._tracks:
            return assignments

        candidates: List[Tuple[float, int, int]] = []
        algo = getattr(Settings, "MOTION_ALGO", "flow").lower()

        if algo == "iou_tracker":
            # Experimental: Can be integrated if needed, fallback to distance match for now.
            # But let's actually implement a basic IoU matching here
            from src.postprocess import _iou, _bbox
            for det_idx, det in vehicles:
                det_box = _bbox(det)
                for track_id, track in self._tracks.items():
                    if not getattr(track, 'last_box', None):
                        continue
                    iou = _iou(det_box, track.last_box)
                    if iou > 0.1:  # Simple threshold
                        # use 1 - iou as distance so lower is better matches the sorting later
                        candidates.append((1.0 - iou, det_idx, track_id))
        else:
            # Standard distance match
            for det_idx, det in vehicles:
                cx, cy = self._center(det)
                for track_id, track in self._tracks.items():
                    if not track.history:
                        continue
                    tx, ty = track.history[-1][:2]
                    dx = cx - tx
                    dy = cy - ty
                    dist = (dx * dx + dy * dy) ** 0.5
                    if dist <= Settings.MOVEMENT_MATCH_DISTANCE_PX:
                        candidates.append((dist, det_idx, track_id))

        used_dets = set()
        used_tracks = set()
        for _, det_idx, track_id in sorted(candidates, key=lambda item: item[0]):
            if det_idx in used_dets or track_id in used_tracks:
                continue
            assignments[det_idx] = track_id
            used_dets.add(det_idx)
            used_tracks.add(track_id)
        return assignments

    def _age_tracks(self, matched_track_ids: set) -> None:
        to_delete: List[int] = []
        for track_id, track in self._tracks.items():
            if track_id not in matched_track_ids:
                track.missed += 1
                if track.missed > Settings.MOVEMENT_MAX_MISSED_FRAMES:
                    to_delete.append(track_id)
        for track_id in to_delete:
            self._tracks[track_id].history.clear()
            del self._tracks[track_id]

    def _create_track(self, center: Tuple[float, float]) -> int:
        track_id = self._next_track_id
        self._next_track_id += 1
        track = _Track(history=deque(maxlen=Settings.MOVEMENT_WINDOW_FRAMES))
        cx, cy = center
        track.history.append((cx, cy, self._cam_total_x, self._cam_total_y))
        self._tracks[track_id] = track
        return track_id

    @staticmethod
    def _center(det: Dict) -> Tuple[float, float]:
        return (
            (float(det.get("top_left_x", 0)) + float(det.get("bottom_right_x", 0))) / 2.0,
            (float(det.get("top_left_y", 0)) + float(det.get("bottom_right_y", 0))) / 2.0,
        )

    @staticmethod
    def _prepare_flow_gray(gray: np.ndarray) -> Tuple[np.ndarray, float]:
        scale = float(getattr(Settings, "MOTION_COMP_DOWNSCALE", 1.0))
        if scale >= 1.0 or scale <= 0.1:
            return gray, 1.0

        h, w = gray.shape[:2]
        target_w = max(64, int(w * scale))
        target_h = max(64, int(h * scale))
        if target_w >= w or target_h >= h:
            return gray, 1.0

        resized = cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_AREA)
        inv_scale = w / float(target_w)
        return resized, inv_scale

    @staticmethod
    def _detect_features(gray: np.ndarray) -> Optional[np.ndarray]:
        return cv2.goodFeaturesToTrack(
            gray,
            maxCorners=Settings.MOTION_COMP_MAX_CORNERS,
            qualityLevel=Settings.MOTION_COMP_QUALITY_LEVEL,
            minDistance=Settings.MOTION_COMP_MIN_DISTANCE,
        )

    @staticmethod
    def _robust_median_shift(
        old: np.ndarray,
        new: np.ndarray,
    ) -> Tuple[float, float]:
        deltas = new - old
        dx = deltas[:, 0]
        dy = deltas[:, 1]

        low, high = 10, 90
        dx_l, dx_h = np.percentile(dx, [low, high])
        dy_l, dy_h = np.percentile(dy, [low, high])
        keep = (dx >= dx_l) & (dx <= dx_h) & (dy >= dy_l) & (dy <= dy_h)
        if np.count_nonzero(keep) >= 5:
            dx = dx[keep]
            dy = dy[keep]

        return float(np.median(dx)), float(np.median(dy))

    def _estimate_camera_shift(self, frame_ctx: "FrameContext") -> Tuple[float, float]:
        if isinstance(frame_ctx, np.ndarray):
            from src.utils import FrameContext
            frame_ctx = FrameContext(frame_ctx)
        gray, self._flow_inv_scale = self._prepare_flow_gray(frame_ctx.gray)

        if self._prev_gray is None:
            self._prev_gray = gray
            self._prev_points = self._detect_features(gray)
            self._frame_diff = float("inf")
            return 0.0, 0.0

        self._frame_diff = float(cv2.absdiff(self._prev_gray, gray).mean())

        if self._prev_points is None or len(self._prev_points) < Settings.MOTION_COMP_MIN_FEATURES:
            self._prev_points = self._detect_features(self._prev_gray)
            if self._prev_points is None or len(self._prev_points) < 5:
                self._prev_gray = gray
                self._prev_points = self._detect_features(gray)
                return 0.0, 0.0

        if self._prev_points is None:
            return 0.0, 0.0

        win = Settings.MOTION_COMP_WIN_SIZE
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray,
            gray,
            self._prev_points,
            None,
            winSize=(win, win),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if next_pts is None or status is None:
            self._prev_gray = gray
            self._prev_points = self._detect_features(gray)
            return 0.0, 0.0

        valid = status.flatten() == 1
        old = self._prev_points[valid].reshape(-1, 2)
        new = next_pts[valid].reshape(-1, 2)
        fb_max_error = float(getattr(Settings, "MOTION_COMP_FB_MAX_ERROR", 1.5))
        if len(new) >= 5 and fb_max_error > 0.0:
            back_pts, back_status, _ = cv2.calcOpticalFlowPyrLK(
                gray,
                self._prev_gray,
                new.reshape(-1, 1, 2),
                None,
                winSize=(win, win),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            )
            if back_pts is not None and back_status is not None:
                back_ok = back_status.flatten() == 1
                back = back_pts.reshape(-1, 2)
                fb_error = np.linalg.norm(back - old, axis=1)
                fb_keep = back_ok & (fb_error <= fb_max_error)
                if np.count_nonzero(fb_keep) >= 5:
                    old = old[fb_keep]
                    new = new[fb_keep]

        if len(new) < 5:
            self._prev_gray = gray
            self._prev_points = self._detect_features(gray)
            return 0.0, 0.0

        # Calculate shift based on chosen algorithm
        algo = getattr(Settings, "MOTION_ALGO", "flow").lower()

        if algo == "homography" and len(new) >= 10:
            # Use RANSAC Homography for robust global camera motion estimation
            H, mask = cv2.findHomography(old, new, cv2.RANSAC, 3.0)
            valid_h = (
                H is not None
                and mask is not None
                and int(np.count_nonzero(mask)) >= 6
                and np.all(np.isfinite(H))
            )
            if valid_h:
                # Extract pure translation from the 3x3 homography matrix
                cam_dx = float(H[0, 2])
                cam_dy = float(H[1, 2])
            else:
                cam_dx, cam_dy = self._robust_median_shift(old, new)
        else:
            # Standard median optical flow (fallback/default)
            cam_dx, cam_dy = self._robust_median_shift(old, new)

        cam_dx *= self._flow_inv_scale
        cam_dy *= self._flow_inv_scale
        max_shift = float(getattr(Settings, "MOTION_COMP_MAX_SHIFT_PX", 0.0))
        if max_shift > 0.0:
            cam_dx = max(-max_shift, min(max_shift, cam_dx))
            cam_dy = max(-max_shift, min(max_shift, cam_dy))
        if not np.isfinite(cam_dx):
            cam_dx = 0.0
        if not np.isfinite(cam_dy):
            cam_dy = 0.0

        self._prev_gray = gray
        self._prev_points = new.reshape(-1, 1, 2)
        if len(self._prev_points) < Settings.MOTION_COMP_MIN_FEATURES // 2:
            self._prev_points = self._detect_features(gray)
            if self._prev_points is None:
                self._prev_points = None  # Açık atama, sonraki frame L210'da yakalar

        return cam_dx, cam_dy
