"""
Taşıt hareketlilik (movement_status) kestirimi için hafif takip modülü.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

from config.settings import Settings


@dataclass
class _Track:
    history: Deque[Tuple[float, float]] = field(default_factory=deque)
    missed: int = 0


class MovementEstimator:
    """
    Basit merkez-nokta takibi ile taşıtların hareket durumunu belirler.

    movement_status:
        - "1": Hareketli
        - "0": Hareketsiz
        - "-1": Taşıt değil
    """

    def __init__(self) -> None:
        self._tracks: Dict[int, _Track] = {}
        self._next_track_id: int = 1

    def annotate(self, detections: List[Dict]) -> List[Dict]:
        vehicles: List[Tuple[int, Dict]] = []
        for idx, det in enumerate(detections):
            if det.get("cls") == "0":
                vehicles.append((idx, det))
            else:
                det["movement_status"] = "-1"

        if not vehicles:
            self._age_tracks(set())
            return detections

        centers = {idx: self._center(det) for idx, det in vehicles}
        assignments = self._match(centers)
        matched_track_ids = set(assignments.values())
        self._age_tracks(matched_track_ids)

        for idx, det in vehicles:
            track_id = assignments.get(idx)
            if track_id is None:
                track_id = self._create_track(centers[idx])
            track = self._tracks[track_id]
            track.history.append(centers[idx])
            track.missed = 0
            det["movement_status"] = self._status(track.history)

        return detections

    def _status(self, history: Deque[Tuple[float, float]]) -> str:
        if len(history) < Settings.MOVEMENT_MIN_HISTORY:
            return "0"

        x0, y0 = history[0]
        x1, y1 = history[-1]
        dx = x1 - x0
        dy = y1 - y0
        dist = (dx * dx + dy * dy) ** 0.5
        return "1" if dist >= Settings.MOVEMENT_THRESHOLD_PX else "0"

    def _match(self, centers: Dict[int, Tuple[float, float]]) -> Dict[int, int]:
        assignments: Dict[int, int] = {}
        if not self._tracks:
            return assignments

        candidates: List[Tuple[float, int, int]] = []
        for det_idx, (cx, cy) in centers.items():
            for track_id, track in self._tracks.items():
                if not track.history:
                    continue
                tx, ty = track.history[-1]
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
            self._tracks[track_id].history.clear()  # Belleği temizle
            del self._tracks[track_id]

    def _create_track(self, center: Tuple[float, float]) -> int:
        track_id = self._next_track_id
        self._next_track_id += 1
        track = _Track(history=deque(maxlen=Settings.MOVEMENT_WINDOW_FRAMES))
        track.history.append(center)
        self._tracks[track_id] = track
        return track_id

    @staticmethod
    def _center(det: Dict) -> Tuple[float, float]:
        return (
            (float(det.get("top_left_x", 0)) + float(det.get("bottom_right_x", 0))) / 2.0,
            (float(det.get("top_left_y", 0)) + float(det.get("bottom_right_y", 0))) / 2.0,
        )
