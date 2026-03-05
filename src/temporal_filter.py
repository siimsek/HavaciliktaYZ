"""Zamansal tutarlılık filtresi: anlık FP (1–2 kare görünüp kaybolan) bastırma.
Gerçek nesnelere odaklanmak için son N karede en az K kez görünen tespitleri kabul eder."""

from collections import deque
from typing import Deque, Dict, List, Tuple

import numpy as np

from config.settings import Settings
from src.utils import Logger

log = Logger("TemporalFilter")


def _bbox(det: Dict) -> Tuple[float, float, float, float]:
    return (
        float(det.get("top_left_x", 0)),
        float(det.get("top_left_y", 0)),
        float(det.get("bottom_right_x", 0)),
        float(det.get("bottom_right_y", 0)),
    )


def _iou(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / (area_a + area_b - inter)


class TemporalConsistencyFilter:
    """Anlık yanlış tespitleri bastırmak için zamansal tutarlılık filtresi."""

    def __init__(self) -> None:
        self._history: Deque[List[Dict]] = deque(
            maxlen=max(2, int(getattr(Settings, "TEMPORAL_FILTER_WINDOW_FRAMES", 5)))
        )
        self._min_appearances = max(1, int(getattr(Settings, "TEMPORAL_FILTER_MIN_APPEARANCES", 2)))
        self._iou_threshold = float(getattr(Settings, "TEMPORAL_FILTER_IOU_THRESHOLD", 0.3))
        self._conf_exempt = float(getattr(Settings, "TEMPORAL_FILTER_CONFIDENCE_EXEMPT", 0.7))
        self._exempt_classes = frozenset(
            str(c) for c in getattr(Settings, "TEMPORAL_FILTER_EXEMPT_CLASSES", ("2", "3"))
        )
        self._suppressed_count = 0

    def filter(self, detections: List[Dict]) -> List[Dict]:
        if not getattr(Settings, "TEMPORAL_FILTER_ENABLED", True):
            self._history.append(detections)
            return detections

        if not detections:
            self._history.append([])
            return []

        result: List[Dict] = []
        for det in detections:
            if self._is_exempt(det):
                result.append(det)
                continue

            cls_key = str(det.get("cls", det.get("cls_int", "")))
            matches = self._count_matches(det, cls_key)
            if matches >= self._min_appearances - 1:
                result.append(det)
            else:
                self._suppressed_count += 1

        self._history.append(list(detections))
        return result

    def _is_exempt(self, det: Dict) -> bool:
        cls_key = str(det.get("cls", det.get("cls_int", "")))
        if cls_key in self._exempt_classes:
            return True
        conf = float(det.get("confidence", det.get("_confidence", 0.0)))
        return conf >= self._conf_exempt

    def _count_matches(self, det: Dict, cls_key: str) -> int:
        box = _bbox(det)
        count = 0
        for frame_dets in self._history:
            for prev in frame_dets:
                if str(prev.get("cls", prev.get("cls_int", ""))) != cls_key:
                    continue
                prev_box = _bbox(prev)
                if _iou(box, prev_box) >= self._iou_threshold:
                    count += 1
                    break
        return count

    def get_stats(self) -> Dict[str, int]:
        return {"temporal_suppressed": self._suppressed_count}

    def reset(self) -> None:
        self._history.clear()
        self._suppressed_count = 0
