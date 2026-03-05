"""Post-process guardrails: overlap resolution, scene consistency, crowd adaptivity.
Şartname Ref: Satır 117 (tüm taşıt/insan tespit edilmeli), Satır 147 (kısmi nesneler dahil).
Amaç: YOLO çıktısındaki mantıksız FP'leri (çatı=araba, direk=insan) baskılamak.

ASSUMPTION: Guardrail eşikleri deneysel olarak belirlenmiştir;
config ile kapatılabilir (GUARDRAILS_ENABLED=False).
"""
import logging
from typing import Dict, List, Tuple
import numpy as np
from config.settings import Settings
from src.utils import Logger

log = Logger("Postprocess")


# ─── Settings defaults (config/settings.py'de tanımlı olmazsa) ───────────────

def _cfg(attr: str, default):
    return getattr(Settings, attr, default)


# ─── IoU helper ──────────────────────────────────────────────────────────────

def _iou(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
    """Intersection-over-Union for two (x1, y1, x2, y2) boxes."""
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


def _area(det: Dict) -> float:
    w = float(det["bottom_right_x"]) - float(det["top_left_x"])
    h = float(det["bottom_right_y"]) - float(det["top_left_y"])
    return max(1.0, w * h)


def _bbox(det: Dict) -> Tuple[float, float, float, float]:
    return (
        float(det["top_left_x"]),
        float(det["top_left_y"]),
        float(det["bottom_right_x"]),
        float(det["bottom_right_y"]),
    )


# ─── Public API ──────────────────────────────────────────────────────────────

def apply_guardrails(detections: List[Dict]) -> Tuple[List[Dict], Dict[str, int]]:
    """Return (filtered detections, stats dict with elimination reasons).

    Stats keys: 'overlap_suppressed', 'scene_outlier', 'crowd_trimmed', 'total_input'.
    """
    if not _cfg("GUARDRAILS_ENABLED", True):
        return detections, {"total_input": len(detections)}

    stats: Dict[str, int] = {
        "total_input": len(detections),
        "overlap_suppressed": 0,
        "scene_outlier": 0,
        "crowd_trimmed": 0,
    }

    result = detections

    # 1. Overlap Resolution: dev taşıt bbox + normal insan bbox → büyük olan bastırılır
    result, n_overlap = _overlap_resolution(result)
    stats["overlap_suppressed"] = n_overlap

    # 2. Scene Consistency: aynı sınıf içinde outlier boyut → bastır
    result, n_outlier = _scene_consistency(result)
    stats["scene_outlier"] = n_outlier

    # 3. Crowd Adaptivity: çok fazla tespit → düşük conf olanları kes
    result, n_crowd = _crowd_adaptivity(result)
    stats["crowd_trimmed"] = n_crowd

    total_removed = n_overlap + n_outlier + n_crowd
    if total_removed > 0:
        log.debug(
            f"Guardrails: {total_removed} tespit elendi "
            f"(overlap={n_overlap}, outlier={n_outlier}, crowd={n_crowd})"
        )

    return result, stats


# ─── Internal rules ──────────────────────────────────────────────────────────

def _overlap_resolution(detections: List[Dict]) -> Tuple[List[Dict], int]:
    """İnsan bbox'u düzgünken dev Taşıt bbox'ı aynı bölgede → büyüğünü bastır.

    Şartname: Satır 149-150 - motosiklet sürücüsü Taşıt olmalı.
    Burada sadece MANTIK DIŞI boyut farkı olan çakışmayı çözeriz.
    """
    area_ratio_threshold = float(_cfg("GUARDRAIL_OVERLAP_AREA_RATIO", 5.0))
    iou_threshold = float(_cfg("GUARDRAIL_OVERLAP_IOU", 0.15))

    suppress_ids = set()
    n = len(detections)

    exempt = frozenset(str(c) for c in _cfg("GUARDRAIL_EXEMPT_CLASSES", ("2", "3")))
    for i in range(n):
        if i in suppress_ids:
            continue
        for j in range(i + 1, n):
            if j in suppress_ids:
                continue
            di, dj = detections[i], detections[j]
            if str(di.get("cls", "")) in exempt or str(dj.get("cls", "")) in exempt:
                continue
            bi, bj = _bbox(di), _bbox(dj)
            overlap = _iou(bi, bj)
            if overlap < iou_threshold:
                continue
            ai, aj = _area(di), _area(dj)
            ratio = max(ai, aj) / max(min(ai, aj), 1.0)
            if ratio >= area_ratio_threshold:
                if ai > aj:
                    suppress_ids.add(i)
                else:
                    suppress_ids.add(j)

    removed = len(suppress_ids)
    if removed:
        detections = [d for idx, d in enumerate(detections) if idx not in suppress_ids]
    return detections, removed


def _scene_consistency(detections: List[Dict]) -> Tuple[List[Dict], int]:
    """Aynı sınıf içinde median alanın N katını aşan tespit → outlier.

    Örnek: 4 araba ~2000px², biri 50000px² → outlier.
    """
    outlier_factor = float(_cfg("GUARDRAIL_SCENE_OUTLIER_FACTOR", 8.0))
    min_samples = int(_cfg("GUARDRAIL_SCENE_MIN_SAMPLES", 3))

    # Sınıf bazında alanları topla
    class_areas: Dict[str, List[Tuple[int, float]]] = {}
    for idx, det in enumerate(detections):
        cls = str(det.get("cls", ""))
        a = _area(det)
        class_areas.setdefault(cls, []).append((idx, a))

    exempt = frozenset(str(c) for c in _cfg("GUARDRAIL_EXEMPT_CLASSES", ("2", "3")))
    suppress_ids = set()
    for cls, items in class_areas.items():
        if cls in exempt:
            continue
        if len(items) < min_samples:
            continue
        areas = [a for _, a in items]
        median_area = float(np.median(areas))
        if median_area < 1.0:
            continue
        threshold = median_area * outlier_factor
        for idx, a in items:
            if a > threshold:
                suppress_ids.add(idx)

    removed = len(suppress_ids)
    if removed:
        detections = [d for idx, d in enumerate(detections) if idx not in suppress_ids]
    return detections, removed


def _crowd_adaptivity(detections: List[Dict]) -> Tuple[List[Dict], int]:
    """Tespit sayısı çok fazlaysa düşük conf olanları kes.

    Şartname max limit: RESULT_MAX_OBJECTS = 100 (per frame).
    Bunun altında bile olsa 30+ taşıt → gürültü olabilir.
    """
    crowd_threshold = int(_cfg("GUARDRAIL_CROWD_THRESHOLD", 30))
    crowd_conf_boost = float(_cfg("GUARDRAIL_CROWD_CONF_BOOST", 0.15))

    if len(detections) <= crowd_threshold:
        return detections, 0

    exempt = frozenset(str(c) for c in _cfg("GUARDRAIL_EXEMPT_CLASSES", ("2", "3")))
    base_conf = float(_cfg("CONFIDENCE_THRESHOLD", 0.40))
    elevated_conf = base_conf + crowd_conf_boost

    before = len(detections)
    detections = [
        d for d in detections
        if str(d.get("cls", "")) in exempt
        or float(d.get("confidence", 1.0)) >= elevated_conf
    ]
    removed = before - len(detections)
    return detections, removed
