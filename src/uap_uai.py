"""UAP and UAI detection validation and landing status rules.
Şartname Ref: 
- Satır 167: Boşsa = 1
- Satır 168-169: Nesne varsa = 0
- Satır 182-183: Tamamı kare içinde değilse = 0
- Satır 185-187: Perspektif yanılsaması (genişletilmiş kutu) ile kesişim = 0
"""

from typing import Dict, List, Tuple
import cv2
import numpy as np
from config.settings import Settings

_UAP_CLASS = "2"
_UAI_CLASS = "3"

def _bbox(det: Dict) -> Tuple[float, float, float, float]:
    return (
        float(det["top_left_x"]),
        float(det["top_left_y"]),
        float(det["bottom_right_x"]),
        float(det["bottom_right_y"]),
    )

def _intersection_area(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
    inter_x1 = max(box_a[0], box_b[0])
    inter_y1 = max(box_a[1], box_b[1])
    inter_x2 = min(box_a[2], box_b[2])
    inter_y2 = min(box_a[3], box_b[3])
    w = max(0.0, inter_x2 - inter_x1)
    h = max(0.0, inter_y2 - inter_y1)
    return w * h

def _expand_bbox(box: Tuple[float, float, float, float], margin_ratio: float) -> Tuple[float, float, float, float]:
    """Perspektif toleransı: UAP/UAİ alanını dışa doğru genişlet"""
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    dx = w * margin_ratio
    dy = h * margin_ratio
    return (x1 - dx, y1 - dy, x2 + dx, y2 + dy)

def determine_landing_status(
    detections: List[Dict],
    frame_w: int,
    frame_h: int,
    frame_rgb: np.ndarray = None
) -> List[Dict]:
    """Şartname kurallarına göre UAP/UAİ iniş uygunluğunu hesaplar."""
    landing_zones: List[Tuple[int, Dict]] = []
    obstacles: List[Tuple[float, float, float, float]] = []

    for idx, det in enumerate(detections):
        cls = str(det.get("cls", ""))
        det["landing_status"] = "-1"  # Default (Taşıt/İnsan)
        if cls in (_UAP_CLASS, _UAI_CLASS):
            landing_zones.append((idx, det))
        elif cls in ("0", "1"):
            # Engel
            obstacles.append(_bbox(det))

    if not landing_zones:
        return detections

    edge_px_w = int(frame_w * getattr(Settings, "EDGE_MARGIN_RATIO", 0.005))
    edge_px_h = int(frame_h * getattr(Settings, "EDGE_MARGIN_RATIO", 0.005))
    proximity_margin = getattr(Settings, "LANDING_PROXIMITY_MARGIN", 0.15)
    do_cv_check = getattr(Settings, "UAP_CV_VERIFICATION", False)

    for idx, det in landing_zones:
        box = _bbox(det)
        x1, y1, x2, y2 = box

        # 1. Edge Check (Kısmi görünürlük landing=0)
        # Şartname 182-183: UAP/UAİ alanının TAMAMI kare içinde bulunmalıdır.
        if (
            x1 <= edge_px_w or
            y1 <= edge_px_h or
            x2 >= frame_w - edge_px_w or
            y2 >= frame_h - edge_px_h
        ):
            det["landing_status"] = "0"
            continue

        # 2. Obstacle / Perspective Interference Check
        # Şartname 185-187: Çekim açısına bağlı olarak alana yakın cisimler üstünde gibi görülebilir
        expanded_box = _expand_bbox(box, proximity_margin)
        is_clear = True
        
        for obs_box in obstacles:
            inter = _intersection_area(expanded_box, obs_box)
            if inter > 0:
                is_clear = False
                break

        # Sadece UAP/UAİ var ama iç içe girmişse
        if is_clear:
            for other_idx, other_det in landing_zones:
                if idx == other_idx:
                    continue
                other_box = _bbox(other_det)
                if _intersection_area(expanded_box, other_box) > 0:
                    is_clear = False
                    break

        if not is_clear:
            det["landing_status"] = "0"
            continue

        # 3. Shape Validation (Opsiyonel)
        if do_cv_check and frame_rgb is not None:
            # Hough circles directly on the cropped RGB (gray)
            ix1, iy1 = max(0, int(x1)), max(0, int(y1))
            ix2, iy2 = min(frame_w, int(x2)), min(frame_h, int(y2))
            crop = frame_rgb[iy1:iy2, ix1:ix2]
            if crop.size > 0:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                circles = cv2.HoughCircles(
                    blurred, cv2.HOUGH_GRADIENT, 1, 20,
                    param1=50, param2=30, 
                    minRadius=int(crop.shape[0]*0.2), 
                    maxRadius=int(crop.shape[0]*0.8)
                )
                if circles is None:
                    # Klasik CV "daire değil" diyor
                    det["landing_status"] = "-1" # Şartname 170: İniş alanı niteliği taşımıyorsa -1
                    continue

        # Hepsi geçildi → İnişe Uygun
        det["landing_status"] = "1"

    return detections
