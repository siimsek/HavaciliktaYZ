"""Logger (seviyeli log), Visualizer (bbox çizimi), log_json_to_disk (gelen/giden JSON kayıt)."""

import os
import json
import re
import subprocess
import sys
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple

import cv2
import numpy as np

try:
    from colorama import init as colorama_init, Fore, Style
    colorama_init(autoreset=True)
    _HAS_COLORAMA = True
except ImportError:
    _HAS_COLORAMA = False

from config.settings import Settings


# ─── GPS Health (gps_health.py birleşik) ────────────────────────────────────
GPS_HEALTH_UNKNOWN = "unknown"
GPS_HEALTH_HEALTHY = "1"
GPS_HEALTH_UNHEALTHY = "0"


def normalize_gps_health(
    gps_health: Any,
    gps_health_status: Any = None,
) -> Tuple[Optional[int], str]:
    """Return (value, state) where value is 0/1/None and state is '0'/'1'/'unknown'."""
    raw = gps_health if gps_health is not None else gps_health_status
    if raw is None:
        return None, GPS_HEALTH_UNKNOWN
    text = str(raw).strip().lower()
    if text in {"", "unknown", "none", "null", "nan"}:
        return None, GPS_HEALTH_UNKNOWN
    try:
        parsed = int(float(raw))
    except (TypeError, ValueError):
        return None, GPS_HEALTH_UNKNOWN
    if parsed == 1:
        return 1, GPS_HEALTH_HEALTHY
    if parsed == 0:
        return 0, GPS_HEALTH_UNHEALTHY
    return None, GPS_HEALTH_UNKNOWN


# ─── FrameContext (frame_context.py birleşik) ────────────────────────────────
class FrameContext:
    """Frame için ortak hesaplamalar (gray conversion). Detection, movement, localization tekrar hesaplamasın."""

    def __init__(self, frame: np.ndarray) -> None:
        self.frame = frame
        self._gray: Optional[np.ndarray] = None

    @property
    def gray(self) -> np.ndarray:
        if self._gray is None:
            self._gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        return self._gray


# ─── Display / Logger ──────────────────────────────────────────────────────
def get_display_size() -> Tuple[int, int]:
    """Ekran (primary display) çözünürlüğünü döndür. 4K/ekrana sığdırma için kullanılır."""
    try:
        if sys.platform == "win32":
            import ctypes
            user32 = ctypes.windll.user32  # type: ignore[union-attr]
            w = user32.GetSystemMetrics(0)  # SM_CXSCREEN
            h = user32.GetSystemMetrics(1)  # SM_CYSCREEN
            if w > 0 and h > 0:
                return (w, h)
        elif sys.platform.startswith("linux"):
            result = subprocess.run(
                ["xrandr", "--query"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0 and result.stdout:
                for line in result.stdout.splitlines():
                    if " connected" in line.lower() and "*" in line:
                        parts = line.split()
                        for i, p in enumerate(parts):
                            if p == "primary" and i + 1 < len(parts):
                                res = parts[i + 1]
                                if "x" in res:
                                    wh = res.split("x")
                                    if len(wh) == 2:
                                        return (int(wh[0]), int(wh[1].split("+")[0]))
                            if "x" in p and "+" in p:
                                wh = p.split("x")
                                if len(wh) == 2:
                                    return (int(wh[0]), int(wh[1].split("+")[0]))
    except Exception:
        pass
    return (1920, 1080)


class Logger:
    """Seviyeli log (DEBUG, INFO, WARN, ERROR, SUCCESS)."""

    def __init__(self, module_name: str) -> None:
        self.module_name = module_name

    def _timestamp(self) -> str:
        now = datetime.now()
        return now.strftime("%H:%M:%S.") + f"{now.microsecond // 1000:03d}"

    def _print(self, level: str, color: str, message: str) -> None:
        ts = self._timestamp()
        prefix = f"[{ts}] [{level:^7}] [{self.module_name}]"
        if _HAS_COLORAMA:
            print(f"{color}{prefix}{Style.RESET_ALL} {message}")
        else:
            print(f"{prefix} {message}")

    def debug(self, message: str) -> None:
        if Settings.DEBUG:
            color = Fore.WHITE if _HAS_COLORAMA else ""
            self._print("DEBUG", color, message)

    def info(self, message: str) -> None:
        color = Fore.GREEN if _HAS_COLORAMA else ""
        self._print("INFO", color, message)

    def warn(self, message: str) -> None:
        color = Fore.YELLOW if _HAS_COLORAMA else ""
        self._print("WARN", color, message)

    def error(self, message: str) -> None:
        color = Fore.RED if _HAS_COLORAMA else ""
        self._print("ERROR", color, message)

    def success(self, message: str) -> None:
        color = Fore.CYAN if _HAS_COLORAMA else ""
        self._print("SUCCESS", color, message)


class Visualizer:
    """Debug: bbox, etiket, iniş durumu çizimi."""

    CLASS_COLORS: Dict[int, tuple] = {
        0: (0, 255, 0),      # Taşıt → Yeşil
        1: (255, 0, 0),      # İnsan → Mavi
        2: (255, 255, 0),    # UAP → Cyan
        3: (0, 0, 255),      # UAİ → Kırmızı
    }

    CLASS_NAMES: Dict[int, str] = {
        0: "Tasit",
        1: "Insan",
        2: "UAP",
        3: "UAI",
    }

    LANDING_LABELS: Dict[str, str] = {
        "-1": "",
        "0": " [UYGUN DEGIL]",
        "1": " [UYGUN]",
    }

    MOTION_LABELS: Dict[str, str] = {
        "1": "Hareketli",
        "0": "Sabit",
        "-1": "",
    }

    MOTION_COLORS: Dict[str, tuple] = {
        "1": (0, 220, 0),      # Hareketli → Yeşil
        "0": (0, 140, 255),    # Sabit → Turuncu
        "-1": (180, 180, 180), # N/A → Gri
    }

    def __init__(self) -> None:
        os.makedirs(Settings.DEBUG_OUTPUT_DIR, exist_ok=True)
        self.log = Logger("Visualizer")
        self._save_counter = 0
        from collections import deque
        max_traj = int(getattr(Settings, "MAP_MAX_TRAJECTORY_LENGTH", 500))
        self._trajectory = deque(maxlen=max_traj)
        # Sabit harita ölçeği — ilk GPS noktalarından hesaplanıp kilitlenir
        self._map_span: Optional[float] = None

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Dict],
        frame_id: str = "unknown",
        position: Optional[Dict] = None,
        save_to_disk: bool = True,
        guardrail_stats: Optional[Dict[str, int]] = None,
        gps_health: Optional[int] = None,
    ) -> np.ndarray:
        annotated = frame.copy()

        # Debug overlay: Guardrail istatistikleri (GR: O=overlap S=scene C=crowd)
        if guardrail_stats and any(v > 0 for v in guardrail_stats.values()):
            o = guardrail_stats.get("overlap_suppressed", 0)
            s = guardrail_stats.get("scene_outlier", 0)
            c = guardrail_stats.get("crowd_trimmed", 0)
            gr_text = f"GR: O={o} S={s} C={c}"
            h, w = annotated.shape[:2]
            scale = min(w / 1920.0, h / 1080.0)
            font_scale = max(0.35, 0.5 * scale)
            thickness = max(1, int(1.5 * scale))
            ts_size, _ = cv2.getTextSize(gr_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            cv2.rectangle(
                annotated,
                (w - ts_size[0] - 12, 10),
                (w - 8, 10 + ts_size[1] + 8),
                (30, 30, 30),
                -1,
            )
            cv2.putText(
                annotated,
                gr_text,
                (w - ts_size[0] - 8, 10 + ts_size[1] + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 255, 200),
                thickness,
            )

        for det in detections:
            cls_id = int(det.get("cls", -1))
            landing = str(det.get("landing_status", "-1"))
            motion = str(det.get("motion_status", "-1"))
            x1 = int(float(det.get("top_left_x", 0)))
            y1 = int(float(det.get("top_left_y", 0)))
            x2 = int(float(det.get("bottom_right_x", 0)))
            y2 = int(float(det.get("bottom_right_y", 0)))
            conf = det.get("confidence", 0.0)

            color = self.CLASS_COLORS.get(cls_id, (200, 200, 200))
            label_name = self.CLASS_NAMES.get(cls_id, "?")
            landing_txt = self.LANDING_LABELS.get(landing, "")
            motion_txt = self.MOTION_LABELS.get(motion, "")

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Line 1: Class name + confidence
            main_label = f"{label_name} {conf:.2f}"
            label_size, baseline = cv2.getTextSize(main_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                annotated,
                (x1, y1 - label_size[1] - 6),
                (x1 + label_size[0], y1),
                color,
                -1,
            )
            cv2.putText(
                annotated, main_label, (x1, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
            )

            # Line 2 (below bbox): Şartname uyumlu durum bilgisi
            # Taşıt(0): sadece hareket durumu  |  İnsan(1): ek bilgi yok
            # UAP(2)/UAİ(3): sadece iniş durumu
            info_text = ""
            info_bg_color = (100, 100, 100)

            if cls_id == 0:  # Taşıt → sadece hareket durumu
                if motion_txt:
                    info_text = motion_txt
                    if motion == "1":
                        info_bg_color = (0, 160, 0)    # Yeşil: hareketli
                    elif motion == "0":
                        info_bg_color = (0, 100, 200)  # Turuncu: sabit
            elif cls_id in (2, 3):  # UAP/UAİ → sadece iniş durumu
                if landing_txt:
                    info_text = landing_txt.strip()
                    if landing == "0":
                        info_bg_color = (0, 0, 180)    # Kırmızı: uygun değil
                    elif landing == "1":
                        info_bg_color = (0, 140, 0)    # Yeşil: uygun
            # İnsan(1): ek bilgi yok, info_text boş kalır

            if info_text:
                info_size, _ = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                info_y = y2 + info_size[1] + 6
                cv2.rectangle(
                    annotated,
                    (x1, y2),
                    (x1 + info_size[0] + 4, info_y),
                    info_bg_color,
                    -1,
                )
                cv2.putText(
                    annotated, info_text, (x1 + 2, info_y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
                )

        if position:
            x_m = position.get("x", 0.0)
            y_m = position.get("y", 0.0)
            z_m = position.get("z", 0.0)
            # GPS=1: sunucu verisi; GPS=0: görsel odometri (ölçüm yoksa pozisyon dondurulur)
            if gps_health is None or gps_health == 1 or gps_health == 0:
                self._trajectory.append((x_m, y_m))

            h, w = annotated.shape[:2]
            
            # Use responsive sizes
            scale = min(w / 1920.0, h / 1080.0)
            map_size = int(280 * scale)
            padding = int(20 * scale)
            font_scale = 0.5 * scale

            map_x1 = w - map_size - padding
            map_y1 = h - map_size - padding

            # Yarı saydam arka plan (overlay)
            overlay = annotated.copy()
            cv2.rectangle(overlay, (map_x1, map_y1), (map_x1 + map_size, map_y1 + map_size), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)
            cv2.rectangle(annotated, (map_x1, map_y1), (map_x1 + map_size, map_y1 + map_size), (0, 255, 120), 2)

            pos_text = f"X:{x_m:.1f} Y:{y_m:.1f} Z:{z_m:.1f}m"
            cv2.putText(
                annotated, pos_text, (map_x1 + 5, map_y1 - int(10*scale)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), max(1, int(1.5*scale)),
            )

            # Grid çizgileri (4x4 ızgara)
            grid_color = (100, 100, 100)
            grid_thickness = max(1, int(1 * scale))
            for i in range(1, 5):
                ix = map_x1 + int(map_size * i / 5)
                iy = map_y1 + int(map_size * i / 5)
                cv2.line(annotated, (ix, map_y1), (ix, map_y1 + map_size), grid_color, grid_thickness)
                cv2.line(annotated, (map_x1, iy), (map_x1 + map_size, iy), grid_color, grid_thickness)

            if len(self._trajectory) >= 1:
                xs = [p[0] for p in self._trajectory]
                ys = [p[1] for p in self._trajectory]
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)

                rng_x = max(max_x - min_x, 15.0)
                rng_y = max(max_y - min_y, 15.0)
                cx, cy = (min_x + max_x) / 2.0, (min_y + max_y) / 2.0

                # Sabit ölçek: ilk hesaplamada kilitlenir, titreme azaltılır
                computed_span = max(rng_x, rng_y) * 1.2 if len(self._trajectory) > 1 else 50.0
                computed_span = max(computed_span, 30.0)
                if self._map_span is None:
                    self._map_span = max(computed_span, 50.0)
                # Trajectory sınırları dışına taşarsa span artır (zoom out)
                elif computed_span > self._map_span:
                    self._map_span = computed_span
                span = self._map_span

                pts = []
                for px, py in self._trajectory:
                    # Invert Y for drawing assuming standard map coordinates vs image coordinates
                    nx = (px - (cx - span / 2.0)) / span
                    ny = 1.0 - ((py - (cy - span / 2.0)) / span)
                    sx = int(map_x1 + nx * map_size)
                    sy = int(map_y1 + ny * map_size)
                    # Clip to map boundaries
                    sx = np.clip(sx, map_x1, map_x1 + map_size)
                    sy = np.clip(sy, map_y1, map_y1 + map_size)
                    pts.append((sx, sy))

                if len(pts) > 1:
                    cv2.polylines(annotated, [np.array(pts)], False, (0, 200, 255), max(1, int(2*scale)), cv2.LINE_AA)
                cv2.circle(annotated, pts[-1], max(3, int(5*scale)), (0, 0, 255), -1)

        if save_to_disk:
            self._save_counter += 1
            if self._save_counter % Settings.DEBUG_SAVE_INTERVAL == 0:
                save_path = os.path.join(Settings.DEBUG_OUTPUT_DIR, f"{frame_id}.jpg")
                cv2.imwrite(save_path, annotated)
                self.log.debug(f"Debug görsel kaydedildi: {save_path}")

        return annotated


def log_json_to_disk(
    data: Any,
    direction: str = "outgoing",
    tag: str = "general",
) -> None:
    try:
        os.makedirs(Settings.LOG_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_direction = _sanitize_log_component(direction)
        safe_tag = _sanitize_log_component(tag)
        filename = f"{timestamp}_{safe_direction}_{safe_tag}.json"
        filepath = os.path.join(Settings.LOG_DIR, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        _prune_old_logs(Settings.LOG_DIR)

    except Exception as exc:
        Logger("Logger").warn(f"JSON log write failed: {exc}")


def _sanitize_log_component(value: Any) -> str:
    text = str(value)
    sanitized = re.sub(r"[^A-Za-z0-9._-]", "_", text)
    return sanitized[:80] if sanitized else "general"


def _prune_old_logs(log_dir: str) -> None:
    max_files = max(1, int(Settings.LOG_MAX_FILES))
    try:
        files = [
            os.path.join(log_dir, name)
            for name in os.listdir(log_dir)
            if name.lower().endswith(".json")
        ]
    except Exception:
        return

    if len(files) <= max_files:
        return

    files.sort(key=lambda path: os.path.getmtime(path))
    for old_file in files[: len(files) - max_files]:
        try:
            os.remove(old_file)
        except Exception:
            continue
