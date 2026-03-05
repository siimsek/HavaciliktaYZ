"""Hibrit konum kestirimi: GPS=1 ise sunucu verisi, GPS=0 ise Lucas-Kanade optik akış.
Piksel kayması focal_length ve irtifa ile metreye çevrilir."""

from typing import Dict, Optional, Tuple, TYPE_CHECKING
import time

if TYPE_CHECKING:
    from src.utils import FrameContext

import cv2
import numpy as np

from config.settings import Settings
from src.utils import normalize_gps_health
from src.utils import Logger


class LatencyCompensator:
    """GPS=0 senaryosu için hız kestirimi + ileri projeksiyon yardımcı sınıfı."""

    def __init__(self, ema_alpha: float) -> None:
        self._ema_alpha = self._clamp_alpha(ema_alpha)
        self._velocity: Dict[str, float] = {"x": 0.0, "y": 0.0, "z": 0.0}
        self._last_position: Optional[Dict[str, float]] = None
        self._last_sample_monotonic: Optional[float] = None

    @staticmethod
    def _safe_float(value, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @classmethod
    def _clamp_alpha(cls, alpha: float) -> float:
        alpha_safe = cls._safe_float(alpha, default=0.0)
        return max(0.0, min(1.0, alpha_safe))

    def reset(self) -> None:
        self._velocity = {"x": 0.0, "y": 0.0, "z": 0.0}
        self._last_position = None
        self._last_sample_monotonic = None

    def update_velocity(
        self,
        position: Dict[str, float],
        sample_monotonic: Optional[float] = None,
    ) -> None:
        sample_t = (
            time.monotonic()
            if sample_monotonic is None
            else self._safe_float(sample_monotonic, default=0.0)
        )
        current_pos = {
            "x": self._safe_float(position.get("x", 0.0)),
            "y": self._safe_float(position.get("y", 0.0)),
            "z": self._safe_float(position.get("z", 0.0)),
        }

        if self._last_position is not None and self._last_sample_monotonic is not None:
            dt = sample_t - self._last_sample_monotonic
            if dt > 1e-6:
                alpha = self._ema_alpha
                for axis in ("x", "y", "z"):
                    raw_v = (current_pos[axis] - self._last_position[axis]) / dt
                    self._velocity[axis] = alpha * raw_v + (1.0 - alpha) * self._velocity[axis]

        self._last_position = current_pos
        self._last_sample_monotonic = sample_t

    def get_velocity(self) -> Dict[str, float]:
        return {
            "x": self._velocity["x"],
            "y": self._velocity["y"],
            "z": self._velocity["z"],
        }

    def project_position(
        self,
        position: Dict[str, float],
        dt_sec: float,
        max_dt_sec: float,
        max_delta_m: float,
    ) -> Tuple[Dict[str, float], float, float]:
        base_pos = {
            "x": self._safe_float(position.get("x", 0.0)),
            "y": self._safe_float(position.get("y", 0.0)),
            "z": self._safe_float(position.get("z", 0.0)),
        }

        dt = max(0.0, self._safe_float(dt_sec, default=0.0))
        max_dt = max(0.0, self._safe_float(max_dt_sec, default=0.0))
        if max_dt > 0.0:
            dt = min(dt, max_dt)
        else:
            dt = 0.0

        dx = self._velocity["x"] * dt
        dy = self._velocity["y"] * dt
        dz = self._velocity["z"] * dt

        delta_norm = float(np.sqrt(dx * dx + dy * dy + dz * dz))
        max_delta = max(0.0, self._safe_float(max_delta_m, default=0.0))
        if max_delta <= 0.0:
            dx = 0.0
            dy = 0.0
            dz = 0.0
            delta_norm = 0.0
        elif delta_norm > max_delta and delta_norm > 1e-9:
            scale = max_delta / delta_norm
            dx *= scale
            dy *= scale
            dz *= scale
            delta_norm = max_delta

        projected = {
            "x": base_pos["x"] + dx,
            "y": base_pos["y"] + dy,
            "z": base_pos["z"] + dz,
        }
        return projected, delta_norm, dt


class VisualOdometry:
    """GPS + optik akış hibrit pozisyon kestirimi."""

    def __init__(self) -> None:
        self.log = Logger("Localization")
        self._latency_comp = LatencyCompensator(Settings.LATENCY_COMP_EMA_ALPHA)

        self.position: Dict[str, float] = {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
        }

        self._last_gps_position: Optional[Dict[str, float]] = None

        self._last_gps_altitude: float = Settings.DEFAULT_ALTITUDE

        self._ema_alpha: float = 0.4
        self._ema_dx: float = 0.0
        self._ema_dy: float = 0.0

        self._max_displacement_per_frame: float = max(
            0.1,
            float(getattr(Settings, "VO_MAX_DISPLACEMENT_PER_FRAME", 1.0)),
        )
        self._gps_reanchor_alpha: float = max(
            0.0, min(1.0, float(Settings.GPS_REANCHOR_ALPHA))
        )
        self._gps_reanchor_max_delta_m: float = max(
            0.0, float(Settings.GPS_REANCHOR_MAX_DELTA_M)
        )

        self._was_gps_healthy: bool = False
        self._mode: str = "GPS_FUSED"
        self._last_update_monotonic: Optional[float] = None
        self._runtime_meta: Dict[str, object] = {
            "update_mode": "init",
            "state_source": "none",
            "quality_flag": "unknown",
            "reason_code": "startup",
        }

        self._last_of_position: Dict[str, float] = {"x": 0.0, "y": 0.0, "z": 0.0}

        self._prev_gray: Optional[np.ndarray] = None
        self._prev_points: Optional[np.ndarray] = None
        self._initial_point_count: int = 0

        self._feature_params = dict(
            maxCorners=100,
            qualityLevel=0.01,
            minDistance=40,
            blockSize=7,
        )
        self._lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                30,
                0.01,
            ),
        )

        self.log.info("Visual Odometry başlatıldı — Başlangıç: (0, 0, 0)")
        if Settings.FOCAL_LENGTH_PX == 800.0:
            self.log.warn(
                "FOCAL_LENGTH_PX=800 (varsayılan) kullanılıyor. "
                "TBD-010 kamera parametreleri yayımlandığında config/settings.py güncellenmeli."
            )

    def update(
        self,
        frame_ctx: "FrameContext",
        server_data: Dict,
    ) -> Dict[str, float]:
        gps_health, _ = normalize_gps_health(
            server_data.get("gps_health"),
            gps_health_status=server_data.get("gps_health_status"),
        )
        now_mono = time.monotonic()
        if self._last_update_monotonic is None:
            self._last_update_monotonic = now_mono

        if isinstance(frame_ctx, np.ndarray):
            gray = cv2.cvtColor(frame_ctx, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame_ctx.gray

        if gps_health == 1:
            # GPS sağlıklı: sunucu verisini kullan, gri kareyi referans için sakla
            self._update_from_gps(
                server_data,
                soft_reanchor=not self._was_gps_healthy,
            )
            self._prev_gray = gray
            self._was_gps_healthy = True
            self._mode = "GPS_FUSED"
            self._runtime_meta = {
                "update_mode": "corrected",
                "state_source": "gps",
                "quality_flag": "nominal",
                "reason_code": "gps_healthy",
            }

        else:
            # GPS kapalı: optik akış. İlk geçişte referans kare oluştur
            if self._was_gps_healthy:
                self._update_reference_frame(
                    self._prev_gray if self._prev_gray is not None else gray
                )
                self._was_gps_healthy = False
                self._ema_dx = 0.0
                self._ema_dy = 0.0

                self.log.info("GPS → Optik Akış geçişi — referans kare oluşturuldu, EMA resetlendi.")

            if self._prev_gray is not None and self._prev_points is not None:
                updated = self._update_from_optical_flow(gray, server_data)
                if updated:
                    self._mode = "VISION_ONLY"
                    self._runtime_meta = {
                        "update_mode": "corrected",
                        "state_source": "optical_flow",
                        "quality_flag": "nominal",
                        "reason_code": "gps_unhealthy_optical_flow",
                    }
                else:
                    self.predict_without_measurement(
                        reason_code="optical_flow_unavailable",
                        gps_health=0 if gps_health == 0 else -1,
                    )
            else:
                self.log.warn(
                    "GPS mevcut değil ve henüz referans kare oluşmadı — "
                    "GPS yok, referans kare henüz oluşmadı — pozisyon (0,0,0) korunuyor."
                )
                self._update_reference_frame(gray)
                self.predict_without_measurement(
                    reason_code="missing_reference_frame",
                    gps_health=0 if gps_health == 0 else -1,
                )

        self._latency_comp.update_velocity(self.position)
        self._last_update_monotonic = now_mono
        return self.get_position()

    def _update_from_gps(self, server_data: Dict, soft_reanchor: bool = False) -> None:
        new_x = float(server_data.get("translation_x", self.position["x"]))
        new_y = float(server_data.get("translation_y", self.position["y"]))
        new_z = float(server_data.get("translation_z", self.position["z"]))

        if soft_reanchor:
            dx = new_x - self.position["x"]
            dy = new_y - self.position["y"]
            dz = new_z - self.position["z"]
            delta_norm = float(np.sqrt(dx * dx + dy * dy + dz * dz))
            if (
                self._gps_reanchor_max_delta_m > 0.0
                and delta_norm > self._gps_reanchor_max_delta_m
                and delta_norm > 1e-9
            ):
                scale = self._gps_reanchor_max_delta_m / delta_norm
                dx *= scale
                dy *= scale
                dz *= scale

            alpha = self._gps_reanchor_alpha
            self.position["x"] += dx * alpha
            self.position["y"] += dy * alpha
            self.position["z"] += dz * alpha
        else:
            self.position["x"] = new_x
            self.position["y"] = new_y
            self.position["z"] = new_z

        if new_z > 0:
            self._last_gps_altitude = new_z

        self._last_gps_position = {
            "x": new_x,
            "y": new_y,
            "z": new_z,
        }

        self.log.debug(
            f"GPS güncelleme → X:{new_x:.2f}m Y:{new_y:.2f}m Z:{new_z:.2f}m"
        )

    def _update_from_optical_flow(
        self,
        gray: np.ndarray,
        server_data: Dict,
    ) -> bool:
        if self._prev_points is None or len(self._prev_points) < 10:
            self._update_reference_frame(gray)
            self.log.warn("Yetersiz köşe noktası — yeniden tespit ediliyor")
            return False

        next_points, status, err = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, self._prev_points, None, **self._lk_params
        )

        if next_points is None or status is None:
            self._update_reference_frame(gray)
            self.log.warn("Optik Akış başarısız — referans kare yenileniyor")
            return False

        mask = status.flatten() == 1

        if self._prev_points is None:
            return False

        good_old = self._prev_points[mask].reshape(-1, 2)
        good_new = next_points[mask].reshape(-1, 2)

        if len(good_new) < 5:
            self._update_reference_frame(gray)
            self.log.warn("Başarılı takip sayısı az — referans yenileniyor")
            return False

        displacements_x = good_new[:, 0] - good_old[:, 0]
        displacements_y = good_new[:, 1] - good_old[:, 1]

        # Rotasyon (pan/yaw) tespiti: kamera sağa/sola dönüyorsa pozisyon güncelleme
        is_rotation = False
        if getattr(Settings, "VO_ROTATION_SUPPRESS_ENABLED", True) and len(good_old) >= 6:
            h_img, w_img = gray.shape[:2]
            cx, cy = w_img / 2.0, h_img / 2.0
            rx = good_old[:, 0] - cx
            ry = good_old[:, 1] - cy
            dot = np.abs(displacements_x * rx + displacements_y * ry)
            flow_norm = np.sqrt(displacements_x ** 2 + displacements_y ** 2 + 1e-9)
            r_norm = np.sqrt(rx ** 2 + ry ** 2 + 1e-9)
            radial_ratio = dot / (flow_norm * r_norm + 1e-9)
            thresh = float(getattr(Settings, "VO_ROTATION_DOT_THRESHOLD", 0.4))
            if float(np.median(radial_ratio)) < thresh and float(np.median(flow_norm)) > 0.5:
                is_rotation = True

        dx_pixels, dy_pixels = self._robust_displacement(
            displacements_x, displacements_y
        )

        scale_ratio = 1.0
        if len(good_new) >= 3:
            good_old_pts = good_old[:, np.newaxis, :]
            good_new_pts = good_new[:, np.newaxis, :]
            dist_old = np.sqrt(np.sum((good_old_pts - good_old[np.newaxis, :, :]) ** 2, axis=-1))
            dist_new = np.sqrt(np.sum((good_new_pts - good_new[np.newaxis, :, :]) ** 2, axis=-1))
            iu = np.triu_indices(len(good_old), k=1)
            old_vals = dist_old[iu]
            new_vals = dist_new[iu]
            valid = old_vals > 5.0
            if np.any(valid):
                scale_ratio = float(np.median(new_vals[valid] / old_vals[valid]))

        raw_alt = server_data.get("translation_z", None)
        try:
            altitude = float(raw_alt)
            if np.isnan(altitude):
                altitude = 0.0
        except (TypeError, ValueError):
            altitude = 0.0

        if altitude <= 0:
            altitude = self._last_gps_altitude

        dx_meters, dy_meters = self._pixel_to_meter(dx_pixels, dy_pixels, altitude)
        alpha = self._ema_alpha
        self._ema_dx = alpha * dx_meters + (1 - alpha) * self._ema_dx
        self._ema_dy = alpha * dy_meters + (1 - alpha) * self._ema_dy

        cap = self._max_displacement_per_frame
        smooth_dx = max(-cap, min(cap, self._ema_dx))
        smooth_dy = max(-cap, min(cap, self._ema_dy))

        if not is_rotation:
            self.position["x"] += smooth_dx
            self.position["y"] += smooth_dy
            dz_meters = max(-cap, min(cap, (1.0 - scale_ratio) * max(altitude, 1.0)))
            self.position["z"] = max(0.0, float(self.position["z"]) + dz_meters)

        self._last_of_position = {k: v for k, v in self.position.items()}

        rot_tag = " [ROT]"
        self.log.debug(
            f"Optik Akış → dX:{dx_meters:.3f}m dY:{dy_meters:.3f}m dZ:{0.0 if is_rotation else (1.0 - scale_ratio) * max(altitude, 1.0):.3f}m | "
            f"Piksel: ({dx_pixels:.1f}, {dy_pixels:.1f}) | Scale: {scale_ratio:.3f} | "
            f"İrtifa: {altitude:.1f}m | "
            f"Takip: {len(good_new)}/{len(self._prev_points)} nokta{rot_tag if is_rotation else ''}"
        )

        if (
            self._initial_point_count > 0
            and len(good_new) < self._initial_point_count * 0.5
        ):
            self._update_reference_frame(gray)
            self.log.debug("Köşe noktası kaybı %50 üzeri — referans yenilendi")
        else:
            self._prev_gray = gray.copy()
            self._prev_points = good_new.reshape(-1, 1, 2)
        return True

    def _robust_displacement(
        self,
        dx_arr: np.ndarray,
        dy_arr: np.ndarray,
    ) -> Tuple[float, float]:
        """IQR ile aykırı piksel kaymalarını filtreleyip medyan döndür."""
        iqr_factor = float(getattr(Settings, "VO_OUTLIER_IQR_FACTOR", 0.0))
        if iqr_factor <= 0 or len(dx_arr) < 6:
            return float(np.median(dx_arr)), float(np.median(dy_arr))

        q1_x, q3_x = float(np.percentile(dx_arr, 25)), float(np.percentile(dx_arr, 75))
        q1_y, q3_y = float(np.percentile(dy_arr, 25)), float(np.percentile(dy_arr, 75))
        iqr_x = max(1e-6, q3_x - q1_x)
        iqr_y = max(1e-6, q3_y - q1_y)
        low_x, high_x = q1_x - iqr_factor * iqr_x, q3_x + iqr_factor * iqr_x
        low_y, high_y = q1_y - iqr_factor * iqr_y, q3_y + iqr_factor * iqr_y

        mask = (
            (dx_arr >= low_x) & (dx_arr <= high_x)
            & (dy_arr >= low_y) & (dy_arr <= high_y)
        )
        if np.sum(mask) < 3:
            return float(np.median(dx_arr)), float(np.median(dy_arr))
        return float(np.median(dx_arr[mask])), float(np.median(dy_arr[mask]))

    def _pixel_to_meter(
        self,
        dx_px: float,
        dy_px: float,
        altitude: float,
    ) -> Tuple[float, float]:
        focal = Settings.FOCAL_LENGTH_PX
        if focal <= 0:
            focal = 800.0

        sign_x = int(getattr(Settings, "VO_SIGN_X", -1))
        sign_y = int(getattr(Settings, "VO_SIGN_Y", -1))
        sign_x = 1 if sign_x >= 0 else -1
        sign_y = 1 if sign_y >= 0 else -1

        # Pinhole: metre = sign * piksel * irtifa / focal_length
        dx_m = sign_x * dx_px * altitude / focal
        dy_m = sign_y * dy_px * altitude / focal

        return dx_m, dy_m

    def _update_reference_frame(self, gray: np.ndarray) -> None:
        self._prev_gray = gray.copy()
        self._prev_points = cv2.goodFeaturesToTrack(
            gray, **self._feature_params
        )

        n_points = len(self._prev_points) if self._prev_points is not None else 0
        self._initial_point_count = n_points
        self.log.debug(f"Referans kare güncellendi — {n_points} köşe noktası")

    def get_position(self) -> Dict[str, float]:
        return {
            "x": round(self.position["x"], 4),
            "y": round(self.position["y"], 4),
            "z": round(self.position["z"], 4),
        }

    def get_last_of_position(self) -> Dict[str, float]:
        return {
            "x": round(self._last_of_position["x"], 4),
            "y": round(self._last_of_position["y"], 4),
            "z": round(self._last_of_position["z"], 4),
        }

    def project_position_with_latency(
        self,
        position: Dict[str, float],
        dt_sec: float,
        max_dt_sec: float,
        max_delta_m: float,
    ) -> Tuple[Dict[str, float], float, float]:
        return self._latency_comp.project_position(
            position=position,
            dt_sec=dt_sec,
            max_dt_sec=max_dt_sec,
            max_delta_m=max_delta_m,
        )

    def predict_without_measurement(
        self,
        reason_code: str,
        gps_health: int = 0,
    ) -> Dict[str, float]:
        now_mono = time.monotonic()
        freeze = bool(getattr(Settings, "GPS_ZERO_POSITION_FREEZE", True))

        if freeze and gps_health == 0:
            self._mode = "DEGRADED"
            self._runtime_meta = {
                "update_mode": "frozen",
                "state_source": "no_measurement",
                "quality_flag": "degraded",
                "reason_code": reason_code,
                "gps_health": int(gps_health),
            }
            self.log.debug(
                "event=position_frozen "
                f"reason_code={reason_code} — ölçüm yok, pozisyon değiştirilmedi (şartname)"
            )
            self._last_update_monotonic = now_mono
            return self.get_position()

        if self._last_update_monotonic is None:
            dt_sec = 0.0
        else:
            dt_sec = max(0.0, now_mono - self._last_update_monotonic)

        max_dt_sec = max(0.0, float(Settings.LATENCY_COMP_MAX_MS) / 1000.0)
        max_delta_m = max(0.0, float(self._max_displacement_per_frame))
        projected_position, applied_delta_m, used_dt_sec = self.project_position_with_latency(
            position=self.position,
            dt_sec=dt_sec,
            max_dt_sec=max_dt_sec,
            max_delta_m=max_delta_m,
        )

        self.position = dict(projected_position)
        self._last_of_position = dict(projected_position)
        self._mode = "DEGRADED"
        self._runtime_meta = {
            "update_mode": "predict-only",
            "state_source": "vision_predict",
            "quality_flag": "degraded",
            "reason_code": reason_code,
            "gps_health": int(gps_health),
            "predict_dt_sec": round(used_dt_sec, 6),
            "predict_delta_m": round(applied_delta_m, 6),
        }
        self.log.debug(
            "event=predict_only_update "
            f"reason_code={reason_code} dt_sec={used_dt_sec:.4f} "
            f"delta_m={applied_delta_m:.4f}"
        )
        self._latency_comp.update_velocity(self.position, sample_monotonic=now_mono)
        self._last_update_monotonic = now_mono
        return self.get_position()

    def get_runtime_meta(self) -> Dict[str, object]:
        return dict(self._runtime_meta)

    def reset(self) -> None:
        self.position = {"x": 0.0, "y": 0.0, "z": 0.0}
        self._last_of_position = {"x": 0.0, "y": 0.0, "z": 0.0}
        self._prev_gray = None
        self._prev_points = None
        self._last_gps_position = None
        self._initial_point_count = 0
        self._latency_comp.reset()
        self.log.info("Visual Odometry sıfırlandı → (0, 0, 0)")
