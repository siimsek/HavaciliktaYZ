"""
TEKNOFEST Havacılıkta Yapay Zeka - Yardımcı Araçlar (Utilities)
================================================================
Logger  : Renkli, seviyeli konsol çıktıları.
Visualizer : Debug modunda görüntü üzerine bounding box ve etiket çizer.
log_json_to_disk : Gelen/giden JSON verilerini diske kaydeder.

Kullanım:
    from src.utils import Logger, Visualizer, log_json_to_disk
    log = Logger("Main")
    log.info("Sistem başlatıldı")
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Optional, Any

import cv2
import numpy as np

try:
    from colorama import init as colorama_init, Fore, Style
    colorama_init(autoreset=True)
    _HAS_COLORAMA = True
except ImportError:
    _HAS_COLORAMA = False

from config.settings import Settings


# =============================================================================
#  LOGGER SINIFI
# =============================================================================

class Logger:
    """
    Renkli ve seviyeli terminal çıktıları üreten log sınıfı.

    Seviyeler:
        DEBUG  → Gri   (yalnızca Settings.DEBUG=True iken görünür)
        INFO   → Yeşil
        WARN   → Sarı
        ERROR  → Kırmızı
        SUCCESS→ Cyan

    Args:
        module_name: Mesajın kaynağı (örn: 'Network', 'Detector')
    """

    def __init__(self, module_name: str) -> None:
        self.module_name = module_name

    def _timestamp(self) -> str:
        """Şu anki zamanı [HH:MM:SS.mmm] formatında döndürür."""
        now = datetime.now()
        return now.strftime("%H:%M:%S.") + f"{now.microsecond // 1000:03d}"

    def _print(self, level: str, color: str, message: str) -> None:
        """Formatlanmış log satırını konsola basar."""
        ts = self._timestamp()
        prefix = f"[{ts}] [{level:^7}] [{self.module_name}]"
        if _HAS_COLORAMA:
            print(f"{color}{prefix}{Style.RESET_ALL} {message}")
        else:
            print(f"{prefix} {message}")

    def debug(self, message: str) -> None:
        """Debug seviyesi — yalnızca DEBUG=True iken çıktı verir."""
        if Settings.DEBUG:
            color = Fore.WHITE if _HAS_COLORAMA else ""
            self._print("DEBUG", color, message)

    def info(self, message: str) -> None:
        """Bilgilendirme seviyesi."""
        color = Fore.GREEN if _HAS_COLORAMA else ""
        self._print("INFO", color, message)

    def warn(self, message: str) -> None:
        """Uyarı seviyesi."""
        color = Fore.YELLOW if _HAS_COLORAMA else ""
        self._print("WARN", color, message)

    def error(self, message: str) -> None:
        """Hata seviyesi."""
        color = Fore.RED if _HAS_COLORAMA else ""
        self._print("ERROR", color, message)

    def success(self, message: str) -> None:
        """Başarı seviyesi."""
        color = Fore.CYAN if _HAS_COLORAMA else ""
        self._print("SUCCESS", color, message)


# =============================================================================
#  VISUALIZER SINIFI
# =============================================================================

class Visualizer:
    """
    Debug modunda görüntü üzerine bounding box, sınıf etiketi,
    güven skoru ve iniş durumu bilgisi çizen yardımcı sınıf.

    Çıktıları Settings.DEBUG_OUTPUT_DIR dizinine kaydeder.
    DEBUG_SAVE_INTERVAL ile kontrol edilen aralıklarla diske yazar.
    """

    # Sınıf ID → Renk eşleştirmesi (BGR formatında)
    CLASS_COLORS: Dict[int, tuple] = {
        0: (0, 255, 0),      # Taşıt → Yeşil
        1: (255, 0, 0),      # İnsan → Mavi
        2: (255, 255, 0),    # UAP → Cyan
        3: (0, 0, 255),      # UAİ → Kırmızı
    }

    # Sınıf ID → Etiket adı
    CLASS_NAMES: Dict[int, str] = {
        0: "Tasit",
        1: "Insan",
        2: "UAP",
        3: "UAI",
    }

    # İniş Durumu → Metin
    LANDING_LABELS: Dict[str, str] = {
        "-1": "",
        "0": " [UYGUN DEGIL]",
        "1": " [UYGUN]",
    }

    def __init__(self) -> None:
        """Debug çıktı dizinini oluşturur."""
        os.makedirs(Settings.DEBUG_OUTPUT_DIR, exist_ok=True)
        self.log = Logger("Visualizer")
        self._save_counter: int = 0

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Dict],
        frame_id: str = "unknown",
        position: Optional[Dict] = None,
    ) -> np.ndarray:
        """
        Görüntü üzerine tespit sonuçlarını çizer ve belirli aralıklarla diske kaydeder.

        Args:
            frame: BGR formatlı OpenCV görüntüsü.
            detections: Tespit edilen nesnelerin listesi (JSON formatında).
            frame_id: Kare kimliği (dosya adı için).
            position: Pozisyon bilgisi dict (x, y, z).

        Returns:
            Üzerine çizim yapılmış görüntü kopyası.
        """
        annotated = frame.copy()

        for det in detections:
            cls_id = int(det.get("cls", -1))
            landing = det.get("landing_status", "-1")
            x1 = int(float(det.get("top_left_x", 0)))
            y1 = int(float(det.get("top_left_y", 0)))
            x2 = int(float(det.get("bottom_right_x", 0)))
            y2 = int(float(det.get("bottom_right_y", 0)))
            conf = det.get("confidence", 0.0)

            color = self.CLASS_COLORS.get(cls_id, (200, 200, 200))
            label_name = self.CLASS_NAMES.get(cls_id, "?")
            landing_txt = self.LANDING_LABELS.get(landing, "")

            # Bounding Box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Etiket Metni
            label = f"{label_name} {conf:.2f}{landing_txt}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                annotated,
                (x1, y1 - label_size[1] - 6),
                (x1 + label_size[0], y1),
                color,
                -1,
            )
            cv2.putText(
                annotated, label, (x1, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
            )

        # Pozisyon bilgisini sol üst köşeye yaz
        if position:
            pos_text = (
                f"X:{position.get('x', 0):.2f}m "
                f"Y:{position.get('y', 0):.2f}m "
                f"Z:{position.get('z', 0):.2f}m"
            )
            cv2.putText(
                annotated, pos_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
            )

        # Diske kaydet — sadece belirli aralıklarla (I/O darboğazı önleme)
        self._save_counter += 1
        if self._save_counter % Settings.DEBUG_SAVE_INTERVAL == 0:
            save_path = os.path.join(Settings.DEBUG_OUTPUT_DIR, f"{frame_id}.jpg")
            cv2.imwrite(save_path, annotated)
            self.log.debug(f"Debug görsel kaydedildi: {save_path}")

        return annotated


# =============================================================================
#  JSON LOGLAMA FONKSİYONU
# =============================================================================

def log_json_to_disk(
    data: Any,
    direction: str = "outgoing",
    tag: str = "general",
) -> None:
    """
    JSON verisini logs/ dizinine zaman damgalı dosya olarak kaydeder.

    Args:
        data: Kaydedilecek veri (dict, list veya JSON-serializable nesne).
        direction: 'incoming' (sunucudan gelen) veya 'outgoing' (gönderilen).
        tag: Ek etiket (örn: 'frame_42').
    """
    try:
        os.makedirs(Settings.LOG_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{timestamp}_{direction}_{tag}.json"
        filepath = os.path.join(Settings.LOG_DIR, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    except Exception:
        # Loglama hatası sistemi durdurmamalı
        pass
