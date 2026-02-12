"""
TEKNOFEST Havacılıkta Yapay Zeka - Ana Orkestrasyon Dosyası
============================================================
Tüm modülleri bir araya getirir ve ana işlem döngüsünü yönetir.

İş Akışı (her kare için):
    1. Sunucudan kare meta verisi al (NetworkManager.get_frame)
    2. Görüntüyü indir (NetworkManager.download_image)
    3. Nesne tespiti yap (ObjectDetector.detect)
    4. Konum kestirimi yap (VisualOdometry.update)
    5. Sonuçları sunucuya gönder (NetworkManager.send_result)

Güvenlik:
    - Global try/except → sistem ASLA çökmez
    - Her modül kendi hatalarını yakalar
    - FPS sayacı sürekli konsola basılır

Kullanım:
    python main.py
"""

import os
import sys
import time
import signal
from typing import Optional

import torch

# Proje kök dizinini Python path'e ekle
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.settings import Settings
from src.utils import Logger, Visualizer
from src.network import NetworkManager
from src.detection import ObjectDetector
from src.localization import VisualOdometry


# =============================================================================
#  SİSTEM BANNER'I
# =============================================================================

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║     🛩️  TEKNOFEST 2025 - HAVACILIKTA YAPAY ZEKA YARIŞMASI    ║
║     ──────────────────────────────────────────────────────    ║
║     Nesne Tespiti (Görev 1) + Konum Kestirimi (Görev 2)     ║
╚══════════════════════════════════════════════════════════════╝
"""


def print_system_info(log: Logger) -> None:
    """Sistem bilgilerini konsola basar — başlangıç diagnostiği."""
    print(BANNER)
    log.info(f"Çalışma Dizini  : {PROJECT_ROOT}")
    log.info(f"Simülasyon Modu : {'AÇIK ✓' if Settings.SIMULATION_MODE else 'KAPALI (YARIŞMA)'}")
    log.info(f"Debug Modu      : {'AÇIK' if Settings.DEBUG else 'KAPALI'}")
    log.info(f"Sunucu          : {Settings.BASE_URL}")
    log.info(f"Model           : {Settings.MODEL_PATH}")
    log.info(f"Cihaz           : {Settings.DEVICE}")
    log.info(f"FP16            : {'AÇIK' if Settings.HALF_PRECISION else 'KAPALI'}")

    if torch.cuda.is_available():
        log.success(f"GPU             : {torch.cuda.get_device_name(0)}")
        mem_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        log.success(f"GPU Bellek      : {mem_total:.1f} GB")
    else:
        log.warn("GPU             : BULUNAMADI — CPU modunda çalışılacak")


# =============================================================================
#  FPS SAYACI
# =============================================================================

class FPSCounter:
    """
    Gerçek zamanlı FPS (Frame Per Second) hesaplayıcı.

    Belirli aralıklarla konsola ortalama FPS değerini basar.
    """

    def __init__(self, report_interval: int = 10) -> None:
        self.report_interval = report_interval
        self.frame_count: int = 0
        self.start_time: float = time.time()
        self.log = Logger("FPS")

    def tick(self) -> Optional[float]:
        """
        Bir kare işlendiğini bildirir.

        Her report_interval karede bir FPS değerini konsola basar.

        Returns:
            Raporlama anında FPS değeri, aksi halde None.
        """
        self.frame_count += 1

        if self.frame_count % self.report_interval == 0:
            elapsed = time.time() - self.start_time
            fps = self.frame_count / elapsed if elapsed > 0 else 0
            self.log.info(
                f"Kare: {self.frame_count} | "
                f"FPS: {fps:.2f} | "
                f"Süre: {elapsed:.1f}s"
            )
            return fps
        return None


# =============================================================================
#  ANA DÖNGÜ
# =============================================================================

def main() -> None:
    """
    Sistemin ana giriş noktası.

    Tüm modülleri başlatır ve sonsuz döngüde kare işleme pipeline'ını çalıştırır.
    Global try/except ile asla çökmez — hata olursa loglar ve devam eder.
    Video sona erdiğinde (veya MAX_FRAMES'e ulaşıldığında) temiz kapanış yapar.
    """
    log = Logger("Main")

    # ======= SİSTEM BİLGİSİ =======
    print_system_info(log)

    # ======= MODÜLLERİ BAŞLAT =======
    log.info("Modüller başlatılıyor...")

    try:
        network = NetworkManager()
        detector = ObjectDetector()
        odometry = VisualOdometry()
        fps_counter = FPSCounter(report_interval=Settings.FPS_REPORT_INTERVAL)

        # Debug modunda Visualizer'ı başlat
        visualizer: Optional[Visualizer] = None
        if Settings.DEBUG:
            visualizer = Visualizer()

        log.success("Tüm modüller başarıyla başlatıldı ✓")

    except Exception as e:
        log.error(f"Modül başlatma hatası: {e}")
        log.error("Sistem başlatılamadı — çıkılıyor.")
        return

    # ======= OTURUM BAŞLAT =======
    if not network.start_session():
        log.error("Sunucu oturumu başlatılamadı!")
        log.warn("Yeniden denenecek...")
        time.sleep(Settings.RETRY_DELAY)
        if not network.start_session():
            log.error("İkinci deneme de başarısız — çıkılıyor.")
            return

    log.success("═" * 50)
    log.success("  SİSTEM HAZIR — İşlem döngüsü başlıyor...")
    log.success("═" * 50)

    # ======= ANA İŞLEM DÖNGÜSÜ =======
    running = True
    consecutive_none_count = 0  # Ardışık None sayacı (video sonu tespiti)

    # Ctrl+C ile temiz kapanış
    def signal_handler(sig, frame):
        nonlocal running
        running = False
        log.warn("\nKapatma sinyali alındı (Ctrl+C) — döngü durduruluyor...")

    signal.signal(signal.SIGINT, signal_handler)

    while running:
        try:
            # ---- KARE LİMİTİ KONTROLÜ ----
            if fps_counter.frame_count >= Settings.MAX_FRAMES:
                log.success(
                    f"Maksimum kare sayısına ulaşıldı ({Settings.MAX_FRAMES}) — "
                    f"oturum tamamlandı ✓"
                )
                break

            # ---- 1) SUNUCUDAN KARE META VERİSİ AL ----
            frame_data = network.get_frame()

            if frame_data is None:
                consecutive_none_count += 1
                if consecutive_none_count >= 5:
                    log.info("Video sona erdi (5 ardışık boş yanıt) — çıkılıyor")
                    break
                log.warn("Kare verisi alınamadı — bekleniyor...")
                time.sleep(0.5)
                continue

            consecutive_none_count = 0  # Başarılı çekim → sıfırla
            frame_id = frame_data.get("frame_id", "unknown")

            # ---- 2) GÖRÜNTÜYÜ İNDİR ----
            frame = network.download_image(frame_data)

            if frame is None:
                log.warn(f"Kare {frame_id}: Görüntü indirilemedi — atlanıyor")
                continue

            # ---- 3) NESNE TESPİTİ (GÖREV 1) ----
            detected_objects = detector.detect(frame)

            # ---- 4) KONUM KESTİRİMİ (GÖREV 2) ----
            position = odometry.update(frame, frame_data)

            # TEKNOFEST formatına dönüştür
            detected_translation = {
                "translation_x": position["x"],
                "translation_y": position["y"],
                "translation_z": position["z"],
            }

            # ---- 5) SONUÇLARI GÖNDER ----
            success = network.send_result(
                frame_id, detected_objects, detected_translation
            )

            if not success:
                log.warn(f"Kare {frame_id}: Sonuç gönderilemedi!")

            # ---- 6) DEBUG ÇIKTISI ----
            if Settings.DEBUG and visualizer is not None:
                visualizer.draw_detections(
                    frame, detected_objects,
                    frame_id=str(frame_id),
                    position=position,
                )

            # ---- 7) FPS GÜNCELLE ----
            fps_counter.tick()

            # ---- 8) DÖNGÜ ARASI BEKLEME ----
            if Settings.LOOP_DELAY > 0:
                time.sleep(Settings.LOOP_DELAY)

        except KeyboardInterrupt:
            log.warn("Kullanıcı tarafından durduruldu (KeyboardInterrupt)")
            break

        except Exception as e:
            # ===== GLOBAL HATA YAKALAMA =====
            # Sistem ASLA çökmemeli — hata logla, devam et
            log.error(f"İşlem hatası: {e}")
            log.warn("Sonraki kareye geçiliyor...")
            time.sleep(0.5)  # Hata döngüsüne girmeyi engelle

    # ======= TEMİZ KAPANIŞ =======
    log.info("─" * 50)
    log.info(f"Toplam işlenen kare: {fps_counter.frame_count}")
    elapsed = time.time() - fps_counter.start_time
    if elapsed > 0:
        avg_fps = fps_counter.frame_count / elapsed
        log.info(f"Ortalama FPS: {avg_fps:.2f}")
    log.info(f"Toplam süre: {elapsed:.1f} saniye")

    # GPU belleğini temizle
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        log.info("GPU belleği temizlendi")

    log.success("Sistem kapatıldı. Güle güle! 👋")


if __name__ == "__main__":
    main()
