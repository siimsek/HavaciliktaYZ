"""
TEKNOFEST Havacılıkta Yapay Zeka - Ana Orkestrasyon Dosyası
============================================================
Tüm modülleri bir araya getirir ve ana işlem döngüsünü yönetir.

Çalışma Modları:
    1. Yarışma Modu (varsayılan):
       Sunucudan kare alır → tespit → konum → sonuç gönderir.

    2. Otonom Test Modu (--simulate):
       VisDrone veri setinden kare okur → tespit → konum → renkli log.

İş Akışı (her kare için):
    1. Kare al (Sunucu veya DatasetLoader)
    2. Nesne tespiti yap (ObjectDetector.detect)
    3. Konum kestirimi yap (VisualOdometry.update)
    4. Sonuçları raporla (Sunucuya gönder veya terminale bas)

Güvenlik:
    - Global try/except → sistem ASLA çökmez
    - Her modül kendi hatalarını yakalar
    - FPS sayacı sürekli konsola basılır

Kullanım:
    python main.py                  # Yarışma modu (settings'e göre)
    python main.py --simulate       # Otonom test modu (VisDrone)
    python main.py --simulate det   # Sadece DET veri seti (Görev 1)
"""

import os
import sys
import time
import signal
import argparse
import traceback
from collections import Counter
from typing import Optional

import cv2

import torch

# Proje kök dizinini Python path'e ekle
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.settings import Settings
from src.utils import Logger, Visualizer
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


def print_system_info(log: Logger, simulate: bool = False) -> None:
    """Sistem bilgilerini konsola basar — başlangıç diagnostiği."""
    print(BANNER)
    log.info(f"Çalışma Dizini  : {PROJECT_ROOT}")

    if simulate:
        log.info("Çalışma Modu    : 🧪 OTONOM TEST (VisDrone)")
    elif Settings.SIMULATION_MODE:
        log.info("Simülasyon Modu : AÇIK ✓ (Statik görüntü)")
    else:
        log.info("Simülasyon Modu : KAPALI (YARIŞMA)")

    log.info(f"Debug Modu      : {'AÇIK' if Settings.DEBUG else 'KAPALI'}")
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
#  OTONOM TEST DÖNGÜSÜ (VisDrone)
# =============================================================================

def run_simulation(
    log: Logger,
    prefer_vid: bool = True,
    show: bool = False,
    save: bool = False,
) -> None:
    """
    VisDrone veri seti üzerinde otonom test çalıştırır.

    Sunucu gerektirmez — DatasetLoader'dan kareler okunur,
    tespit + odometri yapılır, sonuçlar renkli olarak terminale basılır.

    Args:
        log: Logger instance.
        prefer_vid: True → VID (sekans, Görev 2), False → DET (tekil, Görev 1).
        show: True → cv2.imshow ile canlı görüntüleme.
        save: True → Her kareyi debug_output/ dizinine kaydet.
    """
    from src.data_loader import DatasetLoader

    # --- Modüller ---
    log.info("Modüller başlatılıyor...")

    try:
        loader = DatasetLoader(prefer_vid=prefer_vid)
        if not loader.is_ready:
            log.error("Veri seti yüklenemedi — çıkılıyor.")
            return

        detector = ObjectDetector()
        odometry = VisualOdometry()
        fps_counter = FPSCounter(report_interval=Settings.FPS_REPORT_INTERVAL)

        visualizer = Visualizer()

        # Kayıt dizinini hazırla (--save için)
        if save:
            os.makedirs(Settings.DEBUG_OUTPUT_DIR, exist_ok=True)
            log.info(f"Görseller kaydedilecek: {Settings.DEBUG_OUTPUT_DIR}")

        log.success("Tüm modüller başarıyla başlatıldı ✓")

    except Exception as e:
        log.error(f"Modül başlatma hatası: {e}")
        log.error(f"Stack trace:\n{traceback.format_exc()}")
        return

    log.success("═" * 50)
    log.success(f"  OTONOM TEST BAŞLIYOR — {len(loader)} kare işlenecek")
    log.success("═" * 50)

    # --- Döngü ---
    running = True

    def signal_handler(sig, frame):
        nonlocal running
        running = False
        log.warn("\nKapatma sinyali alındı — döngü durduruluyor...")

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    for frame_info in loader:
        if not running:
            break

        # Kare limiti
        if fps_counter.frame_count >= Settings.MAX_FRAMES:
            log.success(f"Maksimum kare sayısına ulaşıldı ({Settings.MAX_FRAMES})")
            break

        try:
            frame = frame_info["frame"]
            frame_idx = frame_info["frame_idx"]
            server_data = frame_info["server_data"]
            gps_health = frame_info["gps_health"]

            # ---- NESNE TESPİTİ (Görev 1) ----
            detected_objects = detector.detect(frame)

            # ---- KONUM KESTİRİMİ (Görev 2) ----
            position = odometry.update(frame, server_data)

            # ---- RENKLİ SONUÇ LOGU ----
            _print_simulation_result(
                log, frame_idx, detected_objects, position, gps_health,
                frame_info["filename"]
            )

            # ---- GÖRSEL ÇIKTI ----
            if show or save:
                annotated = visualizer.draw_detections(
                    frame, detected_objects,
                    frame_id=str(frame_idx),
                    position=position,
                )

                # Ekstra bilgi: GPS/OF modu ve FPS
                mode_text = "GPS" if gps_health == 1 else "Optical Flow"
                cv2.putText(
                    annotated, f"Mode: {mode_text}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if gps_health else (0, 165, 255), 2,
                )

                if show:
                    cv2.imshow("TEKNOFEST - Otonom Test", annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord('q'), 27):  # q veya ESC
                        log.info("Kullanıcı pencereyi kapattı (q/ESC)")
                        break

                if save:
                    save_path = os.path.join(
                        Settings.DEBUG_OUTPUT_DIR,
                        f"frame_{frame_idx:04d}.jpg",
                    )
                    cv2.imwrite(save_path, annotated)

            # ---- FPS ----
            fps_counter.tick()

        except Exception as e:
            log.error(f"Kare {frame_info.get('frame_idx', '?')} hatası: {e}")
            log.error(f"Stack trace:\n{traceback.format_exc()}")
            continue

    # --- Temiz Kapanış ---
    if show:
        cv2.destroyAllWindows()
    if save:
        log.success(f"Görseller kaydedildi: {Settings.DEBUG_OUTPUT_DIR}/")
    _print_summary(log, fps_counter)


def _print_simulation_result(
    log: Logger,
    frame_idx: int,
    detected_objects: list,
    position: dict,
    gps_health: int,
    filename: str,
) -> None:
    """Simülasyon sonucunu renkli olarak terminale basar."""
    # Sınıf sayımı
    cls_counts = Counter(obj["cls"] for obj in detected_objects)
    tasit = cls_counts.get("0", 0)
    insan = cls_counts.get("1", 0)
    uap = cls_counts.get("2", 0)
    uai = cls_counts.get("3", 0)

    # Konum bilgisi
    loc_mode = "GPS" if gps_health == 1 else "OF"
    pos_str = (
        f"x={position['x']:+.1f}m "
        f"y={position['y']:+.1f}m "
        f"z={position['z']:.0f}m"
    )

    # Renkli log
    log.success(
        f"Frame: {frame_idx:04d} | "
        f"Tespit: {len(detected_objects)} "
        f"({tasit} Taşıt, {insan} İnsan"
        f"{f', {uap} UAP' if uap else ''}"
        f"{f', {uai} UAİ' if uai else ''}) | "
        f"Konum: {pos_str} ({loc_mode})"
    )


# =============================================================================
#  YARIŞMA DÖNGÜSÜ (Sunucu)
# =============================================================================

def run_competition(log: Logger) -> None:
    """
    Yarışma/sunucu modunda ana işlem döngüsünü çalıştırır.

    Sunucudan kare alır → tespit → konum → sonuç gönderir.
    """
    from src.network import NetworkManager

    # --- Modüller ---
    log.info("Modüller başlatılıyor...")

    try:
        network = NetworkManager()
        detector = ObjectDetector()
        odometry = VisualOdometry()
        fps_counter = FPSCounter(report_interval=Settings.FPS_REPORT_INTERVAL)

        visualizer: Optional[Visualizer] = None
        if Settings.DEBUG:
            visualizer = Visualizer()

        log.success("Tüm modüller başarıyla başlatıldı ✓")

    except Exception as e:
        log.error(f"Modül başlatma hatası: {e}")
        log.error("Sistem başlatılamadı — çıkılıyor.")
        return

    # --- Oturum Başlat ---
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

    # --- Döngü ---
    running = True
    consecutive_none_count = 0

    def signal_handler(sig, frame):
        nonlocal running
        running = False
        log.warn("\nKapatma sinyali alındı (Ctrl+C) — döngü durduruluyor...")

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

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

            consecutive_none_count = 0
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
            log.error(f"İşlem hatası: {e}")
            log.error(f"Stack trace:\n{traceback.format_exc()}")
            log.warn("Sonraki kareye geçiliyor...")
            time.sleep(0.5)

    # --- Temiz Kapanış ---
    _print_summary(log, fps_counter)


# =============================================================================
#  YARDIMCI FONKSİYONLAR
# =============================================================================

def _print_summary(log: Logger, fps_counter: FPSCounter) -> None:
    """Oturum sonunda özet bilgileri basar."""
    log.info("─" * 50)
    log.info(f"Toplam işlenen kare: {fps_counter.frame_count}")
    elapsed = time.time() - fps_counter.start_time
    if elapsed > 0:
        avg_fps = fps_counter.frame_count / elapsed
        log.info(f"Ortalama FPS: {avg_fps:.2f}")
    log.info(f"Toplam süre: {elapsed:.1f} saniye")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        log.info("GPU belleği temizlendi")

    log.success("Sistem kapatıldı. Güle güle! 👋")


def _ask_choice(prompt: str, options: dict) -> str:
    """
    Kullanıcıdan geçerli bir seçim ister.

    Args:
        prompt: Kullanıcıya gösterilecek soru.
        options: {tuş: açıklama} sözlüğü.

    Returns:
        Seçilen tuş (string).
    """
    print()
    print(prompt)
    for key, desc in options.items():
        print(f"  [{key}] {desc}")
    print()

    while True:
        choice = input("  Seçiminiz: ").strip()
        if choice in options:
            return choice
        print(f"  ⚠ Geçersiz seçim! Lütfen {', '.join(options.keys())} girin.")


def show_interactive_menu() -> dict:
    """
    Başlangıç menüsünü gösterir ve kullanıcı tercihlerini toplar.

    Returns:
        Dict: mode, prefer_vid, show, save anahtarları.
    """
    print("\n" + "═" * 56)
    print("  🎯  ÇALIŞMA MODU SEÇİMİ")
    print("═" * 56)

    # 1) Mod seçimi
    mode = _ask_choice(
        "  Hangi modda çalıştırmak istiyorsunuz?",
        {
            "1": "🏆  Yarışma Modu (sunucu bağlantısı)",
            "2": "🎬  Otonom Test — VID (sıralı kareler, Görev 2)",
            "3": "📸  Otonom Test — DET (tekil fotoğraflar, Görev 1)",
        },
    )

    if mode == "1":
        return {"mode": "competition", "prefer_vid": True, "show": False, "save": False}

    prefer_vid = (mode == "2")

    # 2) Görsel çıktı seçimi
    print("\n" + "─" * 56)
    output = _ask_choice(
        "  Sonuçları nasıl görmek istiyorsunuz?",
        {
            "1": "📊  Sadece terminal çıktısı (en hızlı)",
            "2": "🖥️   Canlı pencerede göster (cv2.imshow)",
            "3": "💾  Kareleri diske kaydet (debug_output/)",
            "4": "🖥️💾 Hem pencerede göster hem kaydet",
        },
    )

    show = output in ("2", "4")
    save = output in ("3", "4")

    return {"mode": "simulate", "prefer_vid": prefer_vid, "show": show, "save": save}


# =============================================================================
#  ANA GİRİŞ NOKTASI
# =============================================================================

def main() -> None:
    """
    Sistemin ana giriş noktası.

    Kullanıcıya interaktif menü sunar — seçimlere göre
    yarışma veya otonom test modu başlatılır.
    """
    log = Logger("Main")

    # Banner
    print(BANNER)

    # İnteraktif menü
    choices = show_interactive_menu()

    # Sistem bilgisi
    simulate = (choices["mode"] == "simulate")
    print_system_info(log, simulate=simulate)

    if simulate:
        run_simulation(
            log,
            prefer_vid=choices["prefer_vid"],
            show=choices["show"],
            save=choices["save"],
        )
    else:
        run_competition(log)


if __name__ == "__main__":
    main()
