"""
TEKNOFEST Havacılıkta Yapay Zeka Yarışması - Merkezi Konfigürasyon Dosyası
===========================================================================
Tüm sistem parametreleri bu dosyada tanımlanır. Yarışma günü yalnızca
bu dosyadaki değerler güncellenerek sisteme adapte olunur.

Kullanım:
    from config.settings import Settings
    print(Settings.BASE_URL)
"""

import os
from pathlib import Path


# =============================================================================
#  PROJE KÖK DİZİNİ
# =============================================================================
# Bu dosyanın iki üst dizini = proje kökü (HavaciliktaYZ/)
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


class Settings:
    """
    Tüm sistem ayarlarını barındıran merkezi konfigürasyon sınıfı.

    Yarışma günü değiştirilmesi gereken parametreler en üstte,
    nadiren değişen parametreler altta gruplandırılmıştır.
    """

    # =========================================================================
    #  YARIŞMA GÜNÜ DEĞİŞECEK PARAMETRELER
    # =========================================================================

    # Sunucu Bağlantısı - Yarışma günü güncellenecek
    BASE_URL: str = "http://127.0.0.1:5000"

    # API Endpoint'leri
    ENDPOINT_NEXT_FRAME: str = "/next_frame"
    ENDPOINT_SUBMIT_RESULT: str = "/submit_result"

    # Takım Bilgileri - Yarışma günü güncellenecek
    TEAM_NAME: str = "Takim_ID"

    # Çalışma Modları
    SIMULATION_MODE: bool = True    # True: Yerel test, False: Yarışma
    DEBUG: bool = True              # True: Detaylı log + görsel çıktı

    # =========================================================================
    #  MODEL AYARLARI
    # =========================================================================

    # YOLOv8 Model Dosyası (yerel diskten yüklenir - OFFLINE MODE)
    MODEL_PATH: str = str(PROJECT_ROOT / "models" / "yolov8n.pt")

    # Tespit Güven Eşiği (0.0 - 1.0)
    CONFIDENCE_THRESHOLD: float = 0.25

    # NMS IoU Eşiği (Non-Maximum Suppression)
    NMS_IOU_THRESHOLD: float = 0.45

    # Cihaz Seçimi (cuda: GPU, cpu: İşlemci)
    DEVICE: str = "cuda"

    # FP16 Yarı Hassasiyet — RTX 3060'ta ~%40 hız artışı sağlar
    HALF_PRECISION: bool = True

    # Model ısınma tekrar sayısı (ilk kare gecikmesini önler)
    WARMUP_ITERATIONS: int = 3

    # =========================================================================
    #  SINIF TANIMLARI (TEKNOFEST Şartname)
    # =========================================================================

    # TEKNOFEST Sınıf ID'leri
    CLASS_TASIT: int = 0       # Taşıt
    CLASS_INSAN: int = 1       # İnsan
    CLASS_UAP: int = 2         # Uçan Araba Park Alanı
    CLASS_UAI: int = 3         # Uçan Ambulans İniş Alanı

    # İniş Durumu Kodları
    LANDING_NOT_AREA: str = "-1"    # İniş alanı değil (Taşıt/İnsan)
    LANDING_NOT_SUITABLE: str = "0" # İniş için uygun değil
    LANDING_SUITABLE: str = "1"     # İniş için uygun

    # COCO → TEKNOFEST Sınıf Eşleştirme Tablosu
    # COCO Dataset Sınıf Numaraları:
    #   0=person, 1=bicycle, 2=car, 3=motorcycle, 4=airplane,
    #   5=bus, 6=train, 7=truck, 8=boat, 9=traffic light ...
    #
    # Şartname Kuralları:
    #   - Tüm motorlu karayolu taşıtları → Taşıt (0)
    #   - Raylı taşıtlar (tren, tramvay) → Taşıt (0)
    #   - Tüm deniz taşıtları → Taşıt (0)
    #   - Bisiklet/motosiklet sürücüsü → Taşıt (sürücüyle birlikte bütün)
    #   - Tüm insanlar → İnsan (1)
    COCO_TO_TEKNOFEST: dict = {
        0: 1,    # person → İnsan
        1: 0,    # bicycle → Taşıt (sürücüsüyle birlikte)
        2: 0,    # car → Taşıt
        3: 0,    # motorcycle → Taşıt
        5: 0,    # bus → Taşıt
        6: 0,    # train → Taşıt (vagonlar dahil)
        7: 0,    # truck → Taşıt
        8: 0,    # boat → Taşıt (deniz taşıtı)
    }

    # İniş alanı üzerine nesne kontrolü için kesişim eşiği
    # Şartname: "Herhangi bir nesne varsa iniş için uygun değildir"
    # Bu yüzden eşik 0.0 — herhangi bir kesişim = uygun değil
    LANDING_IOU_THRESHOLD: float = 0.0

    # =========================================================================
    #  KAMERA PARAMETRELERİ (Yarışma günü kalibrasyon ile güncellenecek)
    # =========================================================================

    # Odak Uzaklığı (piksel cinsinden) - Kamera modeline göre değişir
    FOCAL_LENGTH_PX: float = 800.0

    # Görüntü Merkez Noktası (piksel)
    CAMERA_CX: float = 960.0   # 1920 / 2
    CAMERA_CY: float = 540.0   # 1080 / 2

    # Varsayılan İrtifa (metre) - Optik akış hesabında fallback
    DEFAULT_ALTITUDE: float = 50.0

    # =========================================================================
    #  AĞ AYARLARI
    # =========================================================================

    # HTTP İstek Timeout (saniye)
    REQUEST_TIMEOUT: int = 5

    # Bağlantı Hatası Retry Sayısı
    MAX_RETRIES: int = 3

    # Retry Arası Bekleme (saniye)
    RETRY_DELAY: float = 1.0

    # =========================================================================
    #  DOSYA YOLLARI
    # =========================================================================

    # Log Dizini
    LOG_DIR: str = str(PROJECT_ROOT / "logs")

    # Geçici Kare Dosyası
    TEMP_FRAME_PATH: str = str(PROJECT_ROOT / "temp_frame.jpg")

    # Debug Çıktı Dizini
    DEBUG_OUTPUT_DIR: str = str(PROJECT_ROOT / "debug_output")

    # Simülasyon Test Görseli
    SIMULATION_IMAGE_PATH: str = str(PROJECT_ROOT / "bus.jpg")

    # =========================================================================
    #  PERFORMANS AYARLARI
    # =========================================================================

    # FPS Raporlama Aralığı (her N karede bir)
    FPS_REPORT_INTERVAL: int = 10

    # Ana Döngü Bekleme Süresi (saniye) - 0 = maksimum hız
    LOOP_DELAY: float = 0.0

    # GPU Bellek Temizleme Aralığı (her N karede bir)
    GPU_CLEANUP_INTERVAL: int = 100

    # Debug görsel kaydetme aralığı (her N karede diske yaz)
    DEBUG_SAVE_INTERVAL: int = 50

    # =========================================================================
    #  YARIŞMA LİMİTLERİ (Şartname)
    # =========================================================================

    # Oturum başına toplam kare sayısı (şartnameye göre 2250)
    MAX_FRAMES: int = 2250
