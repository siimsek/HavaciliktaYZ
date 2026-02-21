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
    # yolov8m = Medium model (25.9M param, mAP50: 50.2)
    # s modelden ~%12 daha doğru — yarışma hızı (7.5 FPS) buna izin verir
    MODEL_PATH: str = str(PROJECT_ROOT / "models" / "yolov8m.pt")

    # Tespit Güven Eşiği (0.0 - 1.0)
    # 0.20 = Çatı/Bina gibi yanlış pozitifleri azaltır
    CONFIDENCE_THRESHOLD: float = 0.20

    # NMS IoU Eşiği (Non-Maximum Suppression)
    # Daha düşük (0.35) = daha agresif bastırma → çift tespitleri (kaput/tampon) birleştirir
    NMS_IOU_THRESHOLD: float = 0.35

    # Cihaz Seçimi (cuda: GPU, cpu: İşlemci)
    DEVICE: str = "cuda"

    # FP16 Yarı Hassasiyet — RTX 3060'ta ~%40 hız artışı sağlar
    HALF_PRECISION: bool = True

    # Inference Çözünürlüğü (piksel)
    # 1280 = drone görüntüleri için en iyi (uzaktaki insanlar/araçlar)
    # Yarışma offline — gerçek zamanlı hız kısıtı yok, kalite öncelikli
    INFERENCE_SIZE: int = 1280

    # Sınıflar arası NMS — aynı bölgede farklı sınıf çakışmalarını da bastırır
    AGNOSTIC_NMS: bool = True

    # Maksimum tespit sayısı (SAHI ile daha fazla sonuç gelir)
    MAX_DETECTIONS: int = 300

    # Test-Time Augmentation (çoklu ölçekte inference → mAP artışı)
    # Tepeden görünümde (dikey) ve farklı açılarda tespiti iyileştirir
    AUGMENTED_INFERENCE: bool = True

    # Ön-İşleme: CLAHE Kontrast İyileştirme (drone görüntülerinde
    # karanlık/düşük kontrastlı bölgelerdeki nesneleri ortaya çıkarır)
    CLAHE_ENABLED: bool = True
    CLAHE_CLIP_LIMIT: float = 2.0
    CLAHE_TILE_SIZE: int = 8

    # Minimum bbox boyutu (piksel) — altındakiler false positive sayılır
    # 10px = Uzaktaki insan için alt sınır. Daha küçüğü (kol/bacak) elenir.
    MIN_BBOX_SIZE: int = 10

    # Maksimum bbox boyutu (piksel) — üstündekiler bina/çatı vs. sayılır
    # 50m irtifada: Otobüs ~200px. Bina çatısı > 300px.
    MAX_BBOX_SIZE: int = 300

    # =========================================================================
    #  SAHI (Slicing Aided Hyper Inference) — Tepeden Görünüm İyileştirmesi
    # =========================================================================
    # Görüntüyü örtüşen parçalara böler → her parçada ayrı inference → birleştir
    # Küçük nesneleri (tepeden araç/insan) dramatik şekilde iyileştirir
    SAHI_ENABLED: bool = True
    SAHI_SLICE_SIZE: int = 640       # Her parçanın boyutu (piksel)
    SAHI_OVERLAP_RATIO: float = 0.35 # Parçalar arası örtüşme (%35) - Kenarları yakalar
    SAHI_MERGE_IOU: float = 0.35     # Birleştirme NMS IoU eşiği (Çift tespitleri önler)

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

    # VisDrone → TEKNOFEST Sınıf Eşleştirme Tablosu
    # VisDrone sınıfları:
    #   0=ignored, 1=pedestrian, 2=people, 3=bicycle, 4=car, 5=van,
    #   6=truck, 7=tricycle, 8=awning-tricycle, 9=bus, 10=motor, 11=others
    VISDRONE_TO_TEKNOFEST: dict = {
        1: 1,     # pedestrian → İnsan
        2: 1,     # people → İnsan
        3: 0,     # bicycle → Taşıt
        4: 0,     # car → Taşıt
        5: 0,     # van → Taşıt
        6: 0,     # truck → Taşıt
        7: 0,     # tricycle → Taşıt
        8: 0,     # awning-tricycle → Taşıt
        9: 0,     # bus → Taşıt
        10: 0,    # motor → Taşıt
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

    # Simülasyon Test Görseli (eski statik mod için)
    SIMULATION_IMAGE_PATH: str = str(PROJECT_ROOT / "bus.jpg")

    # Veri Seti Dizini
    DATASETS_DIR: str = str(PROJECT_ROOT / "datasets")

    # Simülasyon DET modu: rastgele seçilecek fotoğraf sayısı
    SIMULATION_DET_SAMPLE_SIZE: int = 100

    # =========================================================================
    #  PERFORMANS AYARLARI
    # =========================================================================

    # FPS Raporlama Aralığı (her N karede bir)
    FPS_REPORT_INTERVAL: int = 10

    # Ana Döngü Bekleme Süresi (saniye) - 0 = maksimum hız
    LOOP_DELAY: float = 0.0

    # GPU Bellek Temizleme Aralığı (her N karede bir)
    GPU_CLEANUP_INTERVAL: int = 200

    # Debug görsel kaydetme aralığı (her N karede diske yaz)
    DEBUG_SAVE_INTERVAL: int = 50

    # Taşıt hareketlilik kestirimi (movement_status) parametreleri
    MOVEMENT_WINDOW_FRAMES: int = 24
    MOVEMENT_MIN_HISTORY: int = 6
    MOVEMENT_THRESHOLD_PX: float = 12.0
    MOVEMENT_MATCH_DISTANCE_PX: float = 80.0
    MOVEMENT_MAX_MISSED_FRAMES: int = 8

    # =========================================================================
    #  YARIŞMA LİMİTLERİ (Şartname)
    # =========================================================================

    # Oturum başına toplam kare sayısı (şartnameye göre 2250)
    MAX_FRAMES: int = 2250
