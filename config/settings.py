"""Merkezi konfigürasyon. Yarışma günü sadece bu dosya güncellenir.
BASE_URL, TEAM_NAME, kamera parametreleri buradan güncellenir."""

from pathlib import Path
from typing import Optional
import os


PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


class Settings:
    """Merkezi sistem ayarları."""

    # Yarışma günü
    BASE_URL: str = "http://127.0.0.1:5000"
    ENDPOINT_NEXT_FRAME: str = "/next_frame"
    ENDPOINT_SUBMIT_RESULT: str = "/submit_result"
    TEAM_NAME: str = "Takim_ID"
    SIMULATION_MODE: bool = True
    DEFAULT_RUNTIME_MODE: str = "visual_validation"
    COMPETITION_RUNTIME_MODE: str = "competition"
    VISUAL_VALIDATION_RUNTIME_MODE: str = "visual_validation"
    DEBUG: bool = True

    # Model
    MODEL_PATH: str = os.path.join(
        str(PROJECT_ROOT), "model", "best_mAP50-0.923_mAP50-95-0.766.pt"
    )
    CONFIDENCE_THRESHOLD: float = (
        0.40  # 0.0-1.0, düşük = daha fazla tespit (noise riski)
    )
    # UAP/UAİ için düşük eşik — model bazen daha düşük conf ile tespit eder
    CONFIDENCE_THRESHOLD_UAP_UAI: Optional[float] = 0.28  # None = global eşik kullan
    NMS_IOU_THRESHOLD: float = 0.15  # Çakışan kutuları bastırma eşiği
    NMS_MODE: str = "class_aware"  # class_aware|agnostic|hybrid
    HYBRID_NMS_IOU_THRESHOLD: float = 0.65
    DEVICE: str = "cuda"
    HALF_PRECISION: bool = True
    INFERENCE_SIZE: int = 1280
    AGNOSTIC_NMS: bool = True
    MAX_DETECTIONS: int = 300
    AUGMENTED_INFERENCE: bool = False

    # Ön-işleme (CLAHE)
    CLAHE_ENABLED: bool = True
    CLAHE_CLIP_LIMIT: float = 2.0
    CLAHE_TILE_SIZE: int = 8
    MIN_BBOX_SIZE: int = 20
    MIN_BBOX_SIZE_FLOOR: int = 8  # Yüksek irtifada min_size bu değerin altına düşmez
    CLASS_ADAPTIVE_FILTERS: dict = {
        "0": {"min_size": 20, "max_size": 600, "max_aspect": 4.5},  # Taşıt (çatıları engeller)
        "1": {"min_size": 15, "max_size": 160, "max_aspect": 4.0},  # İnsan (direk/ağaç engeller)
        "2": {"min_size": 12, "max_size": 1000, "max_aspect": 2.5, "min_floor": 6},  # UAP
        "3": {"min_size": 12, "max_size": 1000, "max_aspect": 2.5, "min_floor": 6},  # UAİ
    }

    # Guardrails (postprocess): overlap, scene consistency, crowd adaptivity
    GUARDRAILS_ENABLED: bool = True
    GUARDRAIL_EXEMPT_CLASSES: tuple = ("2", "3")  # UAP, UAİ — modele güven, bastırma
    GUARDRAIL_OVERLAP_AREA_RATIO: float = 5.0
    GUARDRAIL_OVERLAP_IOU: float = 0.15
    GUARDRAIL_SCENE_OUTLIER_FACTOR: float = 8.0
    GUARDRAIL_SCENE_MIN_SAMPLES: int = 3
    GUARDRAIL_CROWD_THRESHOLD: int = 30
    GUARDRAIL_CROWD_CONF_BOOST: float = 0.15

    # Zamansal tutarlılık (anlık FP bastırma): en az K kare görünen tespitler kabul
    TEMPORAL_FILTER_ENABLED: bool = True
    TEMPORAL_FILTER_MIN_APPEARANCES: int = 2
    TEMPORAL_FILTER_WINDOW_FRAMES: int = 5
    TEMPORAL_FILTER_IOU_THRESHOLD: float = 0.3
    TEMPORAL_FILTER_CONFIDENCE_EXEMPT: float = 0.7
    TEMPORAL_FILTER_EXEMPT_CLASSES: tuple = ("2", "3")  # UAP, UAİ — şartname zorunlu

    # SAHI: Görüntüyü parçalara böl, küçük nesneleri (drone tepeden bakış) yakala
    SAHI_ENABLED: bool = True
    SAHI_SLICE_SIZE: int = 640
    SAHI_OVERLAP_RATIO: float = 0.35
    SAHI_MERGE_IOU: float = 0.25

    WARMUP_ITERATIONS: int = 3

    # Görev 3 (Image Matching)
    TASK3_ENABLED: bool = True
    TASK3_REFERENCE_DIR: str = str(PROJECT_ROOT / "datasets" / "task3_references")
    TASK3_SIMILARITY_THRESHOLD: float = 0.72
    TASK3_FALLBACK_THRESHOLD: float = 0.66
    TASK3_FALLBACK_INTERVAL: int = 5
    TASK3_GRID_STRIDE: int = 32
    TASK3_MAX_REFERENCES: int = 10
    TASK3_REFERENCE_BATCH_SIZE: int = 5
    TASK3_FEATURE_METHOD: str = "ORB"
    TASK3_DUPLICATE_DEGRADE_RATIO: float = 0.50
    TASK3_DUPLICATE_DEGRADE_MIN_COUNT: int = 3
    TASK3_INCLUDE_QUALITY_FIELDS: bool = False
    TASK3_QUALITY_HIGH_THRESHOLD: float = 0.85
    TASK3_QUALITY_MEDIUM_THRESHOLD: float = 0.72
    TASK3_DOMAIN_FALLBACK_ENABLED: bool = True
    TASK3_DOMAIN_FALLBACK_METHOD: str = "AKAZE"
    TASK3_DOMAIN_FALLBACK_THRESHOLD: float = 0.58
    TASK3_DOMAIN_FALLBACK_INTERVAL: int = 3

    # Sınıflar (şartname)
    CLASS_TASIT: int = 0  # Taşıt
    CLASS_INSAN: int = 1  # İnsan
    CLASS_UAP: int = 2  # Uçan Araba Park Alanı
    CLASS_UAI: int = 3  # Uçan Ambulans İniş Alanı

    # Model sınıf eşlemesi manuel override (model ID → TEKNOFEST ID)
    # Eğitilmiş 4 sınıflı model (Taşıt, İnsan, UAP, UAİ) için doğrudan eşleme:
    # CUSTOM_CLASS_MAP = {0: 0, 1: 1, 2: 2, 3: 3}
    # Logda "Model UAP/UAİ sınıfı İÇERMİYOR" görüyorsanız bu ayarı ekleyin.
    CUSTOM_CLASS_MAP: Optional[dict] = None

    # İniş Durumu Kodları
    LANDING_NOT_AREA: str = "-1"  # İniş alanı değil (Taşıt/İnsan)
    LANDING_NOT_SUITABLE: str = "0"  # İniş için uygun değil
    LANDING_SUITABLE: str = "1"

    # COCO → TEKNOFEST eşleme
    COCO_TO_TEKNOFEST: dict = {
        0: 1,  # person → İnsan
        1: 0,  # bicycle → Taşıt (sürücüsüyle birlikte)
        2: 0,  # car → Taşıt
        3: 0,  # motorcycle → Taşıt
        5: 0,  # bus → Taşıt
        6: 0,  # train → Taşıt (vagonlar dahil)
        7: 0,  # truck → Taşıt
        8: 0,  # boat → Taşıt (deniz taşıtı)
    }

    # VisDrone/benzeri format → TEKNOFEST (pedestrian, car, van vb. sınıf ID'leri)
    VISDRONE_TO_TEKNOFEST: dict = {
        1: 1,  # pedestrian → İnsan
        2: 1,  # people → İnsan
        3: 0,  # bicycle → Taşıt
        4: 0,  # car → Taşıt
        5: 0,  # van → Taşıt
        6: 0,  # truck → Taşıt
        7: 0,  # tricycle → Taşıt
        8: 0,  # awning-tricycle → Taşıt
        9: 0,  # bus → Taşıt
        10: 0,  # motor → Taşıt
    }

    LANDING_IOU_THRESHOLD: float = 0.0  # 0 = herhangi kesişim → uygun değil (şartname)
    LANDING_PROXIMITY_MARGIN: float = 0.10  # Perspektif toleransı: bbox %10 genişletme

    EDGE_MARGIN_RATIO: float = 0.004  # UAP/UAİ kadraj kenarına değiyorsa → uygun değil
    UNKNOWN_OBJECTS_AS_OBSTACLES: bool = True
    UAP_CV_VERIFICATION: bool = False  # UAP/UAİ Hough daire doğrulaması

    LANDING_ZONE_CONTAINMENT_IOU: float = 0.70

    # Kamera (kalibrasyon)
    FOCAL_LENGTH_PX: float = 800.0

    # Görüntü Merkez Noktası (piksel)
    CAMERA_CX: float = 960.0
    CAMERA_CY: float = 540.0
    DEFAULT_ALTITUDE: float = 50.0
    CAMERA_CALIBRATION_GUARD_ENABLED: bool = True

    # Ağ
    REQUEST_TIMEOUT: int = 5
    REQUEST_CONNECT_TIMEOUT_SEC: float = 1.5
    REQUEST_READ_TIMEOUT_SEC_FRAME_META: float = 2.5
    REQUEST_READ_TIMEOUT_SEC_IMAGE: float = 4.0
    REQUEST_READ_TIMEOUT_SEC_SUBMIT: float = 3.5
    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 1.0
    BACKOFF_BASE_SEC: float = 0.4
    BACKOFF_MAX_SEC: float = 5.0
    BACKOFF_JITTER_RATIO: float = 0.25
    SEEN_FRAME_LRU_SIZE: int = 512
    IDEMPOTENCY_KEY_PREFIX: str = "aia"

    # Circuit breaker
    CB_TRANSIENT_WINDOW_SEC: float = 30.0
    CB_TRANSIENT_MAX_EVENTS: int = 12
    CB_OPEN_COOLDOWN_SEC: float = 8.0
    CB_MAX_OPEN_CYCLES: int = 6
    CB_SESSION_MAX_TRANSIENT_SEC: float = 120.0
    CB_SESSION_SOFT_TRANSIENT_SEC: float = 90.0
    DEGRADE_FETCH_ONLY_ENABLED: bool = True
    DEGRADE_SEND_INTERVAL_FRAMES: int = 5
    DEGRADE_REPLAY_ENABLED: bool = True
    DEGRADE_REPLAY_MAX_AGE_FRAMES: int = 6
    DEGRADE_REPLAY_MAX_OBJECTS: int = 40
    DEGRADE_FALLBACK_RATIO_WINDOW: int = 40
    DEGRADE_FALLBACK_RATIO_HIGH: float = 0.75
    PERMANENT_REJECT_RETRY_LIMIT: int = 1
    DUPLICATE_STORM_THRESHOLD: int = 5
    DUPLICATE_STORM_ACTION: str = "terminate_session"  # terminate_session|continue

    # Dosya yolları
    LOG_DIR: str = os.path.join(str(PROJECT_ROOT), "logs")
    TEMP_FRAME_PATH: str = os.path.join(str(PROJECT_ROOT), "temp_frame.jpg")
    DEBUG_OUTPUT_DIR: str = os.path.join(str(PROJECT_ROOT), "debug_output")
    DATASETS_DIR: str = os.path.join(str(PROJECT_ROOT), "datasets")
    # datasets/ recursive taranır, sadece uzantıya göre eşleşen dosyalar alınır
    IMAGE_EXTENSIONS: tuple = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    VIDEO_EXTENSIONS: tuple = (".mp4", ".avi", ".mov", ".mkv")
    SIMULATION_DET_SAMPLE_SIZE: int = 100
    # GPS sağlıksızken video duraklatma (Show window modunda SPACE ile devam)
    SIMULATION_PAUSE_ON_GPS_LOSS: bool = False
    # True: Simülasyonda GPS=1 olsa bile gps_health=0 simüle et (görsel odometri her zaman çalışsın)
    SIMULATION_FORCE_GPS_UNHEALTHY: bool = True

    # Performans
    FPS_REPORT_INTERVAL: int = 10
    COMPETITION_RESULT_LOG_INTERVAL: int = 10
    COMPETITION_DEBUG_DRAW_INTERVAL: int = 10
    LOOP_DELAY: float = 0.0
    GPU_CLEANUP_INTERVAL: int = 200
    DEBUG_SAVE_INTERVAL: int = 50
    MAP_MAX_TRAJECTORY_LENGTH: int = 500  # Mini-map trajectory buffer (performans)
    ENABLE_JSON_LOGGING: bool = True
    JSON_LOG_EVERY_N_FRAMES: int = 10
    DYNAMIC_JSON_LOG_INTERVAL_ENABLED: bool = True
    DYNAMIC_JSON_LOG_SLOW_INTERVAL: int = 40
    DYNAMIC_JSON_LOG_MEDIUM_INTERVAL: int = 20
    DYNAMIC_JSON_LOG_FAST_INTERVAL: int = 10
    LOG_MAX_FILES: int = 2000
    LOW_FPS_GUARD_ENABLED: bool = True
    LOW_FPS_GUARD_THRESHOLD: float = 1.0
    LOW_FPS_GUARD_RECOVERY_THRESHOLD: float = 1.4
    LOW_FPS_GUARD_WINDOW: int = 20
    LOW_FPS_GUARD_RECOVERY_STREAK: int = 12
    PROTECTIVE_INFERENCE_SIZE: int = 960
    PROTECTIVE_MAX_DETECTIONS: int = 180
    PROTECTIVE_CONFIDENCE_THRESHOLD: float = 0.50
    PROTECTIVE_LOG_INTERVAL: int = 25
    PROTECTIVE_DEGRADE_SEND_INTERVAL_FRAMES: int = 8
    PROTECTIVE_DISABLE_SAHI: bool = True
    LIGHT_PROFILE_INFERENCE_SIZE: int = 960
    LIGHT_PROFILE_MAX_DETECTIONS: int = 180
    LIGHT_PROFILE_CONFIDENCE_THRESHOLD: float = 0.50
    LIGHT_PROFILE_AUGMENTED_INFERENCE: bool = False
    LIGHT_PROFILE_SAHI_ENABLED: bool = False
    DETERMINISM_SEED: int = 42
    DETERMINISM_CPU_THREADS: int = 1
    MOTION_FIELD_NAME: str = "motion_status"
    PAYLOAD_CLS_AS_INT: bool = False
    PAYLOAD_STATUS_TYPE_PROFILE: str = "int"  # int|string
    PAYLOAD_ADAPTER_VERSION: str = "v1"  # v1|v1_legacy|v2_int

    # Hareket (movement_status)
    MOTION_ALGO: str = "homography"  # flow | homography | iou_tracker
    MOVEMENT_WINDOW_FRAMES: int = 24
    MOVEMENT_MIN_HISTORY: int = 6
    MOVEMENT_THRESHOLD_PX: float = 24.0
    MOVEMENT_EARLY_SIGNAL_RATIO: float = 0.75
    MOVEMENT_HYSTERESIS_RATIO: float = 0.65
    MOVEMENT_MATCH_DISTANCE_PX: float = 80.0
    MOVEMENT_MAX_MISSED_FRAMES: int = 8
    MOVEMENT_THRESHOLD_REF_WIDTH: int = 1920
    MOVEMENT_ADAPTIVE_PAN_ENABLED: bool = True
    MOVEMENT_ADAPTIVE_PAN_PX: float = 15.0  # Ortalama kamera kayması/frame bu değeri aşarsa eşik artar
    MOVEMENT_ADAPTIVE_PAN_FACTOR: float = 1.35  # Büyük pan/tilt'ta eşik çarpanı (1.2–1.5)
    FROZEN_FRAME_DIFF_THRESHOLD: float = 1.0
    MOTION_COMP_ENABLED: bool = True
    MOTION_COMP_MIN_FEATURES: int = 40
    MOTION_COMP_MAX_CORNERS: int = 200
    MOTION_COMP_QUALITY_LEVEL: float = 0.01
    MOTION_COMP_MIN_DISTANCE: int = 20
    MOTION_COMP_WIN_SIZE: int = 21

    # Visual Odometry (GPS=0): piksel→metre işaret düzeltmesi (drift azaltma)
    # İleri gidince haritada geri görünüyorsa VO_SIGN_Y veya VO_SIGN_X'i 1 yapın
    VO_SIGN_X: int = -1  # 1 veya -1; kamera koordinat dönüşümü
    VO_SIGN_Y: int = 1  # 1 veya -1; ileri hareket haritada doğru yön için 1
    # Rotasyon (pan/yaw) tespiti: kamera dönerken pozisyon güncellemesini bastır
    VO_ROTATION_SUPPRESS_ENABLED: bool = True
    VO_ROTATION_DOT_THRESHOLD: float = 0.4  # Bu altında = rotasyon, güncelleme yapma
    VO_MAX_DISPLACEMENT_PER_FRAME: float = 1.0  # Kare başına max metre (drift sınırlama)
    VO_OUTLIER_IQR_FACTOR: float = 1.5  # IQR çarpanı; 0 = devre dışı
    # Şartname: GPS=0'da yalnızca görsel ölçüm kullanılır; ölçüm yoksa pozisyon değiştirme
    GPS_ZERO_POSITION_FREEZE: bool = True  # True = ölçüm yoksa pozisyon dondur

    # GPS=0 latency compensation (feature flag)
    LATENCY_COMP_ENABLED: bool = True
    LATENCY_COMP_MAX_MS: float = 120.0
    LATENCY_COMP_MAX_DELTA_M: float = 0.3
    LATENCY_COMP_EMA_ALPHA: float = 0.35
    GPS_REANCHOR_ALPHA: float = 0.35
    GPS_REANCHOR_MAX_DELTA_M: float = 8.0

    # Yarışma limitleri
    MAX_FRAMES: int = 2250
    RESULT_MAX_OBJECTS: int = 100
    RESULT_CLASS_QUOTA: dict = {
        "0": 40,
        "1": 40,
        "2": 10,
        "3": 10,
    }
    BASE_URL_ALLOWLIST: tuple = ("127.0.0.1", "localhost", "test")


# task3_params.yaml → Görev 3 parametre override
_TASK3_YAML_PATH: Path = Path(__file__).resolve().parent / "task3_params.yaml"
if _TASK3_YAML_PATH.is_file():
    try:
        import yaml

        with open(_TASK3_YAML_PATH, encoding="utf-8") as f:
            _task3_override = yaml.safe_load(f)
        if isinstance(_task3_override, dict):
            if "t_confirm" in _task3_override:
                Settings.TASK3_SIMILARITY_THRESHOLD = float(
                    _task3_override["t_confirm"]
                )
            if "t_fallback" in _task3_override:
                Settings.TASK3_FALLBACK_THRESHOLD = float(_task3_override["t_fallback"])
            if "n_fallback_interval" in _task3_override:
                Settings.TASK3_FALLBACK_INTERVAL = int(
                    _task3_override["n_fallback_interval"]
                )
            if "grid_stride" in _task3_override:
                Settings.TASK3_GRID_STRIDE = int(_task3_override["grid_stride"])
    except (ImportError, OSError, ValueError, TypeError):
        pass
