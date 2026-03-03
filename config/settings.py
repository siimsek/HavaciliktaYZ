"""Merkezi konfigürasyon. Yarışma günü sadece bu dosya güncellenir.
BASE_URL, TEAM_NAME, kamera parametreleri buradan güncellenir."""

from pathlib import Path
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
    NMS_IOU_THRESHOLD: float = 0.15  # Çakışan kutuları bastırma eşiği
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
    MAX_BBOX_SIZE: int = 9999

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
    TASK3_FEATURE_METHOD: str = "ORB"
    TASK3_DUPLICATE_DEGRADE_RATIO: float = 0.50
    TASK3_DUPLICATE_DEGRADE_MIN_COUNT: int = 3

    # Sınıflar (şartname)
    CLASS_TASIT: int = 0  # Taşıt
    CLASS_INSAN: int = 1  # İnsan
    CLASS_UAP: int = 2  # Uçan Araba Park Alanı
    CLASS_UAI: int = 3  # Uçan Ambulans İniş Alanı

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

    LANDING_ZONE_CONTAINMENT_IOU: float = 0.70

    # Kamera (kalibrasyon)
    FOCAL_LENGTH_PX: float = 800.0

    # Görüntü Merkez Noktası (piksel)
    CAMERA_CX: float = 960.0
    CAMERA_CY: float = 540.0
    DEFAULT_ALTITUDE: float = 50.0

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
    DEGRADE_FETCH_ONLY_ENABLED: bool = True
    DEGRADE_SEND_INTERVAL_FRAMES: int = 5

    # Dosya yolları
    LOG_DIR: str = os.path.join(str(PROJECT_ROOT), "logs")
    TEMP_FRAME_PATH: str = os.path.join(str(PROJECT_ROOT), "temp_frame.jpg")
    DEBUG_OUTPUT_DIR: str = os.path.join(str(PROJECT_ROOT), "debug_output")
    DATASETS_DIR: str = os.path.join(str(PROJECT_ROOT), "datasets")
    # datasets/ recursive taranır, sadece uzantıya göre eşleşen dosyalar alınır
    IMAGE_EXTENSIONS: tuple = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    SIMULATION_DET_SAMPLE_SIZE: int = 100

    # Performans
    FPS_REPORT_INTERVAL: int = 10
    COMPETITION_RESULT_LOG_INTERVAL: int = 10
    LOOP_DELAY: float = 0.0
    GPU_CLEANUP_INTERVAL: int = 200
    DEBUG_SAVE_INTERVAL: int = 50
    ENABLE_JSON_LOGGING: bool = True
    JSON_LOG_EVERY_N_FRAMES: int = 10
    LOG_MAX_FILES: int = 2000
    DETERMINISM_SEED: int = 42
    DETERMINISM_CPU_THREADS: int = 1
    MOTION_FIELD_NAME: str = "motion_status"
    PAYLOAD_CLS_AS_INT: bool = False

    # Hareket (movement_status)
    MOVEMENT_WINDOW_FRAMES: int = 24
    MOVEMENT_MIN_HISTORY: int = 6
    MOVEMENT_THRESHOLD_PX: float = 12.0
    MOVEMENT_MATCH_DISTANCE_PX: float = 80.0
    MOVEMENT_MAX_MISSED_FRAMES: int = 8
    MOVEMENT_THRESHOLD_REF_WIDTH: int = 1920
    FROZEN_FRAME_DIFF_THRESHOLD: float = 1.0
    MOTION_COMP_ENABLED: bool = True
    MOTION_COMP_MIN_FEATURES: int = 40
    MOTION_COMP_MAX_CORNERS: int = 200
    MOTION_COMP_QUALITY_LEVEL: float = 0.01
    MOTION_COMP_MIN_DISTANCE: int = 20
    MOTION_COMP_WIN_SIZE: int = 21

    # GPS=0 latency compensation (feature flag)
    LATENCY_COMP_ENABLED: bool = True
    LATENCY_COMP_MAX_MS: float = 120.0
    LATENCY_COMP_MAX_DELTA_M: float = 0.3
    LATENCY_COMP_EMA_ALPHA: float = 0.35

    # Yarışma limitleri
    MAX_FRAMES: int = 2250
    RESULT_MAX_OBJECTS: int = 100
    RESULT_CLASS_QUOTA: dict = {
        "0": 40,
        "1": 40,
        "2": 10,
        "3": 10,
    }


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
