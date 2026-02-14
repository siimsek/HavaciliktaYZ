"""
TEKNOFEST Havacılıkta Yapay Zeka - Nesne Tespit Modülü (Görev 1)
=================================================================
YOLOv8 tabanlı nesne tespiti ve İniş Uygunluğu (Landing Status) mantığı.

İniş Uygunluğu Algoritması (teknofest_context.md'den):
    1. Taşıt (0) ve İnsan (1) → landing_status = "-1" (iniş alanı değil)
    2. UAP (2) ve UAİ (3) için:
       a. Bounding box kadrajın kenarına değiyorsa → "0" (alan tam görünmüyor)
       b. Alan tamamen kadrajda VE üzerinde İnsan/Taşıt varsa (kesişim kontrolü) → "0"
       c. Alan tamamen kadrajda VE üzerinde engel yoksa → "1" (uygun)

Kullanım:
    from src.detection import ObjectDetector
    detector = ObjectDetector()
    detections = detector.detect(frame)
"""

import os
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from ultralytics import YOLO

from config.settings import Settings
from src.utils import Logger


class ObjectDetector:
    """
    YOLOv8 tabanlı nesne tespit sınıfı.

    CUDA üzerinde çalışır, COCO sınıflarını TEKNOFEST sınıflarına dönüştürür
    ve UAP/UAİ alanları için İniş Uygunluğu hesabı yapar.

    Attributes:
        model: YOLOv8 model nesnesi.
        device: Kullanılan cihaz ('cuda' veya 'cpu').
        log: Logger nesnesi.
    """

    def __init__(self) -> None:
        """
        YOLOv8 modelini yükler ve CUDA cihazına taşır.

        Model dosyası Settings.MODEL_PATH'ten yerel diskten yüklenir.
        CUDA kullanılamıyorsa CPU'ya düşer ve uyarı verir.
        İlk kare gecikmesini önlemek için warmup inference yapılır.
        """
        self.log = Logger("Detector")
        self._frame_count: int = 0
        self._use_half: bool = False

        # Cihaz Seçimi
        if Settings.DEVICE == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            self.log.success(f"GPU aktif: {gpu_name} ({gpu_mem:.1f} GB)")
        else:
            self.device = "cpu"
            self.log.warn("CUDA bulunamadı! CPU modunda çalışılıyor (yavaş olacak)")

        # Model Yükleme (Yerel Diskten - OFFLINE MODE)
        self.log.info(f"YOLOv8 modeli yükleniyor: {Settings.MODEL_PATH}")
        try:
            if not os.path.exists(Settings.MODEL_PATH):
                raise FileNotFoundError(
                    f"Model dosyası bulunamadı: {Settings.MODEL_PATH}\n"
                    f"  → 'models/' dizinine yolov8n.pt dosyasını kopyalayın."
                )
            self.model = YOLO(Settings.MODEL_PATH)
            self.model.to(self.device)

            # FP16 Yarı Hassasiyet — GPU'da ~%40 hız artışı
            if self.device == "cuda" and Settings.HALF_PRECISION:
                self._use_half = True
                self.log.info("FP16 (Half Precision) aktif — hız optimizasyonu ✓")

            self.log.success("Model başarıyla yüklendi ✓")
        except Exception as e:
            self.log.error(f"Model yükleme hatası: {e}")
            raise RuntimeError(f"YOLOv8 modeli yüklenemedi: {e}")

        # Warmup — ilk kare gecikmesini önle
        self._warmup()

    def _warmup(self) -> None:
        """
        Model ısınması — GPU belleğini hazırlar ve ilk kare gecikmesini önler.

        Dummy (boş) bir tensor ile birkaç inference yaparak CUDA kernel'larını
        ve cuDNN autotuner'ı önceden başlatır.
        """
        self.log.info(f"Model ısınması başlıyor ({Settings.WARMUP_ITERATIONS} iterasyon)...")
        try:
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            with torch.no_grad():
                for i in range(Settings.WARMUP_ITERATIONS):
                    self.model.predict(
                        source=dummy,
                        imgsz=Settings.INFERENCE_SIZE,
                        conf=Settings.CONFIDENCE_THRESHOLD,
                        device=self.device,
                        verbose=False,
                        save=False,
                        half=self._use_half,
                    )
            self.log.success(f"Model ısınması tamamlandı ✓")
        except Exception as e:
            self.log.warn(f"Warmup sırasında hata (görmezden geliniyor): {e}")

    # =========================================================================
    #  ANA TESPİT FONKSİYONU
    # =========================================================================

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Bir video karesinde nesne tespiti yapar ve sonuçları TEKNOFEST formatında döndürür.

        İşlem Sırası:
            1. YOLOv8 ile inference (FP16 opsiyonel)
            2. COCO → TEKNOFEST sınıf dönüşümü
            3. İniş Uygunluğu (Landing Status) hesaplaması
            4. TEKNOFEST JSON formatında sonuç üretimi

        Args:
            frame: BGR formatlı OpenCV görüntüsü (numpy array).

        Returns:
            Tespit edilen nesnelerin listesi. Her nesne dict formatında:
            {
                "cls": "0",            # Sınıf ID (string)
                "landing_status": "-1", # İniş durumu
                "top_left_x": 100,     # Sol üst X (piksel)
                "top_left_y": 200,     # Sol üst Y (piksel)
                "bottom_right_x": 300, # Sağ alt X (piksel)
                "bottom_right_y": 400, # Sağ alt Y (piksel)
                "confidence": 0.85     # Güven skoru (dahili, sunucuya gönderilmez)
            }
        """
        try:
            # ---- 1) YOLOv8 Inference ----
            with torch.no_grad():
                results = self.model.predict(
                    source=frame,
                    imgsz=Settings.INFERENCE_SIZE,
                    conf=Settings.CONFIDENCE_THRESHOLD,
                    iou=Settings.NMS_IOU_THRESHOLD,
                    device=self.device,
                    verbose=False,
                    save=False,
                    half=self._use_half,
                    agnostic_nms=Settings.AGNOSTIC_NMS,
                    max_det=Settings.MAX_DETECTIONS,
                )

            # ---- 2) COCO → TEKNOFEST Dönüşümü ----
            raw_detections: List[Dict] = []
            for result in results:
                if result.boxes is None:
                    continue

                for box in result.boxes:
                    coco_id: int = int(box.cls[0].item())
                    conf: float = float(box.conf[0].item())
                    coords = box.xyxy[0].tolist()
                    x1: float = float(coords[0])
                    y1: float = float(coords[1])
                    x2: float = float(coords[2])
                    y2: float = float(coords[3])

                    # COCO → TEKNOFEST eşleştirme
                    tf_id = Settings.COCO_TO_TEKNOFEST.get(coco_id, -1)
                    if tf_id == -1:
                        # Yarışma dışı nesne — atla
                        continue

                    raw_detections.append({
                        "cls_int": tf_id,
                        "cls": str(tf_id),
                        "confidence": int(conf * 10000) / 10000,  # 4 ondalık
                        "top_left_x": int(x1),
                        "top_left_y": int(y1),
                        "bottom_right_x": int(x2),
                        "bottom_right_y": int(y2),
                        "bbox": (x1, y1, x2, y2),  # dahili hesaplama için
                    })

            # ---- 3) İniş Uygunluğu Hesaplaması ----
            frame_h, frame_w = frame.shape[:2]
            final_detections = self._determine_landing_status(
                raw_detections, frame_w, frame_h
            )

            # ---- 4) Dahili alanları temizle, TEKNOFEST formatı döndür ----
            output: List[Dict] = []
            for det in final_detections:
                output.append({
                    "cls": det["cls"],
                    "landing_status": det["landing_status"],
                    "top_left_x": det["top_left_x"],
                    "top_left_y": det["top_left_y"],
                    "bottom_right_x": det["bottom_right_x"],
                    "bottom_right_y": det["bottom_right_y"],
                    "confidence": det["confidence"],  # debug için tutuyoruz
                })

            # Debug log — tek geçişte sınıf sayımı (Counter ile)
            if Settings.DEBUG:
                cls_counts = Counter(d["cls"] for d in output)
                self.log.debug(
                    f"Tespit: {len(output)} nesne "
                    f"(Taşıt: {cls_counts.get('0', 0)}, "
                    f"İnsan: {cls_counts.get('1', 0)}, "
                    f"UAP: {cls_counts.get('2', 0)}, "
                    f"UAİ: {cls_counts.get('3', 0)})"
                )

            # ---- GPU Bellek Temizliği (periyodik) ----
            self._frame_count += 1
            if (
                self.device == "cuda"
                and self._frame_count % Settings.GPU_CLEANUP_INTERVAL == 0
            ):
                torch.cuda.empty_cache()
                self.log.debug("GPU bellek temizlendi (empty_cache)")

            return output

        except Exception as e:
            self.log.error(f"Tespit hatası: {e}")
            # Sistem çökmemeli — boş liste döndür
            return []

    # =========================================================================
    #  İNİŞ UYGUNLUĞU (LANDING STATUS) MANTIĞI
    # =========================================================================

    def _determine_landing_status(
        self,
        detections: List[Dict],
        frame_w: int,
        frame_h: int,
    ) -> List[Dict]:
        """
        Tespit edilen nesneler için iniş durumunu belirler.

        Mantık (teknofest_context.md - Bölüm 4.6):
            - Taşıt (0) ve İnsan (1): landing_status = "-1" (sabit)
            - UAP (2) ve UAİ (3):
                a) bbox kadrajın kenarına değiyorsa → "0" (alan tam görünmüyor)
                b) Üzerinde taşıt/insan varsa (kesişim kontrolü) → "0"
                c) Yukarıdaki koşullar sağlanmıyorsa → "1" (uygun)

        ÖNEMLİ: Engel kontrolünde IoU değil "intersection-over-area-of-landing-zone"
        kullanılır. Çünkü şartname "alanın **üzerinde** nesne var mı" soruyor —
        küçük bir insan büyük iniş alanının üzerinde olsa IoU çok düşük çıkar
        ama yine de uygun değildir.

        Args:
            detections: Ham tespit listesi (bbox alanı dahil).
            frame_w: Görüntü genişliği (piksel).
            frame_h: Görüntü yüksekliği (piksel).

        Returns:
            landing_status alanı doldurulmuş tespit listesi.
        """
        # Taşıt ve İnsan listesini ayır (engel olarak kullanılacak)
        obstacles: List[Tuple[float, float, float, float]] = []
        for det in detections:
            if det["cls_int"] in (Settings.CLASS_TASIT, Settings.CLASS_INSAN):
                obstacles.append(det["bbox"])

        # Her nesne için iniş durumunu belirle
        for det in detections:
            cls_id = det["cls_int"]

            if cls_id in (Settings.CLASS_TASIT, Settings.CLASS_INSAN):
                # ------ Taşıt veya İnsan: İniş alanı değil ------
                det["landing_status"] = Settings.LANDING_NOT_AREA

            elif cls_id in (Settings.CLASS_UAP, Settings.CLASS_UAI):
                # ------ UAP veya UAİ: İniş uygunluğu hesapla ------
                bbox = det["bbox"]

                # (a) Alan kadrajın kenarına değiyor mu?
                if self._is_touching_edge(bbox, frame_w, frame_h):
                    det["landing_status"] = Settings.LANDING_NOT_SUITABLE
                    self.log.debug(
                        f"  UAP/UAİ kenar temas → uygun değil"
                    )
                    continue

                # (b) Üzerinde engel var mı? (Kesişim kontrolü)
                has_obstacle = False
                for obs_bbox in obstacles:
                    overlap_ratio = self._intersection_over_area(bbox, obs_bbox)
                    if overlap_ratio > Settings.LANDING_IOU_THRESHOLD:
                        has_obstacle = True
                        self.log.debug(
                            f"  UAP/UAİ üzerinde engel (overlap={overlap_ratio:.3f}) → uygun değil"
                        )
                        break

                if has_obstacle:
                    det["landing_status"] = Settings.LANDING_NOT_SUITABLE
                else:
                    # (c) Alan tamamen kadrajda ve engelsiz → uygun
                    det["landing_status"] = Settings.LANDING_SUITABLE
                    self.log.debug("  UAP/UAİ → iniş uygun ✓")

            else:
                det["landing_status"] = Settings.LANDING_NOT_AREA

        return detections

    # =========================================================================
    #  KESİŞİM HESAPLAMA
    # =========================================================================

    @staticmethod
    def _intersection_over_area(
        landing_box: Tuple[float, float, float, float],
        obstacle_box: Tuple[float, float, float, float],
    ) -> float:
        """
        İniş alanı ile engel arasındaki kesişimin, iniş alanına oranını hesaplar.

        Bu, standart IoU'dan farklıdır: küçük bir engel (insan) büyük bir iniş
        alanının üzerinde olsa bile IoU çok düşük çıkar. Ancak bu hesaplama
        iniş alanının ne kadarının engelle örtüştüğünü doğru ölçer.

        Formül:
            overlap_ratio = intersection_area / landing_area

        Args:
            landing_box: İniş alanı (x1, y1, x2, y2).
            obstacle_box: Engel nesnesi (x1, y1, x2, y2).

        Returns:
            0.0 ile 1.0 arasında kesişim oranı.
        """
        # Kesişim alanının köşeleri
        inter_x1 = max(landing_box[0], obstacle_box[0])
        inter_y1 = max(landing_box[1], obstacle_box[1])
        inter_x2 = min(landing_box[2], obstacle_box[2])
        inter_y2 = min(landing_box[3], obstacle_box[3])

        # Kesişim alanı
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        if inter_area == 0:
            return 0.0

        # İniş alanının toplam alanı
        landing_area = (
            max(0.0, landing_box[2] - landing_box[0])
            * max(0.0, landing_box[3] - landing_box[1])
        )

        if landing_area == 0:
            return 0.0

        return inter_area / landing_area


    # =========================================================================
    #  KENAR TEMAS KONTROLÜ
    # =========================================================================

    @staticmethod
    def _is_touching_edge(
        bbox: Tuple[float, float, float, float],
        frame_w: int,
        frame_h: int,
        margin: int = 5,
    ) -> bool:
        """
        Bounding box'ın kadrajın kenarına değip değmediğini kontrol eder.

        Şartnameye göre: UAP/UAİ alanının tamamı kadraj içinde olmalıdır,
        aksi halde iniş durumu "uygun" olamaz.

        Args:
            bbox: (x1, y1, x2, y2) formatında bounding box.
            frame_w: Görüntü genişliği.
            frame_h: Görüntü yüksekliği.
            margin: Kenar toleransı (piksel).

        Returns:
            Kenarına değiyorsa True.
        """
        x1, y1, x2, y2 = bbox
        return (
            x1 <= margin
            or y1 <= margin
            or x2 >= (frame_w - margin)
            or y2 >= (frame_h - margin)
        )
