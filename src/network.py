"""
TEKNOFEST Havacılıkta Yapay Zeka - Ağ İletişim Katmanı (Network Layer)
=======================================================================
Sunucu ile tüm HTTP iletişimini yöneten sınıf.
Retry mekanizması, hata yönetimi ve simülasyon modu desteği içerir.

Kullanım:
    from src.network import NetworkManager
    net = NetworkManager()
    net.start_session()
    frame_data = net.get_frame()
    net.send_result(frame_id, detected_objects, detected_translation)
"""

import time
from typing import Dict, List, Optional, Any

import cv2
import numpy as np
import requests

from config.settings import Settings
from src.utils import Logger, log_json_to_disk


class NetworkManager:
    """
    Sunucu ile iletişimi yöneten ana sınıf.

    Özellikler:
        - Otomatik retry mekanizması (bağlantı koparsa tekrar dener)
        - Hatalı JSON paketlerini filtreler
        - Simülasyon modunda yerel dosyadan test verisi döndürür
        - Gelen/giden tüm verileri diske loglar

    Args:
        base_url: Sunucu adresi (varsayılan: Settings.BASE_URL)
    """

    def __init__(self, base_url: Optional[str] = None) -> None:
        self.base_url = base_url or Settings.BASE_URL
        self.log = Logger("Network")
        self.session = requests.Session()
        self._frame_counter: int = 0

        # Simülasyon görseli cache (her seferinde diskten okumamak için)
        self._sim_image_cache: Optional[np.ndarray] = None

    # =========================================================================
    #  OTURUM YÖNETİMİ
    # =========================================================================

    def start_session(self) -> bool:
        """
        Sunucu ile oturum başlatır.

        Simülasyon modunda her zaman True döner.
        Gerçek modda sunucuya ping atarak bağlantıyı doğrular.

        Returns:
            Bağlantı başarılı ise True, değilse False.
        """
        if Settings.SIMULATION_MODE:
            self.log.success(
                f"[SİMÜLASYON] Oturum başlatıldı → {self.base_url}"
            )
            return True

        # Gerçek mod: Sunucuya ulaşılıyor mu kontrolü
        for attempt in range(1, Settings.MAX_RETRIES + 1):
            try:
                self.log.info(
                    f"Sunucuya bağlanılıyor... (Deneme {attempt}/{Settings.MAX_RETRIES})"
                )
                response = self.session.get(
                    self.base_url, timeout=Settings.REQUEST_TIMEOUT
                )
                if response.status_code == 200:
                    self.log.success(f"Sunucuya bağlantı başarılı → {self.base_url}")
                    return True
                else:
                    self.log.warn(f"Sunucu yanıtı beklenmeyen: {response.status_code}")
            except requests.ConnectionError:
                self.log.error(
                    f"Bağlantı hatası! {Settings.RETRY_DELAY}s sonra tekrar denenecek..."
                )
            except requests.Timeout:
                self.log.error(
                    f"Zaman aşımı! {Settings.RETRY_DELAY}s sonra tekrar denenecek..."
                )
            except Exception as e:
                self.log.error(f"Beklenmeyen hata: {e}")

            time.sleep(Settings.RETRY_DELAY)

        self.log.error("Sunucuya bağlanılamadı! Tüm denemeler tükendi.")
        return False

    # =========================================================================
    #  KARE ÇEKME (GET FRAME)
    # =========================================================================

    def get_frame(self) -> Optional[Dict[str, Any]]:
        """
        Sunucudan bir sonraki video karesinin meta verisini çeker.

        Simülasyon modunda sabit bir test verisi döndürür.
        Gerçek modda sunucunun /next_frame endpoint'inden JSON alır.

        Returns:
            Kare verisi dict veya hata durumunda None.
        """
        if Settings.SIMULATION_MODE:
            return self._get_simulation_frame()

        url = f"{self.base_url}{Settings.ENDPOINT_NEXT_FRAME}"

        for attempt in range(1, Settings.MAX_RETRIES + 1):
            try:
                response = self.session.get(url, timeout=Settings.REQUEST_TIMEOUT)

                if response.status_code == 200:
                    data = response.json()

                    # Gelen veriyi doğrula
                    if not self._validate_frame_data(data):
                        self.log.warn("Geçersiz kare verisi alındı, atlanıyor")
                        return None

                    # Gelen veriyi logla
                    log_json_to_disk(data, direction="incoming", tag=f"frame_{self._frame_counter}")
                    self._frame_counter += 1

                    return data

                elif response.status_code == 204:
                    # Video bitti — daha fazla kare yok
                    self.log.info("Video sona erdi (204 No Content)")
                    return None
                else:
                    self.log.warn(
                        f"Beklenmeyen sunucu yanıtı: {response.status_code}"
                    )
                    return None

            except requests.ConnectionError:
                self.log.error(
                    f"Kare çekme hatası! Deneme {attempt}/{Settings.MAX_RETRIES}"
                )
            except requests.Timeout:
                self.log.error(f"Kare çekme zaman aşımı! Deneme {attempt}/{Settings.MAX_RETRIES}")
            except ValueError as e:
                self.log.error(f"JSON parse hatası: {e}")
                return None
            except Exception as e:
                self.log.error(f"Beklenmeyen hata (get_frame): {e}")
                return None

            time.sleep(Settings.RETRY_DELAY)

        self.log.error("Kare çekilemedi — tüm denemeler tükendi")
        return None

    # =========================================================================
    #  GÖRÜNTÜ İNDİRME
    # =========================================================================

    def download_image(self, frame_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Sunucudan veya yerel diskten görüntüyü indirir ve numpy array olarak döndürür.

        Dosyaya yazmak yerine doğrudan bellekte işler (daha hızlı).

        Args:
            frame_data: get_frame() ile alınan kare meta verisi.

        Returns:
            BGR formatlı numpy görüntü dizisi veya hata durumunda None.
        """
        if Settings.SIMULATION_MODE:
            return self._load_simulation_image()

        # Görüntü URL'sini oluştur
        frame_url = frame_data.get("frame_url", "")
        if not frame_url:
            # Alternatif alan adları dene
            frame_url = frame_data.get("image_url", "")

        if not frame_url:
            self.log.error("Kare verisinde görüntü URL'si bulunamadı")
            return None

        # Tam URL oluştur
        if frame_url.startswith("http"):
            full_url = frame_url
        else:
            full_url = f"{self.base_url}{frame_url}"

        for attempt in range(1, Settings.MAX_RETRIES + 1):
            try:
                response = self.session.get(
                    full_url,
                    timeout=Settings.REQUEST_TIMEOUT,
                )

                if response.status_code == 200:
                    # Byte dizisini numpy array'e çevir (bellekte)
                    img_array = np.frombuffer(response.content, dtype=np.uint8)
                    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                    if frame is None:
                        self.log.error("Görüntü decode edilemedi!")
                        return None

                    self.log.debug(
                        f"Görüntü indirildi: {frame.shape[1]}x{frame.shape[0]}"
                    )
                    return frame
                else:
                    self.log.warn(f"Görüntü indirilemedi: HTTP {response.status_code}")

            except requests.Timeout:
                self.log.error(
                    f"Görüntü indirme zaman aşımı! Deneme {attempt}/{Settings.MAX_RETRIES}"
                )
            except Exception as e:
                self.log.error(f"Görüntü indirme hatası: {e}")

            time.sleep(Settings.RETRY_DELAY)

        self.log.error("Görüntü indirilemedi — tüm denemeler tükendi")
        return None

    # =========================================================================
    #  SONUÇ GÖNDERME
    # =========================================================================

    def send_result(
        self,
        frame_id: Any,
        detected_objects: List[Dict],
        detected_translation: Dict[str, float],
    ) -> bool:
        """
        Tespit ve konum sonuçlarını sunucuya gönderir.

        JSON formatı teknofest_context.md şartnamesine uygun:
        {
            "frame": <frame_id>,
            "detected_objects": [...],
            "detected_translations": [...]
        }

        ÖNEMLİ: 'confidence' alanı sunucuya gönderilmez — şartnamede yok.

        Args:
            frame_id: Kare kimliği (sunucu tarafından verilen).
            detected_objects: Tespit edilen nesnelerin listesi.
            detected_translation: Pozisyon kestirimi sonucu {x, y, z}.

        Returns:
            Gönderim başarılı ise True.
        """
        # 'confidence' alanını çıkar — şartnameye uygun JSON oluştur
        clean_objects: List[Dict] = [
            {
                "cls": obj["cls"],
                "landing_status": obj["landing_status"],
                "top_left_x": obj["top_left_x"],
                "top_left_y": obj["top_left_y"],
                "bottom_right_x": obj["bottom_right_x"],
                "bottom_right_y": obj["bottom_right_y"],
            }
            for obj in detected_objects
        ]

        # Sonuç paketini oluştur (Şartname formatı)
        payload: Dict[str, Any] = {
            "frame": frame_id,
            "detected_objects": clean_objects,
            "detected_translations": [detected_translation],
        }

        # JSON logla
        log_json_to_disk(payload, direction="outgoing", tag=f"result_{frame_id}")

        if Settings.SIMULATION_MODE:
            self.log.success(
                f"[SİMÜLASYON] Sonuç gönderildi → Frame: {frame_id} | "
                f"Nesne: {len(clean_objects)} adet"
            )
            return True

        url = f"{self.base_url}{Settings.ENDPOINT_SUBMIT_RESULT}"

        for attempt in range(1, Settings.MAX_RETRIES + 1):
            try:
                response = self.session.post(
                    url,
                    json=payload,
                    timeout=Settings.REQUEST_TIMEOUT,
                    headers={"Content-Type": "application/json"},
                )

                if response.status_code == 200:
                    self.log.debug(f"Sonuç başarıyla gönderildi: Frame {frame_id}")
                    return True
                else:
                    self.log.warn(
                        f"Sonuç gönderimi yanıtı: HTTP {response.status_code}"
                    )
                    return False

            except requests.Timeout:
                self.log.error(
                    f"Sonuç gönderme zaman aşımı! Deneme {attempt}/{Settings.MAX_RETRIES}"
                )
            except Exception as e:
                self.log.error(f"Sonuç gönderme hatası: {e}")

            time.sleep(Settings.RETRY_DELAY)

        self.log.error("Sonuç gönderilemedi — tüm denemeler tükendi")
        return False

    # =========================================================================
    #  YARDIMCI METODLAR (PRİVATE)
    # =========================================================================

    def _get_simulation_frame(self) -> Dict[str, Any]:
        """
        Simülasyon modu için sahte kare verisi üretir.

        Her çağrıda frame_counter artar, böylece her seferinde
        yeni bir kare işleniyormuş gibi görünür.
        """
        frame_id = self._frame_counter
        self._frame_counter += 1

        return {
            "frame_url": Settings.SIMULATION_IMAGE_PATH,
            "frame_id": frame_id,
            "video_name": "simulation_video",
            "translation_x": 0.0,
            "translation_y": 0.0,
            "translation_z": 50.0,
            "gps_health": 1,
        }

    def _load_simulation_image(self) -> Optional[np.ndarray]:
        """
        Yerel test görselini yükler (bus.jpg).

        İlk yüklemeden sonra belleğe cache'ler — her karede
        diskten okuma yapmaz (önemli performans iyileştirmesi).
        """
        # Cache varsa doğrudan döndür — YOLO ve cvtColor kendi kopyalarını
        # oluşturur, bu yüzden .copy() gereksiz (bellek+CPU tasarrufu)
        if self._sim_image_cache is not None:
            return self._sim_image_cache

        img_path = Settings.SIMULATION_IMAGE_PATH
        frame = cv2.imread(img_path)

        if frame is None:
            self.log.error(f"Simülasyon görseli yüklenemedi: {img_path}")
            return None

        # Cache'e kaydet
        self._sim_image_cache = frame
        self.log.debug(f"Simülasyon görseli yüklendi ve cache'lendi: {frame.shape[1]}x{frame.shape[0]}")
        return frame

    def _validate_frame_data(self, data: Dict[str, Any]) -> bool:
        """
        Sunucudan gelen kare verisinin zorunlu alanlarını kontrol eder.

        Args:
            data: Sunucudan gelen JSON verisi.

        Returns:
            Veri geçerli ise True.
        """
        required_fields = ["frame_id"]
        for field in required_fields:
            if field not in data:
                self.log.warn(f"Eksik alan: '{field}' kare verisinde bulunamadı")
                return False
        return True
