# TEKNOFEST 2026 Havacılıkta Yapay Zeka Yarışması Sistem Denetim Raporu

## 1. Özet Bulgular
Geliştirilmiş olan yapay zeka sistemi, yarışma şartnamesinin temel iş aksını (HTTP üzerinden kare alımı, sonuçların tespit edilmesi ve JSON olarak gönderilmesi) yüksek bir mühendislik mimarisi ile ele almaktadır. Sistem modüler tutulmuş olup tespit (Detection), yerelleştirme (Localization), hareket izleme (Movement) ve iletişim/direnç (Network/Resilience) kısımları ayrıştırılmıştır. 
Sistem, istenen saniyede 1 FPS işlemini `concurrent.futures` üzerinden asenkron bir `submit`/`fetch` mimarisi kurarak aşmayı hedeflemektedir. İnternet yasağına (Yerel çalışma) uyumlu olup, dinamik port ve IP yönetimine destek vermektedir. Genel anlamda sistem oldukça yetkin olsa da, Görev 2 kapsamındaki saf görsel odometride doğal sınırları sebebiyle puan kaybına (Drift) açık bir yapısı bulunmaktadır.

## 2. Şartname Uyumluluk Analizi

### Görev 1 (Nesne Tespiti)
*   **İniş Uygunluğu (`landing_status`):** Şartnamenin "UAP/UAİ alanının tamamı kare içinde bulunmalıdır" kuralı, `detection.py` içerisinde `_is_touching_edge` ile çözünürlüğe bağlı dinamik margin uygulanarak başarıyla yerine getirilmiştir. Ayrıca "üzerinde taşıt, insan veya başka herhangi bir nesne varsa İniş Durumu = 0" kuralı `_check_obstacle_overlap` kontrolünde kesişim alanı oranlamasıyla güvenceye alınmıştır.
*   **Taşıtların Hareket Durumu (`motion_status`):** `movement.py`'de kamera hareketini (Optical Flow ile `_estimate_camera_shift`) hesaba katarak hedefin gerçek dünya hareketini izole etme çabası vardır. (Şartnameye tam uyumlu analitik yaklaşım).
*   **Çoklu Gönderilen Dörtgen Sayısı (Duplicate BBox):** SAHI uygulandığı için (`_sahi_detect`), örtüşen tespitlerin tekilliği `_merge_detections_nms` fonksiyonu ile idare edilmiştir.

### Görev 2 (Pozisyon Tespiti)
*   İlk 450 karede "Sağlık değeri = 1" kuralı gözetilerek `localization.py` içinde sunucu pozisyonu (XYZ) güvenli kabul ediliyor (`_update_from_gps`).
*   Sağlık değeri sıfırlandığı an, sistem Lucas-Kanade Optik Akış tabanlı (`cv2.calcOpticalFlowPyrLK`) görgüsel odometriye geçiyor ve frame bazlı pixel hareketini anlık irtifa (altitude) / odak uzaklığı (focal length) üzerinden metreye dönüştürmektedir (`_pixel_to_meter`).

### Görev 3 (Görüntü Eşleme)
*   Sunucudan alınan anlık JSON nesnelerinden (base64 veya path üzerinden) `image_matcher.py` aracılığıyla SIFT/ORB feature (anahtar nokta) çıkarımı yapılarak sahada `knnMatch` uygulanmaktadır. Referansların tamamının sahnede olmama ihtimali (fallback/threshold kullanımıyla) şartnameye uygun şekilde denetlenmiştir.

### Operasyon Modları

#### Competition Modu
- **Amaç:** Yarışma süresince sadece gerekli tespit/lokalizasyon çıktılarının sunucuya güvenli şekilde aktarılması.
- **İzinli Özellikler:** Frame fetch/download, inference, contract kontrollü submit, KPI loglama.
- **Çıktılar:** Konsol logları ve JSON submit trafiği; görsel pencere ve debug kare kaydı policy gereği kapalıdır.

#### Visual Validation Modu
- **Amaç:** Yarışma öncesi doğrulama ve operatör gözlemi ile model davranışını hızla incelemek.
- **İzinli Özellikler:** Simülasyon akışı, görsel pencere (`--show`), disk kaydı (`--save`), detaylı debug.
- **Çıktılar:** Konsol logları + gerçek zamanlı anotasyonlu pencere ve/veya `debug_output/` görselleri.

### Rekabet Döngüsü ve Payload Mimarisi
*   Sistem bir kare için yalnızca **1 sonuç** gönderimi limitini idempotency kilitleri ve `assert_contract_ready` gibi veri kontrat güvenceleri ile savunmaktadır. Çökmelere karşı `SessionResilienceController` üzerinden Circuit Breaker mimarisi uygulanmıştır.

---

## 3. Kritik Riskler

### Risk 1: Görev 2'de Drift ve Z Ekseni (İrtifa) Dağılımı Hatası (Hata Sarmalı)
*   **Risk Tanımı:** Salt `cv2.calcOpticalFlowPyrLK` kullanıldığında, ufuk veya denizde featuresuz alanlarda akış kaybolabilir ve rotasyon (yaw/pitch) çevirileri salt X-Y yer değiştirmesi gibi algılanarak çok hızlı akümülatif hataya yol açar.
*   **Şartname İhlali / Puan Etkisi:** Görev 2 toplam değerlendirmenin %40'ıdır. Ortalama 3D Öklid Hatası dramatik şekilde yükseleceğinden doğrudan ciddi puan kaybıdır.
*   **Olasılık:** Yüksek. (Oturumun sonlarında 1-2 dakika GPS koptuğunda rüzgar dalgalanmasından dolayı kamera tilti (açı değişimi) yanılgısı)
*   **Tetikleyici Senaryo:** Rüzgarlı hava, deniz/çöl temalı pistler (köşe noktası / corner bulamama).
*   **Tespit Yöntemi:** Mantıksal çıkarım (`localization.py` içindeki `_update_from_optical_flow` ve `_pixel_to_meter` mantığı rotasyonu (yaw/pitch angle) affine matrix ile dekuple etmemektedir).
*   **Mimari Düzeyde İyileştirme Yönü:** Sadece LK akışı yerine en azından Homography/Affine dönüşüm hesabı (translation+rotation+scale) yapılmalı veya doğrudan bir görsel Odometri/SLAM kütüphanesi entegre edilmelidir.

### Risk 2: İniş Alanı (UAP/UAİ) Tolerans Çakışması (False Positive BBox Kaybı)
*   **Risk Tanımı:** `detection.py`'deki `_check_obstacle_overlap` fonksiyonunda, iniş zeminindeki engelin kesişim alanı, toplam "iniş alanının yüzölçümüne" oranlanmaktadır (`_intersection_over_area`). Fakat bir insan, UAP'nin yanına çok az da temas etse, şartname uyarınca "Uygun Değil (0)" basılmalıdır.
*   **Şartname İhlali / Puan Etkisi:** UAP veya UAİ objelerinde hatalı `landing_status` gönderimi. İlgili sınıfın mAP AP puanını düşürür.
*   **Olasılık:** Orta.
*   **Tetikleyici Senaryo:** UAP/UAİ'nin köşesinde duran küçük bir nesne/insan. Alan (yüzölçüm) çok küçük kaldığından oran threshold'u aşmayabilir.
*   **Tespit Yöntemi:** Mantıksal çıkarım (`detection.py: `_check_obstacle_overlap` ile Threshold kıyası).
*   **Mimari Düzeyde İyileştirme Yönü:** Nesne "Taşıt" veya "İnsan" ise ve `_intersection_over_area` > `0` (sıfırdan mutlak büyük) ise, threshold filtresi uygulanmaksızın anında engel sayılmalıdır.

---

## 4. Orta ve Düşük Öncelikli Riskler

### Risk 3: Asenkron Gönderim (Threading) Sırasında Frame Senkronizasyon Kaybı
*   **Risk Tanımı:** `main.py`'deki `_submit_competition_step` ve `_fetch_competition_step` iki thread (`ThreadPoolExecutor(max_workers=2)`) vasıtasıyla yürütülüyor. Model GPU üzerinde çalışırken GIL (Global Interpreter Lock) thread kilitlenmesi yaparsa karelerin sunucuya gecikmeli gönderilmesi sorunu.
*   **Şartname İhlali / Puan Etkisi:** Kare atlama veya önceki kareden sonucu süresinde teslim edememe (Geçersiz tahmine sebep olur).
*   **Olasılık:** Düşük/Orta.
*   **Tetikleyici Senaryo:** Güçsüz CPU/Düşük donanımlı sistemlerde IO beklemeleri.
*   **Tespit Yöntemi:** Mantıksal çıkarım (`main.py` içindeki asenkron yapı kurgusu).
*   **Mimari Düzeyde İyileştirme Yönü:** Multiprocessing (Process bazında) veya non-blocking async REST istemcisi (örn. `aiohttp`) daha güvenli bir loop sağlayacaktır.

### Risk 4: Görev 3 Referans Objelerinin Dinamik Ölçeklere Adaptasyonu
*   **Risk Tanımı:** Referans eşleştirme (`image_matcher.py`) genel SIFT/ORB KNN-match mantığında. Ancak referans olarak "uydu görüntüsü" verilip "hava aracı kamerasından (düz, dar açılı)" eşleşmesi istendiği senaryolarda `findHomography` dejenere olabilir (özellikle ORB kullanımında).
*   **Şartname İhlali / Puan Etkisi:** Görev 3'teki Recall / Bulunma puanında düşüş.
*   **Olasılık:** Orta.
*   **Tetikleyici Senaryo:** Oturum başında verilen referans ile anlık görüntüde radikal açı ve ışık değişiklikleri (Termal - RGB kurgusu vs.).
*   **Tespit Yöntemi:** `image_matcher.py: _match_reference`.
*   **Mimari Düzeyde İyileştirme Yönü:** Gelişmiş SuperGlue veya LoFTR tarzı daha toleranslı feature eşleme deep-learning modellerine geçilmesi puan optimizasyonunu sağlar. (Veya en kötü senaryoda ORB yerine daima SIFT'in zorunlu koşulması).

---

## 5. Performans ve Kaynak Değerlendirmesi

*   **Hız / Donanım İhtiyacı:** CUDA aktif ise (FP16/Half precision destegi eklendiği görülmektedir), YOLOv8 inference süresi son derece düşük olacak, minimum gereksinim olan "1 FPS" sınırı rahatça geçilecektir. TTA (Augmented inference) veya SAHI ise hızı radikal düşürür, düşük güçteki bilgisayarlar için darboğaz oluşturabilir.
*   **Bellek Yönetimi:** OutOfMemoryError durumlarına karşı `gc.collect()` ve `torch.cuda.empty_cache()` çağrılarının try-except bloklarına sarılması (`detection.py: detect` metodu) oturumun çökmesini engelleyen çok kıymetli ve başarılı bir hamledir.
*   **Disk Okuma / Cache:** Simülasyon modunda dummy blank_image ve LRU yapılarının belleği patlatma olasılığı sınırlanmıştır (`maxlen=100`).

---

## 6. Belirsizlikler ve Koşullu Riskler

*   **Varsayım Sınıf Etiketleri:** Sistem (`detection.py`) COCO'ya benzer model id'lerini TEKNOFEST id'lerine map ederken normalize edilmiş harf/string tabanlı bir arama kullanmaktadır ("insan", "human", vs.). Eklenecek custom bir YOLO modelinde bu labelların birebir eşleşmeme riski, manuel dictionary check ile korunsa dahi, eğitilmiş güncel model labelları ile mutlak surette align edilmelidir.
*   **Payload `_confidence` Değeri:** Sınır alanlarda obje sayısının filtrelendiği `_apply_object_caps` içinde confidence değerleri hesaba katılmaktadır. Maksimum kutu (`RESULT_MAX_OBJECTS`) aşılırsa güveni düşük nesneler silinmektedir. Ancak yarışma sonucu JSON standartlarında "_confidence" adında gizli alanın sunucu post içeriğinde bulunmaması gerekir (Yarışma formatı dışı). Payload oluşturmada `clean_capped` ile sunucuya taşınmadığı doğrulanmıştır. Ciddi bir güvence senaryosudur.

---

## 7. Genel Sağlık Skoru (0–10)

**Puan: 8.5 / 10**

**Sonuç Özeti:**
Proje, Teknofest standart bir projenin çok ötesinde defansif programlama yaklaşımlarına, contract validator mekanizmalarına (`payload_schema.py`) ve HTTP Session Break toleransına sahiptir. Puan kaybı yaratacak risklerin %90'ı algoritmik zaafiyetlerden ziyade, "Tek Kameradan Derinlik/İrtifa tahmini" olan odometri probleminin saf mekanik handikaplarından (Drift, Scale-Ratio problemleri, Roll-Pitch dekuplajı) kaynaklanacaktır. Yazılım dizaynı, hata yönetimi ve rekabet döngüsü (Competition Loop) mükemmele yakındır.
