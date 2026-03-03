# ADR-0001: Görev 3 için deep matcher entegrasyonunun ertelenmesi

- **Tarih:** 2026-03-03
- **Durum:** Kabul edildi
- **Karar Sahibi:** Görüntü Eşleştirme/Misyon Yazılım Ekibi
- **Kapsam:** Görev 3 referans obje eşleştirme hattı

## Bağlam

Görev 3 kapsamında mevcut üretim hattı `ORB/SIFT + KNN + homography` yaklaşımıyla çalışmaktadır.
Denetim notlarında, cross-modal sahnelerde daha güçlü sonuçlar için SuperGlue/LoFTR gibi deep matcher alternatifleri önerilmiştir.

Bununla birlikte mevcut yarışma/çalışma koşullarında aşağıdaki kısıtlar öne çıkmıştır:

1. **Operasyonel risk:** Yeni deep matcher entegrasyonu; model ağırlıkları, ek bağımlılıklar, cihazlar arası tutarlılık ve hata ayıklama maliyetini artırır.
2. **Performans belirsizliği:** Mevcut donanım/FPS hedefleri altında gecikme ve kaynak tüketimi etkisi güvenli biçimde doğrulanmamıştır.
3. **Entegrasyon maliyeti:** Mevcut boru hattında eşleştirme sonrası geometri/doğrulama katmanları yeniden ele alınmadan hızlı geçiş, yarışma kararlılığını düşürebilir.
4. **Zamanlama önceliği:** Kısa vadede en yüksek güvenilirlik kazanımı, mevcut ORB/SIFT hattında eşik ve kalibrasyon iyileştirmeleriyle elde edilebilmektedir.

## Karar

**Görev 3 için deep matcher (SuperGlue/LoFTR vb.) entegrasyonu bu aşamada yapılmayacaktır.**

Bu iterasyonda yalnızca mevcut ORB/SIFT hattı üzerinde aşağıdaki iyileştirme sınıfına izin verilir:

- Eşik (threshold) optimizasyonu,
- Kalibrasyon/tuning güncellemeleri,
- Parametre setlerinin sahaya/donanıma göre stabilize edilmesi.

Mimari temel (feature çıkarımı + eşleştirme + homography doğrulama) korunacaktır.

## Sonuçlar

### Pozitif

- Daha düşük entegrasyon riski ve daha öngörülebilir teslimat.
- Mevcut test/operasyon bilgisinden yararlanarak kısa vadede stabilite artışı.
- Yarışma döngüsünde bakım ve hata ayıklama karmaşıklığının sınırlanması.

### Negatif

- Zorlu cross-modal senaryolarda deep matcher tabanlı potansiyel doğruluk artışı kısa vadede kullanılamayacaktır.
- Uzun vadede daha güçlü matcher mimarisine geçiş için ayrı bir keşif/PoC fazı gerekecektir.

## Uygulama Notu

- `config/task3_params.yaml` ve ilgili eşik parametreleri üzerinden kalibrasyon odaklı ilerlenir.
- Deep matcher araştırması ayrı bir teknik keşif maddesi olarak backlog'da tutulur; üretim hattına doğrudan alınmaz.
