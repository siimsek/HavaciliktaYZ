# 🛩️ TEKNOFEST 2026 — Havacılıkta Yapay Zeka

<div align="center">

**Otonom hava araçları için gerçek zamanlı nesne tespiti ve görsel odometri sistemi**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg)](https://docs.ultralytics.com/)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 📋 İçindekiler

- [Proje Hakkında](#-proje-hakkında)
- [Mimari](#-mimari)
- [Özellikler](#-özellikler)
- [Kurulum](#-kurulum)
- [Kullanım](#-kullanım)
- [Yapılandırma](#-yapılandırma)
- [Görev 3 Parametre Dosyası](#-görev-3-parametre-dosyası)
- [Deterministiklik Sözleşmesi](#-deterministiklik-sözleşmesi)
- [Dosya Yapısı](#-dosya-yapısı)
- [Yarışma Kuralları](#-yarışma-kuralları)
- [Görev 1 Temporal Karar Mantığı](#-görev-1-temporal-karar-mantığı)
- [Eğitim ve Test Veri Setleri](#-eğitim-ve-test-veri-setleri)

---

## 🎯 Proje Hakkında

Bu proje, **TEKNOFEST 2026 Havacılıkta Yapay Zeka Yarışması** kapsamında geliştirilmiştir. Sistem iki ana görevi yerine getirir:

1. **Nesne Tespiti (Görev 1):** Drone kamera görüntülerinden taşıt, insan, UAP (Uçan Araba Park) ve UAİ (Uçan Ambulans İniş) alanlarını gerçek zamanlı tespit eder. İniş alanlarının uygunluk durumunu belirler.

2. **Pozisyon Kestirimi (Görev 2):** GPS sinyali kesildiğinde görsel odometri (optik akış) ile hava aracının konumunu kestirir.

---

## 🏗️ Mimari

```
┌─────────────────────────────────────────────────────┐
│                    main.py                          │
│              (Ana Orkestrasyon)                      │
│   FPS sayacı • Graceful shutdown • Hata yönetimi    │
└────────────┬────────────┬────────────┬──────────────┘
             │            │            │
     ┌───────▼──────┐  ┌──▼──────┐  ┌─▼─────────────┐
     │  network.py  │  │detection│  │ localization   │
     │              │  │  .py    │  │    .py         │
     │ HTTP istek   │  │ YOLOv8  │  │ GPS + Optik   │
     │ Retry logic  │  │ FP16    │  │ Akış hibrit   │
     │ Simülasyon   │  │ İniş    │  │ Lucas-Kanade  │
     │ JSON log     │  │ durumu  │  │ Odometri      │
     └──────┬───────┘  └────┬────┘  └───────┬───────┘
            │               │               │
     ┌──────▼───────────────▼───────────────▼───────┐
     │              config/settings.py               │
     │   Merkezi yapılandırma • Sınıf eşleştirme    │
     │   Kamera parametreleri • Ağ ayarları          │
     └──────────────────┬───────────────────────────┘
                        │
     ┌──────────────────▼───────────────────────────┐
     │              src/utils.py                     │
     │   Renkli Logger • Visualizer • JSON log      │
     └──────────────────────────────────────────────┘
```

---

## ✨ Özellikler

| Özellik | Detay |
|---------|-------|
| **Model** | YOLOv8n (Ultralytics) — COCO → TEKNOFEST sınıf eşleştirmesi |
| **Hız** | FP16 half-precision + model warmup → **~33 FPS** (RTX 3060) |
| **İniş Tespiti** | Intersection-over-area + kenar temas kontrolü |
| **Lokalizasyon** | Hibrit GPS + Lucas-Kanade optik akış |
| **Ağ** | Otomatik retry, timeout yönetimi, JSON traffic logging |
| **Debug** | Renkli konsol çıktısı, tespit görselleştirme, periyodik kayıt |
| **Güvenilirlik** | Global hata yakalama, SIGINT/SIGTERM handler, asla çökmez |
| **Offline** | İnternet bağlantısı gerektirmez — yarışma kurallarına uygun |

---

## 🚀 Kurulum

### Gereksinimler

- **Python** 3.10+
- **NVIDIA GPU** (önerilen) + CUDA 12.x
- **İşletim Sistemi:** Linux (Ubuntu 22.04 test edildi)

### Adımlar

```bash
# 1. Repoyu klonla
git clone https://github.com/siimsek/HavaciliktaYZ.git
cd HavaciliktaYZ

# 2. Sanal ortam oluştur
python3 -m venv venv
source venv/bin/activate

# 3. PyTorch'u CUDA ile kur (önce)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Diğer bağımlılıkları kur
pip install -r requirements.txt

# 5. Model dosyasını indir (eğer yoksa)
# YOLOv8n modeli models/ dizinine yerleştirilmeli
mkdir -p models
# https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt
```

---

## 💻 Kullanım

### Simülasyon Modu (Test)

Sunucu bağlantısı olmadan, yerel bir görüntü ile test:

```bash
# 1. Simülasyon görselini hazırla
cp test_image.jpg sim_data/test_frame.jpg

# 2. Simülasyon modunda çalıştır
python main.py
```

`config/settings.py` dosyasında `SIMULATION_MODE = True` olduğundan emin olun.

### 🧪 Otonom Test Modu (VisDrone ile)

VisDrone veri setini kullanarak tam sistem testi yapabilirsiniz (Sunucu gerektirmez):

```bash
# Görev 2: Sıralı kareler (Odometri testi)
# datasets/VisDrone2019-VID-val/ içinden rastgele bir sekans seçer
python main.py --simulate

# Görev 1: Tekil fotoğraflar (Nesne tespiti testi)
# datasets/VisDrone2019-DET-train/ içinden rastgele 100 fotoğraf seçer
python main.py --simulate det
```
*Not: Bu modda sonuçlar renkli olarak terminale basılır.*

### Yarışma Modu

```bash
# 1. settings.py'yi güncelle
#    SIMULATION_MODE = False
#    SERVER_URL = "http://<yarışma-sunucu-ip>:5000"
#    TEAM_NAME = "<takım-adınız>"

# 2. Sistemi başlat
python main.py
```

### Çıktı Formatı (Sunucuya Gönderilen JSON)

```json
{
  "frame": "http://server/frame/123",
  "detected_objects": [
    {
      "cls": "0",
      "landing_status": "-1",
      "top_left_x": 150,
      "top_left_y": 200,
      "bottom_right_x": 400,
      "bottom_right_y": 350
    }
  ],
  "detected_translations": [
    {
      "translation_x": 1.25,
      "translation_y": -0.43,
      "translation_z": 0.0
    }
  ]
}
```

---

## ⚙️ Yapılandırma

Tüm ayarlar [`config/settings.py`](config/settings.py) içinde merkezi olarak yönetilir:

| Parametre | Varsayılan | Açıklama |
|-----------|-----------|----------|
| `SIMULATION_MODE` | `True` | Test modu (sunucu bağlantısız) |
| `DEBUG` | `True` | Detaylı log + görsel çıktı |
| `CONFIDENCE_THRESHOLD` | `0.25` | Minimum tespit güven eşiği |
| `HALF_PRECISION` | `True` | FP16 hızlandırma (CUDA) |
| `WARMUP_ITERATIONS` | `3` | Model ısınma tekrarı |
| `MAX_FRAMES` | `2250` | Yarışma karesi limiti |

---

## 🎛️ Görev 3 Parametre Dosyası

Görev 3 (dinamik referans obje tespiti) için tüm kritik eşikler tek bir dosyada tanımlanır:

- Dosya: [`config/task3_params.yaml`](config/task3_params.yaml)
- Amaç: `T_confirm`, `T_fallback`, `N`, `grid stride` değerlerini merkezi ve denetlenebilir tutmak

### Parametreler

| Parametre | Dosya Anahtarı | Açıklama |
|-----------|----------------|----------|
| `T_confirm` | `t_confirm` | Stage-2 aday doğrulama minimum benzerlik eşiği |
| `T_fallback` | `t_fallback` | Stage-3 fallback sweep kabul eşiği |
| `N` | `n_fallback_interval` | Stage-3 fallback'in her kaç frame'de bir tetikleneceği |
| `grid stride` | `grid_stride` | Stage-3 grid/sliding-window tarama adımı (piksel) |

### Örnek İçerik

```yaml
t_confirm: 0.72
t_fallback: 0.66
n_fallback_interval: 5
grid_stride: 32
```

Not: Bu değerler çalışma sırasında dinamik değiştirilmemelidir; deterministik ve tekrarlanabilir karar için oturum başında sabitlenmelidir.

---

## 🔒 Deterministiklik Sözleşmesi

Sistem çıktılarının tekrarlanabilir olması için aşağıdaki kurallar zorunludur:

1. **Seed Sabitleme (numpy/torch/random):**
   - Tüm çalıştırmalarda aynı seed kullanılmalıdır.
   - Öneri: `numpy`, `torch`, `random` için tek noktadan seed ataması yapılmalı.

2. **Model Eval Mode:**
   - İnference öncesi tüm modeller `eval` modunda çalıştırılmalıdır.
   - Dropout ve BatchNorm gibi katmanların eğitim davranışı kapatılmalıdır.

3. **Sabit Sürüm Pinleme:**
   - `torch`, `torchvision`, `ultralytics`, CUDA ve cuDNN sürümleri pinlenmelidir.
   - Üretim ortamında sürüm kayması engellenmeli, aynı bağımlılık seti korunmalıdır.

4. **JSON Sırası ve Kararlı Serileştirme:**
   - Çıktı JSON'ları kararlı anahtar sırası ile üretilmelidir (`sort_keys=True` veya sabit alan sırası).
   - Sayısal formatlama ve alan sırası sürümler arasında değiştirilmemelidir.

5. **Frame-Index Tabanlı Karar Kuralları:**
   - Adaptasyonlar wall-clock süreye göre değil, frame index/pencere kuralına göre yapılmalıdır.
   - Bu yaklaşım farklı donanımlarda aynı karar davranışını korur.

---

## 📂 Dosya Yapısı

```
HavaciliktaYZ/
├── main.py                  # Ana giriş noktası
├── requirements.txt         # Python bağımlılıkları
├── README.md               # Bu dosya
├── .gitignore              # Git hariç tutma kuralları
│
├── config/
│   ├── __init__.py
│   ├── settings.py         # Merkezi yapılandırma
│   └── task3_params.yaml   # Görev 3 eşik ve tarama parametreleri
│
├── src/
│   ├── __init__.py
│   ├── detection.py        # YOLOv8 nesne tespiti + iniş durumu
│   ├── network.py          # Sunucu iletişimi + retry + simülasyon
│   ├── localization.py     # GPS + optik akış pozisyon kestirimi
│   └── utils.py            # Logger, Visualizer, yardımcı araçlar
│
├── models/
│   └── yolov8n.pt          # YOLOv8 nano modeli (Git'e dahil değil)
│
├── sim_data/
│   └── test_frame.jpg      # Simülasyon test görseli
│
├── sartname/
│   └── teknofest_context.md # Yarışma şartname özeti
│
├── logs/                   # Çalışma zamanı logları (otomatik)
└── debug_output/           # Debug görselleri (otomatik)
```

---

## 📏 Yarışma Kuralları (Özet)

### Tespit Edilecek Nesneler

| Sınıf | ID | İniş Durumu | Açıklama |
|-------|----|-------------|----------|
| **Taşıt** | 0 | -1 | Otomobil, motosiklet, otobüs, tren, deniz taşıtı |
| **İnsan** | 1 | -1 | Ayakta/oturur tüm insanlar |
| **UAP** | 2 | 0 veya 1 | Uçan Araba Park alanı |
| **UAİ** | 3 | 0 veya 1 | Uçan Ambulans İniş alanı |

### İniş Uygunluk Kuralları

- **Uygun (1):** Alan tamamen kadraj içinde VE üzerinde hiçbir nesne yok
- **Uygun Değil (0):** Alan kısmen kadraj dışı VEYA üzerinde nesne var
- Bisiklet/motosiklet sürücüleri "insan" değil, taşıtla birlikte "taşıt" olarak etiketlenir

## ⏱️ Görev 1 Temporal Karar Mantığı

Görev 1 kararları tek frame üzerinden verilmez. Tüm hareket ve iniş uygunluk çıktıları pencere (window) tabanlı temporal birikim ile üretilir.

### 1) Window (Pencere) Yapısı

- Her hedef nesne/alan için son `W` frame tutulur (örnek: `W=24`).
- `W` değeri sabit konfigürasyon parametresidir; çalışma sırasında dinamik değiştirilmez.
- Karar, tek bir frame yerine pencere içindeki kanıtların birleşimi ile verilir.

### 2) Decay (Ağırlıklandırma)

- Yakın frame'lere daha yüksek, eski frame'lere daha düşük ağırlık verilir.
- Örnek ağırlık şeması: üstel veya doğrusal decay (`w_t`) ve normalize toplam.
- Amaç kısa süreli gürültü/yanlış tespitten etkilenmeden stabil karar üretmektir.

### 3) Threshold (Karar Eşiği)

- Pencere boyunca biriken temporal skor `S` hesaplanır.
- `S >= T_move` ise taşıt için `movement_status=1`, aksi halde `movement_status=0`.
- UAP/UAİ için `S >= T_land` ise `landing_status=1`, aksi halde `landing_status=0`.
- `T_move` ve `T_land` kalibrasyon testleri ile sabitlenir.

### 4) Tek-Frame Karar Yasağı

- Tek frame ile doğrudan `movement_status` veya `landing_status` kararı verilmez.
- Anlık kararlar yalnızca geçici kanıt olarak temporal havuza yazılır; nihai karar pencere sonunda üretilir.

### Teknik Kısıtlamalar

- 📡 İnternet bağlantısı **yasak** (offline çalışma zorunlu)
- 🎬 Oturum başına **2250 kare** (5 dk, 7.5 FPS)
- 📐 Çözünürlük: 1920×1080 veya 3840×2160
- 📊 Değerlendirme: mAP (IoU ≥ 0.5)

---

## 📊 Eğitim ve Test Veri Setleri

Yarışma öncesi modeli eğitmek ve sistemi test etmek için kullanılabilecek veri setleri:

### Önerilen Veri Setleri

| Dataset | İçerik | Neden Uygun? | Link |
|---------|--------|-------------|------|
| **VisDrone** | 260K+ kare, insan + araç | Drone perspektifi, çeşitli ortamlar | [GitHub](https://github.com/VisDrone/VisDrone-Dataset) |
| **UAVDT** | 80K kare, araç tespiti | UAV yükseklik çeşitliliği | [Site](https://sites.google.com/view/grli-uavdt) |
| **TEKNOFEST Resmi** | Örnek video (Mart 2026) | Yarışma formatı ile birebir uyumlu | [GitHub](https://github.com/TEKNOFEST-YARISMALAR/havacilikta-yapay-zeka-yarismasi) |

### VisDrone ile Eğitim

VisDrone sınıfları TEKNOFEST'e doğrudan eşleştirilebilir:

```
VisDrone → TEKNOFEST
──────────────────────
pedestrian    → İnsan (1)
people        → İnsan (1)
car           → Taşıt (0)
van           → Taşıt (0)
truck         → Taşıt (0)
bus           → Taşıt (0)
motor         → Taşıt (0)
bicycle       → Taşıt (0)
tricycle      → Taşıt (0)
```

> ⚠️ **Not:** TEKNOFEST resmi örnek video dağıtım tarihi **10-28 Mart 2026**'tir. [Resmi repo](https://github.com/TEKNOFEST-YARISMALAR/havacilikta-yapay-zeka-yarismasi) takip edilmelidir.

---

## 📜 Lisans

MIT License — Detaylar için [LICENSE](LICENSE) dosyasına bakın.

---

<div align="center">

**TEKNOFEST 2026 Havacılıkta Yapay Zeka Yarışması** için geliştirilmiştir 🇹🇷

</div>
