# 🛩️ TEKNOFEST 2025 — Havacılıkta Yapay Zeka

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
- [Dosya Yapısı](#-dosya-yapısı)
- [Yarışma Kuralları](#-yarışma-kuralları)
- [Eğitim ve Test Veri Setleri](#-eğitim-ve-test-veri-setleri)

---

## 🎯 Proje Hakkında

Bu proje, **TEKNOFEST 2025 Havacılıkta Yapay Zeka Yarışması** kapsamında geliştirilmiştir. Sistem iki ana görevi yerine getirir:

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
│   └── settings.py         # Merkezi yapılandırma
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
| **TEKNOFEST Resmi** | Örnek video (Mart 2025) | Yarışma formatı ile birebir uyumlu | [GitHub](https://github.com/TEKNOFEST-YARISMALAR/havacilikta-yapay-zeka-yarismasi) |

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

> ⚠️ **Not:** TEKNOFEST resmi örnek video dağıtım tarihi **10-28 Mart 2025**'tir. [Resmi repo](https://github.com/TEKNOFEST-YARISMALAR/havacilikta-yapay-zeka-yarismasi) takip edilmelidir.

---

## 📜 Lisans

MIT License — Detaylar için [LICENSE](LICENSE) dosyasına bakın.

---

<div align="center">

**TEKNOFEST 2025 Havacılıkta Yapay Zeka Yarışması** için geliştirilmiştir 🇹🇷

</div>
