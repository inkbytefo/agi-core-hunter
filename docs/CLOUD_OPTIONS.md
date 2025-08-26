# Bulut Bilişim Seçenekleri - AGI Core Hunter

**Güncelleme:** 26 Ağustos 2025

---

## 🚀 Hızlı Karşılaştırma

| Platform | Ücretsiz GPU | Süre Limiti | Kurulum | Önerilen |
|----------|--------------|-------------|---------|----------|
| **Google Colab** | ✅ T4 | 12 saat | Kolay | ⭐⭐⭐⭐⭐ |
| **Kaggle** | ✅ P100/T4 | 30 saat/hafta | Orta | ⭐⭐⭐⭐ |
| **Paperspace** | ❌ (Ücretli) | Sınırsız | Kolay | ⭐⭐⭐ |
| **AWS SageMaker** | ❌ (Ücretli) | Sınırsız | Zor | ⭐⭐ |
| **Vast.ai** | ❌ (Ucuz) | Sınırsız | Orta | ⭐⭐⭐ |

---

## 🥇 1. Google Colab (ÖNERİLEN)

### Avantajlar
- ✅ **Ücretsiz GPU:** Tesla T4 (16GB VRAM)
- ✅ **Kolay kurulum:** Tek tıkla başlat
- ✅ **Jupyter entegrasyonu:** Notebook formatı
- ✅ **Google Drive:** Otomatik kaydetme
- ✅ **Wandb desteği:** Direkt entegrasyon

### Dezavantajlar
- ❌ **12 saat limiti:** Uzun deneyler için yetersiz
- ❌ **Veri kaybı riski:** Session sonunda silinir
- ❌ **Sıra bekleme:** Yoğun saatlerde GPU bulunamayabilir

### Kullanım
```python
# 1. notebooks/AGI_Core_Hunter_Colab.ipynb'i aç
# 2. Runtime → Change runtime type → GPU
# 3. Hücreleri sırayla çalıştır
```

### Pro Sürüm ($10/ay)
- ⚡ **Daha hızlı GPU:** V100, A100
- ⏰ **24 saat limit:** 2x daha uzun
- 💾 **Daha fazla RAM:** 32GB'a kadar

---

## 🥈 2. Kaggle Notebooks

### Avantajlar
- ✅ **Güçlü GPU:** P100 (16GB) veya T4
- ✅ **30 saat/hafta:** Colab'dan daha uzun
- ✅ **Dataset entegrasyonu:** Kolay veri yönetimi
- ✅ **Topluluk:** Paylaşım ve feedback

### Dezavantajlar
- ❌ **Haftalık limit:** 30 saat sonra bekleme
- ❌ **Dataset yükleme:** Proje dosyalarını dataset olarak yüklemek gerekli
- ❌ **İnternet kısıtı:** Bazı sitelere erişim yok

### Kullanım
```bash
# 1. Projeyi zip'le ve Kaggle'a dataset olarak yükle
# 2. Yeni notebook oluştur
# 3. notebooks/kaggle_setup.py'yi çalıştır
```

---

## 🥉 3. Paperspace Gradient

### Avantajlar
- ✅ **Güçlü GPU'lar:** RTX 4000, V100, A100
- ✅ **Sınırsız süre:** Ücretli planlar
- ✅ **Persistent storage:** Veriler kaybolmaz
- ✅ **Terminal erişimi:** Tam kontrol

### Dezavantajlar
- ❌ **Ücretli:** $8/saat'ten başlıyor
- ❌ **Kredi sistemi:** Kullanım başına ödeme

### Kullanım
```bash
# 1. Paperspace hesabı oluştur
# 2. Gradient notebook başlat
# 3. Git clone ile projeyi çek
pip install -r requirements.txt
python experiments/01_mdl_vs_ood/train.py
```

---

## 💰 4. Vast.ai (Ucuz Alternatif)

### Avantajlar
- ✅ **Çok ucuz:** $0.2-0.5/saat
- ✅ **Güçlü GPU'lar:** RTX 3090, 4090, A100
- ✅ **Esnek:** İstediğin kadar kullan
- ✅ **SSH erişimi:** Tam kontrol

### Dezavantajlar
- ❌ **Teknik bilgi gerekli:** Linux/SSH bilgisi
- ❌ **Güvenilirlik:** Makineler bazen kapanabilir
- ❌ **Kurulum:** Manuel setup gerekli

### Kullanım
```bash
# 1. Vast.ai hesabı oluştur
# 2. Uygun makine kirala
# 3. SSH ile bağlan
ssh root@[ip_address]
git clone https://github.com/[username]/agi_core_hunter.git
cd agi_core_hunter
pip install -r requirements.txt
python experiments/01_mdl_vs_ood/train.py
```

---

## ⚡ Hızlı Başlangıç Stratejisi

### 1. İlk Test (5 dakika)
```python
# Google Colab'da hızlı demo
!python experiments/01_mdl_vs_ood/demo.py
```

### 2. Pilot Deney (15 dakika)
```python
# 500 episode test
config['training']['total_episodes'] = 500
```

### 3. Tam Deney (30-60 dakika)
```python
# 5000 episode full run
# Colab Pro veya Kaggle'da çalıştır
```

---

## 🛠️ Platform-Specific Optimizasyonlar

### Google Colab
```python
# JAX GPU optimizasyonu
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

# Hızlı compilation
import jax
jax.config.update('jax_compilation_cache_dir', '/tmp/jax_cache')
```

### Kaggle
```python
# Kaggle GPU'yu zorla
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Kaggle dataset'ten dosya kopyala
!cp -r /kaggle/input/agi-core-hunter/* /kaggle/working/
```

### Vast.ai
```bash
# CUDA kurulumu
apt update && apt install -y nvidia-cuda-toolkit

# Python environment
conda create -n agi python=3.9
conda activate agi
pip install -r requirements.txt
```

---

## 📊 Maliyet Analizi (Aylık)

### Ücretsiz Seçenekler
- **Google Colab Free:** $0 (12 saat/gün limit)
- **Kaggle:** $0 (30 saat/hafta limit)

### Ücretli Seçenekler
- **Colab Pro:** $10/ay (24 saat/gün)
- **Paperspace:** ~$50/ay (orta kullanım)
- **Vast.ai:** ~$20/ay (düşük kullanım)
- **AWS SageMaker:** ~$100/ay (yoğun kullanım)

---

## 🎯 Önerilen Strateji

### Başlangıç (İlk hafta)
1. **Google Colab Free** ile pilot testler
2. Sonuçlar umut verici ise **Colab Pro** upgrade
3. Uzun deneyler için **Kaggle** kullan

### Gelişim (İkinci hafta)
1. **Vast.ai** ile maliyet-etkin uzun deneyler
2. **Paperspace** ile production-ready setup
3. **AWS** ile ölçeklenebilir pipeline

### Üretim (Uzun vadeli)
1. Kendi GPU sunucusu (ROI hesabı yap)
2. Hibrit yaklaşım (lokal + bulut)
3. Akademik hibeler ile kaynak sağla

---

## 🚨 Önemli İpuçları

### Veri Kaybını Önleme
```python
# Sonuçları düzenli kaydet
import pickle
with open('checkpoint.pkl', 'wb') as f:
    pickle.dump(results, f)

# Google Drive'a otomatik yedek
from google.colab import drive
drive.mount('/content/drive')
!cp results.json /content/drive/MyDrive/
```

### GPU Kullanımını Optimize Et
```python
# Batch size'ı artır
config['training']['batch_size'] = 64

# Paralel environment
from jax import vmap
batch_env_step = vmap(env.step)
```

### Monitoring
```python
# Wandb ile canlı takip
import wandb
wandb.watch(model, log='all')

# Email bildirimi
!pip install yagmail
# Deney bitince email gönder
```

---

**Sonuç:** Google Colab ile başla, ihtiyaca göre diğer platformları dene! 🚀