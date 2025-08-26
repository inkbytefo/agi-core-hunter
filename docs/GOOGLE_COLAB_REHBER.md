# Google Colab ile AGI Core Hunter - Detaylı Rehber

**Tarih:** 26 Ağustos 2025  
**Süre:** 5-30 dakika (deney türüne göre)  
**Maliyet:** Ücretsiz (GPU dahil)

---

## 🎯 Bu Rehberde Neler Var?

1. **Adım adım kurulum** (5 dakika)
2. **Hızlı demo** (2 dakika)
3. **Pilot deney** (15 dakika)
4. **Tam deney** (30 dakika)
5. **Sonuç analizi** (5 dakika)
6. **Sorun giderme** (ihtiyaç halinde)

---

## 🚀 Adım 1: Google Colab'a Giriş

### 1.1 Google Hesabı ile Giriş
1. **Tarayıcıda aç:** https://colab.research.google.com
2. **Google hesabınla giriş yap** (Gmail hesabı yeterli)
3. **"New notebook" tıkla** veya mevcut notebook aç

### 1.2 GPU'yu Aktifleştir (ÖNEMLİ!)
1. **Menüden:** Runtime → Change runtime type
2. **Hardware accelerator:** GPU seç
3. **GPU type:** T4 (ücretsiz) seç
4. **Save** tıkla

![GPU Ayarı](https://i.imgur.com/gpu-setup.png)

### 1.3 GPU Kontrolü
```python
# İlk hücreye yaz ve çalıştır (Shift+Enter)
!nvidia-smi
```

**Beklenen çıktı:**
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.xx.xx    Driver Version: 525.xx.xx    CUDA Version: 12.0  |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla T4            Off  | 00000000:00:04.0 Off |                    0 |
| N/A   xx°C    P8    xx W /  70W|      0MiB / 15360MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
```

✅ **Tesla T4 görüyorsan hazırsın!**  
❌ **GPU görünmüyorsa:** Runtime → Restart runtime, tekrar dene

---

## 🛠️ Adım 2: Proje Kurulumu

### 2.1 Projeyi İndir
```python
# Yeni hücre oluştur ve çalıştır
import os

# Eğer GitHub'da yayınladıysan
!git clone https://github.com/[username]/agi_core_hunter.git
%cd agi_core_hunter

# Eğer henüz GitHub'da değilse, dosyaları manuel yükle
# (Aşağıdaki "Manuel Yükleme" bölümüne bak)
```

### 2.2 Manuel Dosya Yükleme (GitHub yoksa)
```python
# Sol panelden "Files" ikonuna tıkla
# "Upload" butonuna tıkla
# Tüm proje dosyalarını sürükle-bırak

# Veya zip dosyası yükle
from google.colab import files
uploaded = files.upload()

# Zip'i aç
!unzip agi_core_hunter.zip
%cd agi_core_hunter
```

### 2.3 Bağımlılıkları Yükle
```python
# Bu biraz zaman alabilir (2-3 dakika)
!pip install jax[cuda] flax optax chex wandb tqdm matplotlib seaborn pandas -q

print("✅ Kurulum tamamlandı!")
```

### 2.4 Kurulum Testi
```python
# Test scripti çalıştır
!python test_setup.py
```

**Beklenen çıktı:**
```
🚀 AGI Core Hunter - Setup Test
========================================
🧪 Testing imports...
✅ MDLAgent import successful
✅ GridWorld import successful
✅ MetricsTracker import successful

🌍 Testing environment...
✅ Environment reset successful, obs shape: (40,)
✅ Environment step successful, reward: -0.10
✅ ASCII rendering successful

🤖 Testing agent...
✅ Agent creation successful
✅ Agent setup successful
✅ Action selection successful, action: 2

📊 Testing metrics...
✅ MetricsTracker successful, mean reward: 9.0
✅ OOD ratio calculation successful: 0.75

========================================
📈 Test Results: 4/4 passed
🎉 All tests passed! Project setup is working correctly.
```

---

## 🎮 Adım 3: Hızlı Demo (2 dakika)

### 3.1 Demo Çalıştır
```python
# Basit demo - agent'ın nasıl çalıştığını gösterir
!python experiments/01_mdl_vs_ood/demo.py
```

**Beklenen çıktı:**
```
🚀 AGI Core Hunter - MDL Agent Demo
========================================
✅ Environment created: 6x6 grid
✅ Agent created with latent_dim=4, β=1.0
✅ Agent setup complete

🎮 Running demo episodes...

--- Episode 1 ---
Initial state:
. . . . . . 
. X . . . .
. X . . . .
. . . . . X
A . . . . .
. X G . . .

Step 1: Down -> Reward: -0.10
Step 2: Right -> Reward: -0.10
...
🎯 Goal reached!

🗜️ Testing compression...
Original observation dim: 40
Compressed latent dim: 4
Compression ratio: 10.0x
```

✅ **Demo başarılıysa devam et!**

---

## ⚡ Adım 4: Pilot Deney (15 dakika)

### 4.1 Hızlı Test Konfigürasyonu
```python
# Hızlı test için parametreleri azalt
import json

# Orijinal manifest'i oku
with open('experiments/01_mdl_vs_ood/manifest.json', 'r') as f:
    config = json.load(f)

# Hızlı test parametreleri
config['training']['total_episodes'] = 500  # 5000 yerine 500
config['training']['eval_frequency'] = 100  # Daha sık değerlendirme
config['training']['batch_size'] = 32       # Daha büyük batch

# OOD test episode'larını azalt
for ood_test in config['evaluation']['ood_tests']:
    ood_test['episodes'] = 25  # 100 yerine 25

# Hızlı manifest kaydet
with open('experiments/01_mdl_vs_ood/manifest_fast.json', 'w') as f:
    json.dump(config, f, indent=2)

print("✅ Hızlı test konfigürasyonu hazır!")
print(f"📊 Toplam episode: {config['training']['total_episodes'] * len(config['agents'])}")
print("⏱️ Tahmini süre: ~15 dakika")
```

### 4.2 Wandb Kurulumu (Opsiyonel)
```python
# Wandb ile sonuçları takip et
import wandb

# Seçenek 1: Wandb hesabı varsa
wandb.login()  # API key iste

# Seçenek 2: Wandb hesabı yoksa (offline mode)
import os
os.environ['WANDB_MODE'] = 'offline'
print("📊 Wandb offline mode aktif")
```

### 4.3 Pilot Deneyi Çalıştır
```python
# Deney dizinine geç
%cd experiments/01_mdl_vs_ood

# Hızlı deneyi başlat
!python train.py --manifest manifest_fast.json
```

**Çalışırken göreceğin çıktı:**
```
Starting experiment: MDL vs OOD Generalization
Training 3 agents for 500 episodes each

--- Training MDL_Low ---
Training MDLAgent: 100%|██████████| 500/500 [05:23<00:00,  1.55it/s]

--- Training MDL_Medium ---
Training MDLAgent: 100%|██████████| 500/500 [05:18<00:00,  1.57it/s]

--- Training MDL_High ---
Training MDLAgent: 100%|██████████| 500/500 [05:25<00:00,  1.54it/s]

Evaluating MDL_Low on standard environment...
Evaluating MDL_Low on OOD scenarios...
...
```

### 4.4 Canlı Takip (Wandb varsa)
```python
# Wandb dashboard linkini kopyala ve yeni sekmede aç
# Örnek: https://wandb.ai/[username]/agi_core_hunter/runs/[run_id]

# Göreceğin metrikler:
# - total_reward: Episode başına toplam ödül
# - success: Hedefe ulaşma oranı  
# - reconstruction_loss: VAE yeniden yapılandırma kaybı
# - kl_loss: KL divergence (sıkıştırma baskısı)
```

---

## 🔥 Adım 5: Tam Deney (30 dakika)

### 5.1 Tam Deney Kararı
```python
# Pilot sonuçları kontrol et
import json
with open('results.json', 'r') as f:
    results = json.load(f)

# Hızlı özet
for agent_name, data in results.items():
    beta = data['agent_config']['beta']
    success = data['standard_success_rate']
    print(f"{agent_name}: β={beta}, Success={success:.3f}")

# Eğer sonuçlar umut verici ise tam deneyi çalıştır
```

### 5.2 Tam Deney Çalıştır
```python
# Orijinal manifest ile tam deney (5000 episode)
!python train.py

# Bu ~30 dakika sürecek, sabırlı ol!
# Colab'ı kapatma, bağlantı kopabilir
```

### 5.3 İlerleme Takibi
```python
# Başka bir hücrede ilerlemeyi kontrol et
import time
import json
import os

def check_progress():
    if os.path.exists('results.json'):
        with open('results.json', 'r') as f:
            results = json.load(f)
        print("✅ Deney tamamlandı!")
        return True
    else:
        print("🔄 Deney devam ediyor...")
        return False

# Her 5 dakikada kontrol et
for i in range(12):  # 60 dakika max
    if check_progress():
        break
    time.sleep(300)  # 5 dakika bekle
```

---

## 📊 Adım 6: Sonuç Analizi

### 6.1 Temel Sonuçlar
```python
# Sonuçları yükle ve analiz et
import json
import numpy as np
import matplotlib.pyplot as plt

with open('results.json', 'r') as f:
    results = json.load(f)

# Agent performanslarını karşılaştır
agents = list(results.keys())
betas = []
ood_ratios = []
standard_success = []

for agent in agents:
    beta = results[agent]['agent_config']['beta']
    std_success = results[agent]['standard_success_rate']
    
    # OOD performance ratio hesapla
    ood_perf = results[agent]['ood_performance_ratios']
    avg_ood = np.mean([v for v in ood_perf.values()])
    
    betas.append(beta)
    ood_ratios.append(avg_ood)
    standard_success.append(std_success)

# Sonuçları yazdır
print("🎯 DENEY SONUÇLARI:")
print("=" * 50)
for i, agent in enumerate(agents):
    print(f"{agent}:")
    print(f"  β (sıkıştırma): {betas[i]}")
    print(f"  Standard başarı: {standard_success[i]:.3f}")
    print(f"  OOD oranı: {ood_ratios[i]:.3f}")
    print()
```

### 6.2 Görselleştirme
```python
# Beta vs OOD performansı grafiği
plt.figure(figsize=(10, 6))
plt.scatter(betas, ood_ratios, s=100, alpha=0.7)

for i, agent in enumerate(agents):
    plt.annotate(agent, (betas[i], ood_ratios[i]), 
                xytext=(5, 5), textcoords='offset points')

# Trend çizgisi
z = np.polyfit(betas, ood_ratios, 1)
p = np.poly1d(z)
plt.plot(betas, p(betas), "r--", alpha=0.8)

plt.xlabel('Beta (Sıkıştırma Baskısı)')
plt.ylabel('Ortalama OOD Performans Oranı')
plt.title('MDL İlkesi: Beta vs OOD Genelleme')
plt.grid(True, alpha=0.3)
plt.show()

# Korelasyon hesapla
correlation = np.corrcoef(betas, ood_ratios)[0, 1]
print(f"📈 Beta-OOD Korelasyonu: {correlation:.3f}")
```

### 6.3 Hipotez Testi
```python
# Hipotezi test et
print("\n🧪 HİPOTEZ TESTİ:")
print("Hipotez: 'Yüksek β (sıkıştırma) daha iyi OOD genelleme sağlar'")
print("-" * 60)

if correlation > 0.5:
    print("✅ GÜÇLÜ DESTEK: Hipotez güçlü şekilde desteklendi!")
    print(f"   Yüksek β değeri OOD performansını {correlation:.1%} oranında iyileştiriyor.")
    
elif correlation > 0.3:
    print("✅ ORTA DESTEK: Hipotez desteklendi.")
    print(f"   Pozitif korelasyon var ama güçlü değil ({correlation:.3f}).")
    
elif correlation < -0.3:
    print("❌ HİPOTEZ REDDEDİLDİ: Yüksek β OOD performansını kötüleştiriyor!")
    print(f"   Negatif korelasyon: {correlation:.3f}")
    
else:
    print("❓ BELİRSİZ SONUÇ: Net bir ilişki bulunamadı.")
    print(f"   Korelasyon çok zayıf: {correlation:.3f}")
    print("   Daha fazla veri veya farklı metrik gerekebilir.")

print(f"\n📊 Detaylı Sonuçlar:")
print(f"   En iyi OOD performansı: {max(ood_ratios):.3f}")
print(f"   En kötü OOD performansı: {min(ood_ratios):.3f}")
print(f"   Performans farkı: {max(ood_ratios) - min(ood_ratios):.3f}")
```

### 6.4 Detaylı Analiz Notebook'u
```python
# Daha detaylı analiz için notebook çalıştır
# (Bu biraz zaman alabilir)

# Jupyter notebook widget'larını yükle
!pip install ipywidgets -q

# Analiz notebook'unu çalıştır
%run eval.ipynb
```

---

## 💾 Adım 7: Sonuçları Kaydet

### 7.1 Dosyaları İndir
```python
# Sonuçları zip'le
!zip -r experiment_results.zip results.json *.png wandb/ -q

# Google Colab'dan indir
from google.colab import files
files.download('experiment_results.zip')

print("✅ Sonuçlar bilgisayarına indirildi!")
```

### 7.2 Google Drive'a Kaydet
```python
# Google Drive'ı bağla
from google.colab import drive
drive.mount('/content/drive')

# Sonuçları Drive'a kopyala
!cp -r results.json *.png /content/drive/MyDrive/AGI_Core_Hunter_Results/
!cp experiment_results.zip /content/drive/MyDrive/

print("✅ Sonuçlar Google Drive'a kaydedildi!")
```

### 7.3 GitHub'a Yükle (Opsiyonel)
```python
# Eğer GitHub repo'n varsa sonuçları commit et
!git add results.json
!git commit -m "Add experiment results from Colab"
!git push origin main

# GitHub token gerekebilir
```

---

## 🚨 Sorun Giderme

### Problem 1: GPU Bulunamıyor
```python
# Çözüm 1: Runtime'ı yeniden başlat
# Runtime → Restart runtime

# Çözüm 2: Farklı GPU türü dene
# Runtime → Change runtime type → GPU type: T4

# Çözüm 3: Bekleme süresi
# Yoğun saatlerde GPU kuyruğu olabilir, 10-15 dakika bekle
```

### Problem 2: Bağlantı Kopuyor
```python
# Çözüm 1: Colab'ı aktif tut
# Her 30 dakikada bir hücre çalıştır

# Çözüm 2: Checkpoint sistemi
import pickle
import time

def save_checkpoint(data, filename='checkpoint.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"✅ Checkpoint kaydedildi: {filename}")

# Her 100 episode'da checkpoint kaydet
```

### Problem 3: Memory Hatası
```python
# Çözüm 1: Batch size'ı azalt
config['training']['batch_size'] = 16  # 32 yerine

# Çözüm 2: JAX memory ayarı
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

# Çözüm 3: Runtime'ı yeniden başlat
```

### Problem 4: Wandb Bağlantı Sorunu
```python
# Çözüm 1: Offline mode
import os
os.environ['WANDB_MODE'] = 'offline'

# Çözüm 2: Yeniden login
!wandb login --relogin

# Çözüm 3: Wandb'siz çalıştır
# train.py'daki wandb satırlarını yoruma al
```

### Problem 5: Dosya Bulunamıyor
```python
# Mevcut dizini kontrol et
!pwd
!ls -la

# Doğru dizine geç
%cd /content/agi_core_hunter/experiments/01_mdl_vs_ood

# Dosya varlığını kontrol et
import os
print("Manifest var mı?", os.path.exists('manifest.json'))
```

---

## ⏰ Zaman Yönetimi

### Colab Limitleri
- **Ücretsiz:** 12 saat sürekli çalışma
- **Pro ($10/ay):** 24 saat sürekli çalışma
- **Idle timeout:** 90 dakika hareketsizlik sonrası kapanır

### Strateji
```python
# 1. Hızlı demo (2 dk) → Sistem testi
# 2. Pilot deney (15 dk) → Hipotez kontrolü  
# 3. Tam deney (30 dk) → Nihai sonuçlar

# Toplam süre: ~50 dakika (12 saat limitinin çok altında)
```

### Aktif Kalma Hilesi
```python
# Her 30 dakikada çalıştır (yeni hücrede)
import time
import random

for i in range(24):  # 12 saat boyunca
    time.sleep(1800)  # 30 dakika bekle
    print(f"⏰ Aktif kalma: {i+1}/24 - {time.strftime('%H:%M:%S')}")
    
    # Rastgele basit işlem (Colab'ı aktif tutar)
    dummy = sum(range(random.randint(1, 100)))
```

---

## 🎉 Başarı Kontrol Listesi

### Kurulum ✅
- [ ] Google Colab'a giriş yaptım
- [ ] GPU'yu aktifleştirdim (T4 görünüyor)
- [ ] Projeyi indirdim/yükledim
- [ ] Bağımlılıkları yükledim
- [ ] Test scripti başarılı

### Demo ✅
- [ ] Demo çalıştırdım
- [ ] Agent grid'de hareket etti
- [ ] Sıkıştırma oranını gördüm (10x)
- [ ] Hiç hata almadım

### Deney ✅
- [ ] Pilot deney çalıştırdım (500 episode)
- [ ] 3 agent eğitildi (Low, Medium, High β)
- [ ] OOD testleri yapıldı
- [ ] Sonuçlar kaydedildi

### Analiz ✅
- [ ] Sonuçları görselleştirdim
- [ ] Beta-OOD korelasyonunu hesapladım
- [ ] Hipotezi test ettim
- [ ] Sonuçları indirdim

### Paylaşım ✅
- [ ] Sonuçları Google Drive'a kaydettim
- [ ] Blog yazısı için notlar aldım
- [ ] Sonraki deney fikirlerini yazdım
- [ ] Toplulukla paylaştım

---

## 🚀 Sonraki Adımlar

### Hemen Yapılacaklar
1. **Blog yazısı yaz:** "MDL İlkesi: İlk Sonuçlar"
2. **LinkedIn'de paylaş:** Bulgularını duyur
3. **GitHub'ı güncelle:** Sonuçları commit et
4. **Sonraki deneyi planla:** Farklı latent boyutları

### Bu Hafta
1. **Deney 02:** Latent boyutları (2, 4, 8, 16, 32)
2. **Baseline karşılaştırma:** PCA, ICA, Random
3. **Nedensellik agent'ı:** Tasarıma başla
4. **Workshop paper:** Outline hazırla

### Gelecek Hafta
1. **Colab Pro:** Upgrade değerlendir ($10/ay)
2. **Kaggle:** Alternatif platform test et
3. **Topluluk feedback:** İlk yorumları topla
4. **Akademik ağ:** İlgili araştırmacılara ulaş

---

**🎯 Özet:** Google Colab ile 1 saatte AGI ilkeni test edebilir, sonuçları analiz edebilir ve dünyayla paylaşabilirsin!**

**Başarılar! 🚀**