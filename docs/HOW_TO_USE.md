# AGI Çekirdek Avcısı - Kullanım Kılavuzu

**Sürüm:** 1.0  
**Tarih:** 26 Ağustos 2025  
**Hedef Kitle:** Araştırmacılar, ML Mühendisleri, AGI Meraklıları

---

## 🚀 Hızlı Başlangıç

### Sistem Gereksinimleri
- **Python:** 3.9+
- **RAM:** 8GB+ (16GB önerilen)
- **GPU:** Opsiyonel (CUDA destekli)
- **Disk:** 5GB boş alan

### Kurulum
```bash
# 1. Projeyi klonla
git clone https://github.com/[username]/agi_core_hunter.git
cd agi_core_hunter

# 2. Sanal ortam oluştur (önerilen)
python -m venv agi_env
source agi_env/bin/activate  # Linux/Mac
# veya
agi_env\Scripts\activate     # Windows

# 3. Bağımlılıkları yükle
pip install -r requirements.txt

# 4. Kurulumu test et
python test_setup.py
```

### İlk Demo
```bash
# Basit demo çalıştır
python experiments/01_mdl_vs_ood/demo.py
```

---

## 🧪 Deney Çalıştırma

### Mevcut Deneyi Çalıştırma

#### 1. MDL vs OOD Deneyi
```bash
cd experiments/01_mdl_vs_ood

# Tam eğitim (5000 episode, ~30 dakika)
python train.py

# Hızlı test (100 episode, ~2 dakika)
python train.py --episodes 100 --eval-freq 25
```

#### 2. Sonuçları Analiz Etme
```bash
# Jupyter notebook ile analiz
jupyter notebook eval.ipynb

# Veya otomatik rapor
python -c "
import json
with open('results.json') as f:
    results = json.load(f)
print('Deney tamamlandı!')
for agent, data in results.items():
    print(f'{agent}: {data[\"standard_success_rate\"]:.3f}')
"
```

### Wandb Entegrasyonu

#### Wandb Kurulumu
```bash
# Wandb hesabı oluştur: https://wandb.ai
wandb login

# Veya offline mod
export WANDB_MODE=offline
```

#### Canlı Takip
- **Dashboard:** https://wandb.ai/[username]/agi_core_hunter
- **Metrikler:** total_reward, success_rate, reconstruction_loss, kl_loss
- **Karşılaştırma:** Farklı β değerleri arasında

---

## 🔧 Yeni Deney Oluşturma

### 1. Deney Klasörü Yapısı
```bash
mkdir experiments/[deney_adi]
cd experiments/[deney_adi]

# Gerekli dosyalar
touch manifest.json    # Deney konfigürasyonu
touch train.py         # Eğitim scripti
touch eval.ipynb       # Analiz notebook'u
touch README.md        # Deney açıklaması
```

### 2. Manifest Dosyası Örneği
```json
{
  "experiment": {
    "name": "Yeni Deney Adı",
    "version": "1.0",
    "description": "Deney açıklaması",
    "hypothesis": "Test edilecek hipotez"
  },
  "environment": {
    "type": "GridWorld",
    "config": {
      "height": 8,
      "width": 8,
      "obstacle_prob": 0.2
    }
  },
  "agents": [
    {
      "name": "TestAgent",
      "type": "MDLAgent",
      "config": {
        "latent_dim": 8,
        "beta": 1.0,
        "learning_rate": 3e-4
      }
    }
  ],
  "training": {
    "total_episodes": 1000,
    "eval_frequency": 100
  }
}
```

### 3. Eğitim Scripti Şablonu
```python
#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from experiments.base_experiment import ExperimentRunner

def main():
    runner = ExperimentRunner("manifest.json")
    results = runner.run_experiment()
    return results

if __name__ == "__main__":
    main()
```

---

## 🤖 Yeni Agent Oluşturma

### 1. Agent Sınıfı Şablonu
```python
# src/agents/my_agent.py
from core.base_agent import BaseAgent
import jax.numpy as jnp

class MyAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        # Agent-specific initialization
    
    def setup(self, rng_key, dummy_obs):
        # Initialize networks and optimizers
        pass
    
    def act(self, observation, rng_key):
        # Select action given observation
        action = jnp.array(0)  # Placeholder
        info = {"value": 0.0}
        return action, info
    
    def update(self, batch):
        # Update agent parameters
        metrics = {"loss": 0.0}
        return metrics
    
    def get_metrics(self):
        return {"training_step": 0}
```

### 2. Agent'ı Kaydetme
```python
# src/agents/__init__.py
from .mdl_agent import MDLAgent
from .my_agent import MyAgent

__all__ = ['MDLAgent', 'MyAgent']
```

### 3. Teori Kartı Oluşturma
```markdown
# docs/theory_cards/MY_PRINCIPLE.md

## 🎯 Temel İddia
"Zekâ [ilke açıklaması]"

## 📐 Matematiksel Formülasyon
[Formüller ve denklemler]

## 🔬 Test Edilebilir Öngörüler
1. [Öngörü 1]
2. [Öngörü 2]

## 🧪 Minimal Test Ortamı
[Test senaryosu açıklaması]

## 📊 Başarı Metrikleri
[Ölçüm kriterleri]
```

---

## 🌍 Yeni Ortam Oluşturma

### 1. Ortam Sınıfı Şablonu
```python
# src/envs/my_environment.py
import jax.numpy as jnp
from typing import Tuple, Dict, Any

class MyEnvironment:
    def __init__(self, config):
        self.config = config
        self.action_dim = 4
        self.obs_dim = 10
    
    def reset(self, rng_key):
        # Initialize environment state
        state = None  # Your state representation
        observation = jnp.zeros(self.obs_dim)
        return state, observation
    
    def step(self, state, action):
        # Execute one step
        new_state = state
        observation = jnp.zeros(self.obs_dim)
        reward = 0.0
        done = False
        info = {}
        return new_state, observation, reward, done, info
```

### 2. OOD Test Senaryoları
```python
def generate_ood_config(self, rng_key, ood_type):
    """Generate out-of-distribution configuration"""
    if ood_type == "parameter_shift":
        return {"param": self.param * 1.5}
    elif ood_type == "structure_change":
        return {"structure": "modified"}
    else:
        raise ValueError(f"Unknown OOD type: {ood_type}")
```

---

## 📊 Sonuç Analizi

### Temel Metrikler
```python
from src.utils.metrics import (
    calculate_ood_performance_ratio,
    calculate_sample_efficiency,
    MetricsTracker
)

# Metrik hesaplama
ood_ratio = calculate_ood_performance_ratio(
    standard_performance=0.8,
    ood_performance=0.6
)

# Metrik takibi
tracker = MetricsTracker()
tracker.update(reward=10.0, success=True)
summary = tracker.get_summary()
```

### Görselleştirme
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Learning curve
plt.plot(episodes, rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Learning Curve')

# OOD comparison
sns.barplot(data=df, x='Agent', y='OOD_Ratio')
plt.title('OOD Performance Comparison')
```

---

## 🔍 Debugging ve Troubleshooting

### Yaygın Sorunlar

#### 1. Import Hataları
```bash
# Çözüm: PYTHONPATH ayarla
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Veya sys.path kullan
import sys
sys.path.append('src')
```

#### 2. JAX/GPU Sorunları
```python
# GPU kontrolü
import jax
print(f"JAX devices: {jax.devices()}")

# CPU'ya zorla
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
```

#### 3. Wandb Bağlantı Sorunları
```bash
# Offline mode
export WANDB_MODE=offline

# Yeniden login
wandb login --relogin
```

### Debug Modları
```bash
# Verbose logging
python train.py --verbose

# Debug mode (küçük dataset)
python train.py --debug --episodes 10

# Profiling
python -m cProfile train.py
```

---

## 🧪 Test ve Validasyon

### Unit Testler
```bash
# Tüm testleri çalıştır
python -m pytest tests/

# Specific test
python -m pytest tests/test_agents.py

# Coverage raporu
python -m pytest --cov=src tests/
```

### Integration Testler
```bash
# Tam pipeline testi
python test_setup.py

# Deney end-to-end testi
python experiments/01_mdl_vs_ood/demo.py
```

### Performance Testler
```bash
# Benchmark çalıştır
python benchmarks/speed_test.py

# Memory profiling
python -m memory_profiler train.py
```

---

## 📚 İleri Düzey Kullanım

### Batch Deney Çalıştırma
```bash
# Çoklu deney paralel çalıştırma
python scripts/batch_experiments.py \
  --experiments "01_mdl_vs_ood,02_latent_dims" \
  --parallel 2
```

### Hyperparameter Sweep
```python
# Wandb sweep konfigürasyonu
sweep_config = {
    'method': 'grid',
    'parameters': {
        'beta': {'values': [0.1, 1.0, 5.0]},
        'latent_dim': {'values': [4, 8, 16]}
    }
}

# Sweep başlat
wandb.sweep(sweep_config, project="agi_core_hunter")
```

### Distributed Training
```python
# JAX pmap kullanımı
from jax import pmap
import jax.numpy as jnp

# Multi-GPU eğitim
@pmap
def train_step(state, batch):
    # Training logic
    return new_state, metrics
```

---

## 🤝 Katkıda Bulunma

### Yeni İlke Önerme
1. **Issue aç:** "New Principle: [İlke Adı]"
2. **Teori kartı yaz:** `docs/theory_cards/[ILKE].md`
3. **Agent implementasyonu:** `src/agents/[ilke]_agent.py`
4. **Test deneyi:** `experiments/[xx]_[ilke]_test/`
5. **Pull request gönder**

### Kod Katkısı
```bash
# Fork ve clone
git clone https://github.com/[username]/agi_core_hunter.git
cd agi_core_hunter

# Feature branch
git checkout -b feature/new-principle

# Değişiklikleri yap
# ...

# Test et
python test_setup.py
python -m pytest

# Commit ve push
git add .
git commit -m "Add new principle: [İlke Adı]"
git push origin feature/new-principle

# Pull request oluştur
```

### Dokümantasyon Katkısı
- Typo düzeltmeleri
- Örnek ekleme
- Tutorial yazma
- Çeviri (İngilizce)

---

## 📞 Destek ve İletişim

### Dokümantasyon
- **Proje Tanıtımı:** `docs/PROJE_TANITIM.md`
- **Teknik Mimari:** `docs/PROJE_TEKNIKMIMARI.md`
- **Yol Haritası:** `ROADMAP.md`
- **Teori Kartları:** `docs/theory_cards/`

### İletişim Kanalları
- **GitHub Issues:** Bug report ve feature request
- **GitHub Discussions:** Genel sorular ve tartışma
- **Email:** [proje-email@example.com]
- **Twitter:** [@agi_core_hunter]

### Topluluk
- **Discord:** [Davet linki]
- **Reddit:** r/agi_core_hunter
- **LinkedIn:** AGI Core Hunter grubu

---

## 📄 Lisans ve Atıf

### Lisans
Bu proje MIT lisansı altında yayınlanmıştır. Detaylar için `LICENSE` dosyasına bakın.

### Atıf
```bibtex
@software{agi_core_hunter,
  title={AGI Core Hunter: Systematic Search for AGI Principles},
  author={[Yazarlar]},
  year={2025},
  url={https://github.com/[username]/agi_core_hunter}
}
```

### Teşekkürler
- JAX/Flax ekibi
- Weights & Biases
- Açık kaynak topluluğu

---

**Son Güncelleme:** 26 Ağustos 2025  
**Versiyon:** 1.0  
**Sonraki Güncelleme:** Faz 2 tamamlandığında