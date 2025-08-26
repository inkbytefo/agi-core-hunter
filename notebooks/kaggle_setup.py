#!/usr/bin/env python3
"""
Kaggle Notebooks için hızlı kurulum scripti
"""

import os
import subprocess
import sys

def setup_kaggle():
    """Kaggle ortamında projeyi kur"""
    
    print("🚀 Kaggle Notebooks - AGI Core Hunter Kurulumu")
    
    # 1. GPU kontrolü
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        print("✅ GPU mevcut:", "Tesla" in result.stdout or "P100" in result.stdout)
    except:
        print("❌ GPU bulunamadı")
    
    # 2. Bağımlılıkları yükle (Modern CUDA 12 desteği ile)
    packages = [
        "jax[cuda12]",  # Updated for CUDA 12 support
        "flax", 
        "optax",
        "chex",
        "wandb",
        "tqdm"
    ]
    
    for package in packages:
        print(f"📦 Yükleniyor: {package}")
        subprocess.run([sys.executable, "-m", "pip", "install", package, "-q"])
    
    # 3. Proje dosyalarını kopyala (Kaggle dataset olarak yüklenmiş olmalı)
    if os.path.exists("/kaggle/input/agi-core-hunter"):
        print("📁 Proje dosyaları bulundu")
        subprocess.run(["cp", "-r", "/kaggle/input/agi-core-hunter", "/kaggle/working/"])
        os.chdir("/kaggle/working/agi-core-hunter")
    else:
        print("❌ Proje dosyaları bulunamadı. Dataset olarak yükleyin.")
        return False
    
    # 4. Test kurulumu
    try:
        subprocess.run([sys.executable, "test_setup.py"], check=True)
        print("✅ Kurulum başarılı!")
        return True
    except:
        print("❌ Kurulum başarısız")
        return False

def run_fast_experiment():
    """Hızlı deney çalıştır"""
    
    print("\n⚡ Hızlı Deney Başlatılıyor...")
    
    # Manifest'i düzenle
    import json
    
    manifest_path = "experiments/01_mdl_vs_ood/manifest.json"
    with open(manifest_path, 'r') as f:
        config = json.load(f)
    
    # Hızlı test parametreleri
    config['training']['total_episodes'] = 300
    config['training']['eval_frequency'] = 50
    
    fast_manifest = "experiments/01_mdl_vs_ood/manifest_kaggle.json"
    with open(fast_manifest, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Deneyi çalıştır
    os.chdir("experiments/01_mdl_vs_ood")
    subprocess.run([sys.executable, "train.py", "--manifest", "manifest_kaggle.json"])
    
    print("✅ Deney tamamlandı!")

if __name__ == "__main__":
    if setup_kaggle():
        run_fast_experiment()