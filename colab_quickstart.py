#!/usr/bin/env python3
"""
Google Colab Hızlı Başlangıç Scripti
Bu dosyayı Colab'a yükle ve çalıştır
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def setup_colab_environment():
    """Colab ortamını hazırla"""
    
    print("🚀 AGI Core Hunter - Colab Hızlı Kurulum")
    print("=" * 50)
    
    # 1. GPU kontrolü
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if 'Tesla' in result.stdout or 'T4' in result.stdout:
            print("✅ GPU mevcut ve hazır!")
        else:
            print("⚠️  GPU bulunamadı, Runtime → Change runtime type → GPU seçin")
            return False
    except:
        print("❌ GPU kontrolü başarısız")
        return False
    
    # 2. Bağımlılıkları yükle
    print("\n📦 Bağımlılıklar yükleniyor...")
    packages = [
        "jax[cuda12]>=0.4.25",  # Updated for CUDA 12 support
        "flax>=0.8.0", 
        "optax>=0.1.9",
        "chex>=0.1.10",
        "wandb>=0.16.0",
        "tqdm>=4.66.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pandas>=2.0.0"
    ]
    
    for package in packages:
        print(f"  Installing {package.split('>=')[0]}...")
        subprocess.run([sys.executable, "-m", "pip", "install", package, "-q"], 
                      check=True)
    
    print("✅ Bağımlılıklar yüklendi!")
    
    # 3. Proje yapısını kontrol et
    required_files = [
        "src/agents/mdl_agent.py",
        "src/envs/grid_world.py", 
        "src/core/base_agent.py",
        "experiments/01_mdl_vs_ood/manifest.json",
        "test_setup.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n❌ Eksik dosyalar: {missing_files}")
        print("📁 Lütfen tüm proje dosyalarını Colab'a yükleyin")
        return False
    
    print("✅ Proje dosyları tamam!")
    
    # 4. Test kurulumu
    try:
        result = subprocess.run([sys.executable, "test_setup.py"], 
                              capture_output=True, text=True, check=True)
        print("✅ Kurulum testi başarılı!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Kurulum testi başarısız: {e.stderr}")
        return False

def create_fast_config():
    """Hızlı test için konfigürasyon oluştur"""
    
    print("\n⚡ Hızlı test konfigürasyonu hazırlanıyor...")
    
    manifest_path = "experiments/01_mdl_vs_ood/manifest.json"
    
    with open(manifest_path, 'r') as f:
        config = json.load(f)
    
    # Hızlı test parametreleri
    config['training']['total_episodes'] = 300
    config['training']['eval_frequency'] = 75
    config['training']['batch_size'] = 32
    
    # OOD testlerini azalt
    for ood_test in config['evaluation']['ood_tests']:
        ood_test['episodes'] = 25
    
    # Hızlı manifest kaydet
    fast_manifest = "experiments/01_mdl_vs_ood/manifest_colab_fast.json"
    with open(fast_manifest, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Hızlı konfigürasyon hazır: {fast_manifest}")
    print(f"📊 Toplam episode: {config['training']['total_episodes'] * len(config['agents'])}")
    print("⏱️ Tahmini süre: ~10 dakika")
    
    return fast_manifest

def run_demo():
    """Hızlı demo çalıştır"""
    
    print("\n🎮 Demo çalıştırılıyor...")
    
    try:
        result = subprocess.run([
            sys.executable, 
            "experiments/01_mdl_vs_ood/demo.py"
        ], capture_output=True, text=True, check=True)
        
        print("✅ Demo başarılı!")
        print("Demo çıktısı:")
        print("-" * 30)
        print(result.stdout[-500:])  # Son 500 karakter
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Demo başarısız: {e.stderr}")
        return False

def setup_wandb():
    """Wandb kurulumu"""
    
    print("\n📊 Wandb kurulumu...")
    
    choice = input("Wandb kullanmak istiyor musunuz? (y/n): ").lower()
    
    if choice == 'y':
        print("Wandb API key'inizi girin (wandb.ai/authorize adresinden alabilirsiniz):")
        try:
            import wandb
            wandb.login()
            print("✅ Wandb başarıyla kuruldu!")
            return True
        except:
            print("❌ Wandb kurulumu başarısız, offline mode kullanılacak")
    
    # Offline mode
    os.environ['WANDB_MODE'] = 'offline'
    print("📊 Wandb offline mode aktif")
    return False

def main():
    """Ana fonksiyon"""
    
    # 1. Ortam kurulumu
    if not setup_colab_environment():
        print("\n❌ Kurulum başarısız, lütfen sorunları çözün ve tekrar deneyin")
        return
    
    # 2. Demo çalıştır
    if not run_demo():
        print("\n❌ Demo başarısız, devam edilemiyor")
        return
    
    # 3. Wandb kurulumu
    setup_wandb()
    
    # 4. Hızlı konfigürasyon
    fast_manifest = create_fast_config()
    
    # 5. Kullanıcıya seçenekler sun
    print("\n🎯 Sonraki adımlar:")
    print("1. Hızlı deney çalıştır (10 dakika)")
    print("2. Tam deney çalıştır (30 dakika)")
    print("3. Sadece analiz yap (mevcut sonuçlar varsa)")
    
    choice = input("\nSeçiminiz (1/2/3): ")
    
    if choice == '1':
        print("\n⚡ Hızlı deney başlatılıyor...")
        os.chdir("experiments/01_mdl_vs_ood")
        subprocess.run([sys.executable, "train.py", "--manifest", "manifest_colab_fast.json"])
        
    elif choice == '2':
        print("\n🔥 Tam deney başlatılıyor...")
        print("⚠️  Bu ~30 dakika sürecek, Colab'ı kapatmayın!")
        os.chdir("experiments/01_mdl_vs_ood")
        subprocess.run([sys.executable, "train.py"])
        
    elif choice == '3':
        print("\n📊 Analiz moduna geçiliyor...")
        if os.path.exists("experiments/01_mdl_vs_ood/results.json"):
            print("✅ Sonuç dosyası bulundu, analiz başlayabilir")
        else:
            print("❌ Sonuç dosyası bulunamadı, önce deney çalıştırın")
    
    print("\n🎉 Kurulum tamamlandı!")
    print("📚 Detaylı rehber için: docs/GOOGLE_COLAB_REHBER.md")
    print("🚀 İyi deneyler!")

if __name__ == "__main__":
    main()