#!/usr/bin/env python3
"""
Lokal makine için optimizasyon scripti
Deneyleri daha hızlı çalıştırmak için çeşitli optimizasyonlar
"""

import json
import os
import sys
import argparse
from pathlib import Path

def create_fast_config(original_manifest: str, output_manifest: str, speed_level: int = 1):
    """
    Hızlı test için konfigürasyon oluştur
    
    speed_level:
    1 = Hızlı test (100 episode, ~2 dakika)
    2 = Orta test (500 episode, ~10 dakika) 
    3 = Tam test (5000 episode, ~60 dakika)
    """
    
    with open(original_manifest, 'r') as f:
        config = json.load(f)
    
    if speed_level == 1:  # Hızlı test
        config['training']['total_episodes'] = 100
        config['training']['eval_frequency'] = 25
        config['training']['batch_size'] = 16
        for ood_test in config['evaluation']['ood_tests']:
            ood_test['episodes'] = 20
        print("⚡ Hızlı test konfigürasyonu (100 episode)")
        
    elif speed_level == 2:  # Orta test
        config['training']['total_episodes'] = 500
        config['training']['eval_frequency'] = 100
        config['training']['batch_size'] = 32
        for ood_test in config['evaluation']['ood_tests']:
            ood_test['episodes'] = 50
        print("🚀 Orta test konfigürasyonu (500 episode)")
        
    elif speed_level == 3:  # Tam test
        config['training']['batch_size'] = 64  # Daha büyük batch
        print("🔥 Tam test konfigürasyonu (5000 episode)")
    
    # Ortam optimizasyonları
    config['environment']['config']['max_steps'] = min(
        config['environment']['config']['max_steps'], 30
    )
    
    with open(output_manifest, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Optimized config saved: {output_manifest}")

def setup_jax_optimization():
    """JAX performans optimizasyonları"""
    
    optimizations = {
        # Memory preallocation'ı kapat (daha esnek)
        'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
        
        # Compilation cache (tekrar çalıştırmalarda hızlanır)
        'XLA_PYTHON_CLIENT_MEM_FRACTION': '0.8',
        
        # CPU parallelism
        'XLA_FLAGS': '--xla_cpu_multi_thread_eigen=false --xla_cpu_use_thunk_runtime=false',
        
        # JAX compilation cache
        'JAX_COMPILATION_CACHE_DIR': str(Path.home() / '.jax_cache'),
    }
    
    for key, value in optimizations.items():
        os.environ[key] = value
        print(f"🔧 {key} = {value}")
    
    # Cache dizinini oluştur
    cache_dir = Path.home() / '.jax_cache'
    cache_dir.mkdir(exist_ok=True)
    
    print("✅ JAX optimizasyonları uygulandı")

def estimate_runtime(episodes: int, agents: int = 3):
    """Çalışma süresi tahmini"""
    
    # Ortalama süre tahminleri (saniye/episode)
    base_time_per_episode = 0.5  # Grid world için
    
    total_episodes = episodes * agents
    estimated_seconds = total_episodes * base_time_per_episode
    
    minutes = estimated_seconds / 60
    hours = minutes / 60
    
    print(f"⏱️  Tahmini süre:")
    print(f"   Episodes: {total_episodes}")
    print(f"   Dakika: {minutes:.1f}")
    print(f"   Saat: {hours:.2f}")
    
    return estimated_seconds

def check_system_resources():
    """Sistem kaynaklarını kontrol et"""
    
    import psutil
    
    # CPU bilgisi
    cpu_count = psutil.cpu_count()
    cpu_percent = psutil.cpu_percent(interval=1)
    
    # Memory bilgisi
    memory = psutil.virtual_memory()
    memory_gb = memory.total / (1024**3)
    memory_available_gb = memory.available / (1024**3)
    
    print(f"💻 Sistem Kaynakları:")
    print(f"   CPU: {cpu_count} core, %{cpu_percent:.1f} kullanım")
    print(f"   RAM: {memory_gb:.1f}GB total, {memory_available_gb:.1f}GB available")
    
    # GPU kontrolü
    try:
        import jax
        devices = jax.devices()
        print(f"   JAX devices: {devices}")
        
        if any('gpu' in str(d).lower() for d in devices):
            print("   ✅ GPU mevcut")
        else:
            print("   ❌ GPU bulunamadı (CPU mode)")
    except:
        print("   ❌ JAX yüklü değil")
    
    # Öneriler
    if memory_available_gb < 4:
        print("⚠️  Uyarı: Düşük RAM, batch_size'ı azaltın")
    
    if cpu_percent > 80:
        print("⚠️  Uyarı: Yüksek CPU kullanımı, diğer programları kapatın")

def run_parallel_experiments(manifest_files: list, max_parallel: int = 2):
    """Paralel deney çalıştırma"""
    
    import subprocess
    import threading
    import time
    
    def run_experiment(manifest_file):
        """Tek deney çalıştır"""
        experiment_dir = Path(manifest_file).parent
        cmd = [sys.executable, "train.py", "--manifest", Path(manifest_file).name]
        
        print(f"🚀 Başlatılıyor: {manifest_file}")
        
        try:
            result = subprocess.run(
                cmd, 
                cwd=experiment_dir,
                capture_output=True,
                text=True,
                timeout=3600  # 1 saat timeout
            )
            
            if result.returncode == 0:
                print(f"✅ Tamamlandı: {manifest_file}")
            else:
                print(f"❌ Hata: {manifest_file}")
                print(result.stderr)
                
        except subprocess.TimeoutExpired:
            print(f"⏰ Timeout: {manifest_file}")
        except Exception as e:
            print(f"💥 Exception: {manifest_file} - {e}")
    
    # Paralel çalıştırma
    threads = []
    for i in range(0, len(manifest_files), max_parallel):
        batch = manifest_files[i:i+max_parallel]
        
        batch_threads = []
        for manifest_file in batch:
            thread = threading.Thread(target=run_experiment, args=(manifest_file,))
            thread.start()
            batch_threads.append(thread)
        
        # Bu batch'in bitmesini bekle
        for thread in batch_threads:
            thread.join()
        
        print(f"📊 Batch {i//max_parallel + 1} tamamlandı")

def main():
    parser = argparse.ArgumentParser(description="Lokal optimizasyon araçları")
    parser.add_argument("--speed", type=int, choices=[1,2,3], default=1,
                       help="Hız seviyesi (1=hızlı, 2=orta, 3=tam)")
    parser.add_argument("--manifest", type=str, 
                       default="experiments/01_mdl_vs_ood/manifest.json",
                       help="Orijinal manifest dosyası")
    parser.add_argument("--output", type=str,
                       help="Çıktı manifest dosyası")
    parser.add_argument("--check-system", action="store_true",
                       help="Sistem kaynaklarını kontrol et")
    parser.add_argument("--setup-jax", action="store_true",
                       help="JAX optimizasyonlarını uygula")
    parser.add_argument("--estimate", action="store_true",
                       help="Çalışma süresi tahmini")
    
    args = parser.parse_args()
    
    print("🛠️  AGI Core Hunter - Lokal Optimizasyon")
    print("=" * 50)
    
    if args.check_system:
        check_system_resources()
        print()
    
    if args.setup_jax:
        setup_jax_optimization()
        print()
    
    if args.manifest:
        if not args.output:
            manifest_path = Path(args.manifest)
            args.output = manifest_path.parent / f"manifest_speed{args.speed}.json"
        
        create_fast_config(args.manifest, args.output, args.speed)
        
        if args.estimate:
            with open(args.output, 'r') as f:
                config = json.load(f)
            episodes = config['training']['total_episodes']
            agents = len(config['agents'])
            estimate_runtime(episodes, agents)
    
    print("\n🎯 Kullanım örnekleri:")
    print("python scripts/optimize_local.py --speed 1 --check-system")
    print("python scripts/optimize_local.py --speed 2 --setup-jax --estimate")
    print("python scripts/optimize_local.py --speed 3 --manifest experiments/01_mdl_vs_ood/manifest.json")

if __name__ == "__main__":
    main()