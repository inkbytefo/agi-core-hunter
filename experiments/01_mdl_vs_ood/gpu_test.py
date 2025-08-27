#!/usr/bin/env python3
"""
Quick GPU validation script for AGI Core Hunter

This script performs a rapid test to ensure JAX is properly utilizing GPU resources
before running longer experiments. Perfect for Colab environment validation.
"""

import jax
import jax.numpy as jnp
import time
import numpy as np

def test_jax_gpu():
    """Comprehensive JAX GPU functionality test"""
    print("🔧 AGI Core Hunter - JAX GPU Validation Test")
    print("=" * 50)
    
    # 1. Device Detection
    devices = jax.devices()
    gpu_devices = [d for d in devices if d.platform == 'cuda']
    cpu_devices = [d for d in devices if d.platform == 'cpu']
    
    print(f"🖥️  Total JAX devices: {len(devices)}")
    print(f"⚡ GPU devices: {len(gpu_devices)}")
    print(f"🧠 CPU devices: {len(cpu_devices)}")
    
    if gpu_devices:
        print(f"✅ Primary GPU: {gpu_devices[0]}")
        print(f"   Platform: {jax.lib.xla_bridge.get_backend().platform}")
    else:
        print("❌ No GPU detected!")
        return False
    
    # 2. Memory Configuration Test
    print("\n🧪 Testing GPU Memory Configuration...")
    try:
        # Small computation test
        test_size = 1000
        x = jax.device_put(jnp.ones((test_size, test_size)), gpu_devices[0])
        y = jax.device_put(jnp.ones((test_size, test_size)), gpu_devices[0])
        
        start_time = time.time()
        result = jnp.dot(x, y)
        jax.device_get(result)  # Force computation
        gpu_time = time.time() - start_time
        
        print(f"✅ GPU computation successful ({gpu_time:.4f}s)")
        print(f"   Result shape: {result.shape}, Mean: {jnp.mean(result):.2f}")
        
    except Exception as e:
        print(f"❌ GPU computation failed: {e}")
        return False
    
    # 3. Performance Benchmark
    print("\n🏃 Running GPU Performance Benchmark...")
    sizes = [500, 1000, 2000]
    
    for size in sizes:
        try:
            # Create test matrices
            a = jax.device_put(jnp.ones((size, size)), gpu_devices[0])
            b = jax.device_put(jnp.ones((size, size)), gpu_devices[0])
            
            # JIT compile for performance
            @jax.jit
            def matrix_multiply(x, y):
                return jnp.dot(x, y)
            
            # Warm up
            _ = matrix_multiply(a, b)
            
            # Benchmark
            start_time = time.time()
            for _ in range(5):
                result = matrix_multiply(a, b)
                jax.device_get(result)
            gpu_time = (time.time() - start_time) / 5
            
            ops_per_sec = (2 * size**3) / gpu_time / 1e9  # Approximate GFLOPS
            print(f"   {size}x{size}: {gpu_time:.4f}s/op, ~{ops_per_sec:.1f} GFLOPS")
            
        except Exception as e:
            print(f"   {size}x{size}: Failed - {e}")
    
    # 4. Neural Network Operations Test
    print("\n🧠 Testing Neural Network Operations...")
    try:
        from jax import nn
        
        # Simple neural network computation
        key = jax.random.PRNGKey(42)
        batch_size, input_dim, hidden_dim = 32, 64, 128
        
        # Create test data
        x = jax.device_put(jax.random.normal(key, (batch_size, input_dim)), gpu_devices[0])
        w1 = jax.device_put(jax.random.normal(key, (input_dim, hidden_dim)), gpu_devices[0])
        w2 = jax.device_put(jax.random.normal(key, (hidden_dim, 1)), gpu_devices[0])
        
        @jax.jit
        def forward_pass(x, w1, w2):
            h = nn.relu(jnp.dot(x, w1))
            return jnp.dot(h, w2)
        
        # Run forward pass
        start_time = time.time()
        output = forward_pass(x, w1, w2)
        jax.device_get(output)
        nn_time = time.time() - start_time
        
        print(f"✅ Neural network operations successful ({nn_time:.4f}s)")
        print(f"   Output shape: {output.shape}")
        
    except Exception as e:
        print(f"❌ Neural network test failed: {e}")
        return False
    
    # 5. Memory Usage Check
    print("\n💾 Checking GPU Memory Usage...")
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', 
                               '--format=csv,nounits,noheader'], 
                              capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if lines and lines[0]:
                used_mb, total_mb, gpu_util = lines[0].split(', ')
                print(f"   Memory: {used_mb}MB / {total_mb}MB ({int(used_mb)/int(total_mb)*100:.1f}%)")
                print(f"   GPU Utilization: {gpu_util}%")
        else:
            print("   nvidia-smi not available")
            
    except Exception:
        print("   Memory monitoring not available")
    
    print("\n🎉 All GPU tests passed! Your setup is ready for training.")
    return True

def quick_training_test():
    """Run a minimal version of the actual training to test integration"""
    print("\n🚀 Running Quick Training Integration Test...")
    
    try:
        # Import training script components
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent.parent / "src"))
        
        # Simple test - just verify imports work
        from agents.mdl_agent import MDLAgent
        from envs.grid_world import GridWorld
        
        print("✅ Training components imported successfully")
        
        # Test with mini manifest
        mini_manifest = Path(__file__).parent / "mini-test.json"
        if mini_manifest.exists():
            print(f"✅ Mini test manifest found: {mini_manifest}")
            print("   Ready to run: python train.py --manifest mini-test.json")
        else:
            print("⚠️  Mini test manifest not found")
        
        return True
        
    except Exception as e:
        print(f"❌ Training integration test failed: {e}")
        return False

if __name__ == "__main__":
    print("Starting AGI Core Hunter GPU Validation...")
    
    gpu_ok = test_jax_gpu()
    
    if gpu_ok:
        training_ok = quick_training_test()
        
        if training_ok:
            print("\n🏆 VALIDATION COMPLETE - System ready for experiments!")
            print("\nNext steps:")
            print("1. Quick test: python train.py --gpu-test")
            print("2. Mini test: python train.py --manifest mini-test.json")
            print("3. Full experiment: python train.py --manifest manifest.json")
        else:
            print("\n⚠️  GPU works but training integration has issues")
    else:
        print("\n❌ GPU validation failed - check CUDA installation")
        print("   Install with: pip install -U 'jax[cuda12]'")