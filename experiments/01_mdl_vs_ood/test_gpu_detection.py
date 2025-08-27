#!/usr/bin/env python3
"""
Test script to verify GPU detection fix
"""

import jax
import jax.numpy as jnp

def test_gpu_detection():
    """Test the updated GPU detection logic"""
    print("🔧 Testing GPU Detection Fix")
    print("=" * 40)
    
    # Get all devices
    devices = jax.devices()
    print(f"All JAX devices: {devices}")
    
    # Test old vs new detection method
    old_gpu_devices = [d for d in devices if hasattr(d, 'device_kind') and d.device_kind == 'gpu']
    new_gpu_devices = [d for d in devices if d.platform == 'cuda']
    
    print(f"\nOld method (device_kind == 'gpu'): {len(old_gpu_devices)} devices")
    for i, d in enumerate(old_gpu_devices):
        print(f"  GPU {i}: {d}")
    
    print(f"\nNew method (platform == 'cuda'): {len(new_gpu_devices)} devices")
    for i, d in enumerate(new_gpu_devices):
        print(f"  CUDA {i}: {d}")
    
    # Check if we have any CUDA devices
    if new_gpu_devices:
        print(f"\n✅ GPU detection fix successful!")
        print(f"   Found {len(new_gpu_devices)} CUDA device(s)")
        
        # Test a simple computation
        try:
            test_array = jax.device_put(jnp.ones((100, 100)), new_gpu_devices[0])
            result = jnp.dot(test_array, test_array.T)
            print(f"   ✅ GPU computation test passed: {jnp.mean(result)}")
        except Exception as e:
            print(f"   ⚠️ GPU computation test failed: {e}")
    else:
        print(f"\n❌ No CUDA devices found")
        print(f"   This could mean:")
        print(f"   1. No GPU available")
        print(f"   2. JAX not installed with CUDA support")
        print(f"   3. CUDA drivers not properly installed")
    
    return len(new_gpu_devices) > 0

if __name__ == "__main__":
    test_gpu_detection()