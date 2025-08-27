#!/usr/bin/env python3
"""
Test script to verify the updated setup_jax_devices function
"""

import sys
import os
from pathlib import Path

# Add src to path like train.py does
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

import jax
import jax.numpy as jnp
import time

def setup_jax_devices():
    """Initialize and verify JAX devices for optimal performance"""
    devices = jax.devices()
    print(f"🔧 JAX Devices Available: {devices}")
    
    # Debug: Show detailed device information
    print("🔍 Device Details:")
    for i, device in enumerate(devices):
        print(f"   Device {i}: {device}")
        print(f"     Platform: '{device.platform}'")
        print(f"     Device Kind: '{getattr(device, 'device_kind', 'N/A')}'")
        print(f"     ID: {device.id}")
    
    # Robust GPU detection that works with multiple JAX versions
    def is_gpu_device(device):
        """Check if device is a GPU using multiple detection methods"""
        # Method 1: Check if platform is 'cuda' (older JAX versions)
        if device.platform == 'cuda':
            return True
        
        # Method 2: Check if platform is 'gpu' (newer JAX versions)
        if device.platform == 'gpu':
            return True
            
        # Method 3: Check if device string contains 'cuda' (fallback)
        if 'cuda' in str(device).lower():
            return True
            
        # Method 4: Check device_kind for 'gpu' (compatibility)
        if hasattr(device, 'device_kind') and device.device_kind == 'gpu':
            return True
            
        return False
    
    gpu_devices = [d for d in devices if is_gpu_device(d)]
    print(f"🔍 GPU Detection: Found {len(gpu_devices)} devices using robust detection")
    
    if gpu_devices:
        print(f"✅ GPU devices found: {len(gpu_devices)} GPU(s)")
        print(f"   Primary GPU: {gpu_devices[0]}")
        
        # Tesla T4 specific information
        primary_gpu = gpu_devices[0]
        if 'tesla' in str(primary_gpu).lower() or 'tesla' in getattr(primary_gpu, 'device_kind', '').lower():
            print(f"   🚀 Tesla T4 GPU detected - optimized for ML workloads!")
            print(f"   Device details: Platform='{primary_gpu.platform}', Kind='{getattr(primary_gpu, 'device_kind', 'N/A')}'")
        
        # Set default device for computations
        try:
            jax.config.update('jax_default_device', gpu_devices[0])
            print(f"   ✅ Default device set to: {gpu_devices[0]}")
        except Exception as e:
            print(f"   ⚠️ Could not set default device: {e}")
        
        # Test GPU with simple computation
        try:
            test_array = jnp.ones((1000, 1000))
            start_time = time.time()
            result = jnp.dot(test_array, test_array.T)
            # Force computation and check device
            result_device = result.device()
            compute_time = time.time() - start_time
            gflops = (2 * 1000**3) / compute_time / 1e9
            
            print(f"   ✅ GPU computation test passed on device: {result_device}")
            print(f"   Performance: {compute_time:.4f}s, {gflops:.1f} GFLOPS")
            print(f"   Result mean: {jnp.mean(result):.1f} (expected: 1000.0)")
            
            # Verify we're actually using GPU
            if 'cuda' in str(result_device).lower():
                print(f"   🎆 CONFIRMED: Computations running on CUDA GPU!")
            else:
                print(f"   ⚠️ WARNING: Computation may not be on GPU - device: {result_device}")
                
        except Exception as e:
            print(f"   ⚠️ GPU computation test failed: {e}")
        
    else:
        print("⚠️  No GPU devices found, using CPU. Performance will be slower.")
        print("   For GPU support, install with: pip install -U 'jax[cuda12]'")
        
        # Try alternative detection methods for debugging
        print("🔍 Trying alternative detection methods:")
        alt_gpu_devices = [d for d in devices if 'gpu' in str(d).lower()]
        print(f"   String-based detection: {len(alt_gpu_devices)} devices")
        alt_cuda_devices = [d for d in devices if 'cuda' in str(d).lower()]
        print(f"   CUDA string detection: {len(alt_cuda_devices)} devices")
    
    return devices, gpu_devices

if __name__ == "__main__":
    print("🧪 Testing Updated setup_jax_devices Function")
    print("=" * 50)
    
    devices, gpu_devices = setup_jax_devices()
    
    print(f"\n📊 Final Results:")
    print(f"   Total devices: {len(devices)}")
    print(f"   GPU devices: {len(gpu_devices)}")
    print(f"   Setup Status: {'✅ SUCCESS' if gpu_devices else '❌ NO GPU'}")
    
    if gpu_devices:
        print(f"   Ready for AGI Core Hunter training with GPU acceleration!")
    else:
        print(f"   Will use CPU - consider checking GPU allocation")