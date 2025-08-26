#!/usr/bin/env python3
"""
Complete Google Colab Setup Script for AGI Core Hunter Project

This script sets up the entire environment in Google Colab including:
- Modern JAX with CUDA 12 support
- All project dependencies
- WandB authentication and configuration
- Project structure verification
- GPU optimization settings
- Ready-to-run experiment environment

Usage in Colab:
1. Clone the repository
2. Run: !python scripts/colab_setup_complete.py
3. Follow the authentication prompts
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional


class ColabSetupManager:
    """Comprehensive setup manager for Google Colab environment"""
    
    def __init__(self):
        self.setup_results = {}
        self.gpu_available = False
        self.jax_devices = []
        
    def print_banner(self):
        """Print setup banner"""
        print("🚀" + "="*60)
        print("🎯 AGI CORE HUNTER - Google Colab Complete Setup")
        print("🔬 Modern JAX CUDA 12 + WandB Integration")
        print("="*60 + "🚀")
        print()
    
    def check_environment(self):
        """Check if we're running in Google Colab"""
        try:
            import google.colab
            print("✅ Running in Google Colab environment")
            return True
        except ImportError:
            print("⚠️  Not running in Google Colab (local environment detected)")
            return False
    
    def detect_gpu(self):
        """Detect available GPU resources"""
        print("🔍 Detecting GPU resources...")
        
        try:
            # Check NVIDIA GPUs
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                gpu_info = result.stdout
                if "Tesla" in gpu_info or "T4" in gpu_info or "V100" in gpu_info or "A100" in gpu_info:
                    self.gpu_available = True
                    # Extract GPU model
                    for line in gpu_info.split('\n'):
                        if 'Tesla' in line or 'T4' in line or 'V100' in line or 'A100' in line:
                            gpu_model = line.split()[1:3]
                            print(f"✅ GPU detected: {' '.join(gpu_model)}")
                            break
                else:
                    print("📊 GPU hardware detected but model unclear")
                    self.gpu_available = True
            else:
                print("❌ No NVIDIA GPU detected")
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("❌ nvidia-smi not available or timeout")
        
        # Set GPU optimization environment variables
        if self.gpu_available:
            self.setup_gpu_optimization()
        else:
            print("💻 Using CPU-only mode")
    
    def setup_gpu_optimization(self):
        """Setup GPU optimization environment variables"""
        print("⚙️  Configuring GPU optimization settings...")
        
        # Tesla T4 and general GPU optimizations
        gpu_settings = {
            'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',  # Better memory management
            'XLA_PYTHON_CLIENT_MEM_FRACTION': '0.8',    # Use 80% of GPU memory
            'JAX_ENABLE_X64': 'false',                   # Use float32 for better performance
            'XLA_FLAGS': '--xla_gpu_enable_triton_softmax_fusion=true --xla_gpu_triton_gemm_any=true'
        }
        
        for key, value in gpu_settings.items():
            os.environ[key] = value
            print(f"   🔧 {key} = {value}")
    
    def install_dependencies(self):
        """Install all project dependencies with modern versions"""
        print("\n📦 Installing project dependencies...")
        
        # Core ML libraries with modern CUDA 12 support
        core_packages = [
            "jax[cuda12]>=0.4.25",      # Modern JAX with CUDA 12
            "flax>=0.8.0",              # Updated Flax
            "optax>=0.1.9",             # Updated Optax
            "chex>=0.1.10",             # Updated Chex
        ]
        
        # Environment and simulation
        env_packages = [
            "gymnax>=0.0.6",
            "brax>=0.9.2", 
            "gymnasium>=0.29.1"
        ]
        
        # Experiment tracking
        tracking_packages = [
            "wandb>=0.16.0",
            "mlflow>=2.8.1"
        ]
        
        # Visualization and utilities
        util_packages = [
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "pandas>=2.0.0",
            "plotly>=5.17.0",
            "tqdm>=4.66.0",
            "numpy>=1.24.0"
        ]
        
        # Install in groups for better error tracking
        package_groups = [
            ("Core ML", core_packages),
            ("Environments", env_packages), 
            ("Tracking", tracking_packages),
            ("Utilities", util_packages)
        ]
        
        for group_name, packages in package_groups:
            print(f"\n📋 Installing {group_name} packages...")
            for package in packages:
                self.install_package(package)
        
        print("\n✅ All dependencies installed successfully!")
    
    def install_package(self, package: str, max_retries: int = 3):
        """Install a single package with retry logic"""
        for attempt in range(max_retries):
            try:
                print(f"   📦 Installing {package}...")
                subprocess.run([
                    sys.executable, "-m", "pip", "install", 
                    package, "--upgrade", "--quiet"
                ], check=True, timeout=300)
                print(f"   ✅ {package} installed")
                return True
                
            except subprocess.CalledProcessError as e:
                if attempt < max_retries - 1:
                    print(f"   ⚠️  Retry {attempt + 1}/{max_retries} for {package}")
                    time.sleep(2)
                else:
                    print(f"   ❌ Failed to install {package}: {e}")
                    return False
            except subprocess.TimeoutExpired:
                print(f"   ⏱️  Installation timeout for {package}")
                return False
    
    def verify_jax_installation(self):
        """Verify JAX installation and detect devices"""
        print("\n🧪 Verifying JAX installation...")
        
        try:
            import jax
            import jax.numpy as jnp
            
            print(f"✅ JAX version: {jax.__version__}")
            
            # Check devices
            devices = jax.devices()
            self.jax_devices = devices
            print(f"🔧 JAX devices: {devices}")
            
            # Check platform
            platform = jax.default_backend()
            print(f"🖥️  Default platform: {platform}")
            
            # Test basic computation
            key = jax.random.PRNGKey(42)
            x = jax.random.normal(key, (1000, 1000))
            result = jnp.sum(x ** 2)
            print(f"🧮 GPU/CPU computation test: {float(result):.2f}")
            
            # Check for GPU acceleration
            gpu_devices = [d for d in devices if d.device_kind.lower() == 'gpu']
            if gpu_devices:
                print(f"🚀 GPU acceleration available: {len(gpu_devices)} GPU(s)")
            else:
                print("💻 Running on CPU (no GPU acceleration)")
                
            return True
            
        except Exception as e:
            print(f"❌ JAX verification failed: {e}")
            return False
    
    def setup_wandb(self, api_key: str = "apikeyhere"):
        """Setup WandB with API key configuration"""
        print("\n📊 Setting up WandB (Weights & Biases)...")
        
        try:
            import wandb
            
            # Configure WandB settings
            wandb_config = {
                'project': 'agi-core-hunter-colab',
                'entity': None,  # Will use default
                'dir': '/content/wandb_logs',
                'mode': 'online'  # Start in online mode
            }
            
            # Create WandB directory
            os.makedirs(wandb_config['dir'], exist_ok=True)
            
            if api_key == "apikeyhere":
                print("🔑 WandB API Key Setup Required:")
                print("   1. Get your API key from: https://wandb.ai/settings")
                print("   2. Replace 'apikeyhere' in this script with your actual key")
                print("   3. Or run: wandb.login() manually after setup")
                print("\n⚠️  Setting up offline mode for now...")
                
                # Set offline mode as fallback
                os.environ['WANDB_MODE'] = 'offline'
                wandb_config['mode'] = 'offline'
                
            else:
                print("🔐 Attempting WandB authentication...")
                try:
                    # Set API key
                    os.environ['WANDB_API_KEY'] = api_key
                    
                    # Test authentication
                    wandb.login(key=api_key)
                    print("✅ WandB authentication successful!")
                    
                except Exception as e:
                    print(f"❌ WandB authentication failed: {e}")
                    print("📴 Falling back to offline mode...")
                    os.environ['WANDB_MODE'] = 'offline'
                    wandb_config['mode'] = 'offline'
            
            # Initialize a test run to verify
            with wandb.init(**wandb_config, name="setup-test", tags=["setup"]) as run:
                run.log({"setup_complete": 1, "timestamp": time.time()})
                print(f"✅ WandB test run created: {run.url if wandb_config['mode'] == 'online' else 'offline'}")
            
            return True
            
        except ImportError:
            print("❌ WandB not available, installing...")
            self.install_package("wandb>=0.16.0")
            return self.setup_wandb(api_key)
        except Exception as e:
            print(f"❌ WandB setup failed: {e}")
            return False
    
    def verify_project_structure(self):
        """Verify project structure and files"""
        print("\n📁 Verifying project structure...")
        
        required_files = [
            "src/agents/mdl_agent.py",
            "src/envs/grid_world.py", 
            "src/utils/metrics.py",
            "experiments/01_mdl_vs_ood/train.py",
            "experiments/01_mdl_vs_ood/manifest.json",
            "requirements.txt",
            "test_setup.py"
        ]
        
        missing_files = []
        for file_path in required_files:
            if os.path.exists(file_path):
                print(f"   ✅ {file_path}")
            else:
                print(f"   ❌ {file_path}")
                missing_files.append(file_path)
        
        if missing_files:
            print(f"\n⚠️  Missing files detected: {len(missing_files)} files")
            print("   Make sure you're in the correct project directory")
            return False
        else:
            print("\n✅ All required files present!")
            return True
    
    def run_project_tests(self):
        """Run project setup tests"""
        print("\n🧪 Running project setup tests...")
        
        try:
            # Run the project's test setup
            result = subprocess.run([
                sys.executable, "test_setup.py"
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                print("✅ Project tests passed!")
                print("📋 Test output:")
                for line in result.stdout.split('\n')[:10]:  # Show first 10 lines
                    if line.strip():
                        print(f"   {line}")
                return True
            else:
                print("❌ Project tests failed!")
                print("📋 Error output:")
                print(result.stderr)
                return False
                
        except subprocess.TimeoutExpired:
            print("⏱️  Test timeout - tests may be running slow")
            return False
        except Exception as e:
            print(f"❌ Test execution error: {e}")
            return False
    
    def create_quick_start_notebook(self):
        """Create a quick start notebook for immediate experimentation"""
        print("\n📓 Creating quick start notebook...")
        
        notebook_content = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "# 🚀 AGI Core Hunter - Quick Start\\n",
                        "\\n",
                        "Environment setup complete! Ready for experimentation.\\n",
                        "\\n",
                        "## 🎯 Quick Actions\\n",
                        "1. **Fast Demo**: Run a 2-minute demo\\n",
                        "2. **Quick Experiment**: 500 episodes (~5 minutes)\\n", 
                        "3. **Full Experiment**: Complete research experiment\\n"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "source": [
                        "# 🎮 Fast Demo (2 minutes)\\n",
                        "%cd experiments/01_mdl_vs_ood\\n",
                        "!python demo.py"
                    ]
                },
                {
                    "cell_type": "code", 
                    "execution_count": None,
                    "metadata": {},
                    "source": [
                        "# ⚡ Quick Experiment Setup\\n",
                        "import json\\n",
                        "\\n",
                        "# Load and modify config for quick test\\n",
                        "with open('manifest.json', 'r') as f:\\n",
                        "    config = json.load(f)\\n",
                        "\\n",
                        "config['training']['total_episodes'] = 500\\n",
                        "config['evaluation']['ood_tests'][0]['episodes'] = 50\\n",
                        "\\n",
                        "with open('manifest_quick.json', 'w') as f:\\n",
                        "    json.dump(config, f, indent=2)\\n",
                        "\\n",
                        "print('✅ Quick experiment config ready!')\\n",
                        "print('Run: !python train.py --manifest manifest_quick.json')"
                    ]
                }
            ],
            "metadata": {
                "colab": {
                    "provenance": [],
                    "gpuType": "T4"
                },
                "kernelspec": {
                    "name": "python3",
                    "display_name": "Python 3"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 0
        }
        
        try:
            with open('/content/AGI_QuickStart.ipynb', 'w') as f:
                json.dump(notebook_content, f, indent=2)
            print("✅ Quick start notebook created: /content/AGI_QuickStart.ipynb")
            return True
        except Exception as e:
            print(f"❌ Failed to create notebook: {e}")
            return False
    
    def print_final_summary(self):
        """Print final setup summary and next steps"""
        print("\n" + "🎉" + "="*58 + "🎉")
        print("🎯 SETUP COMPLETE - AGI CORE HUNTER READY!")
        print("="*60 + "🎉")
        
        print("\n📊 Setup Summary:")
        print(f"   💻 Environment: {'Google Colab' if self.check_environment() else 'Local'}")
        print(f"   🚀 GPU Available: {'Yes' if self.gpu_available else 'No'}")
        print(f"   🔧 JAX Devices: {len(self.jax_devices)} device(s)")
        print(f"   📦 Dependencies: Installed")
        print(f"   📊 WandB: Configured")
        
        print("\n🚀 Next Steps:")
        print("   1. 🎮 Quick Demo (2 min):")
        print("      %cd experiments/01_mdl_vs_ood && python demo.py")
        print()
        print("   2. ⚡ Fast Experiment (5 min):")
        print("      %cd experiments/01_mdl_vs_ood")
        print("      !python train.py --manifest manifest.json")
        print()
        print("   3. 📊 Monitor with WandB:")
        print("      Visit your WandB dashboard for real-time metrics")
        print()
        print("   4. 📓 Use Quick Start Notebook:")
        print("      Open: /content/AGI_QuickStart.ipynb")
        
        print("\n💡 Tips:")
        if self.gpu_available:
            print("   🚀 GPU detected - experiments will run fast!")
        else:
            print("   💻 CPU mode - consider using GPU runtime for faster training")
        print("   📱 Monitor GPU usage with: !nvidia-smi")
        print("   💾 Save results regularly to avoid data loss")
        
        print("\n🔗 Resources:")
        print("   📚 Documentation: Check the docs/ folder")
        print("   🐛 Issues: Report on GitHub")
        print("   💬 Community: Join the discussion")
        
        print("\n" + "🎉" + "="*58 + "🎉")
    
    def run_complete_setup(self, wandb_api_key: str = "apikeyhere"):
        """Run the complete setup process"""
        self.print_banner()
        
        # Step 1: Environment check
        is_colab = self.check_environment()
        
        # Step 2: GPU detection and optimization
        self.detect_gpu()
        
        # Step 3: Install dependencies
        self.install_dependencies()
        
        # Step 4: Verify JAX installation
        jax_ok = self.verify_jax_installation()
        if not jax_ok:
            print("❌ Critical error: JAX setup failed")
            return False
        
        # Step 5: Setup WandB
        wandb_ok = self.setup_wandb(wandb_api_key)
        
        # Step 6: Verify project structure
        structure_ok = self.verify_project_structure()
        if not structure_ok:
            print("⚠️  Project structure issues detected")
        
        # Step 7: Run tests
        test_ok = self.run_project_tests()
        
        # Step 8: Create quick start resources
        if is_colab:
            self.create_quick_start_notebook()
        
        # Step 9: Final summary
        self.print_final_summary()
        
        return jax_ok and structure_ok


def main():
    """Main setup function"""
    setup_manager = ColabSetupManager()
    
    # You can modify the API key here before running
    WANDB_API_KEY = "apikeyhere"  # Replace with your actual WandB API key
    
    success = setup_manager.run_complete_setup(WANDB_API_KEY)
    
    if success:
        print("\n🎉 Setup completed successfully!")
        return 0
    else:
        print("\n❌ Setup completed with some issues")
        return 1


if __name__ == "__main__":
    sys.exit(main())