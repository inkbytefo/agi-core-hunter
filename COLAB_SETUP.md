# AGI Core Hunter - Google Colab Setup Commands
# Copy and paste these commands in Google Colab cells

## Method 1: Complete Setup Script (Recommended)
```python
# 1. Clone the repository (if not already done)
!git clone https://github.com/inkbytefo/agi-core-hunter.git
%cd agi-core-hunter

# 2. Run complete setup
!python scripts/colab_setup_complete.py
```

## Method 2: Quick Setup (Alternative)
```python
# Quick setup using the existing script
!python colab_quickstart.py
```

## Method 3: Manual Steps (If scripts fail)
```python
# Install modern JAX with CUDA 12 support
!pip install -U "jax[cuda12]" flax optax chex wandb tqdm matplotlib seaborn pandas -q

# Verify installation
import jax
print(f"JAX version: {jax.__version__}")
print(f"Devices: {jax.devices()}")

# Set up WandB (replace 'your_api_key_here' with actual key)
import wandb
import os
os.environ['WANDB_API_KEY'] = 'apikeyhere'  # Replace this
# wandb.login()  # Uncomment after setting real API key

# Test project setup
!python test_setup.py
```

## Next Steps After Setup:
```python
# Quick demo (2 minutes)
%cd experiments/01_mdl_vs_ood
!python demo.py

# Fast experiment (5 minutes)
!python train.py --manifest manifest.json
```

## Troubleshooting:
- If GPU not detected: Runtime → Change runtime type → GPU
- If JAX fails: Restart runtime and try again
- If WandB fails: Use offline mode by setting WANDB_MODE='offline'