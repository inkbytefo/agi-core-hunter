#!/usr/bin/env python3
"""
Simple WandB Configuration Script for Google Colab

This script configures WandB for the AGI Core Hunter project.
Update the WANDB_API_KEY before running.
"""

import os
import wandb

# 🔑 REPLACE THIS WITH YOUR ACTUAL WANDB API KEY
WANDB_API_KEY = "apikeyhere"  # Get from https://wandb.ai/settings

def setup_wandb():
    """Setup WandB for experiment tracking"""
    print("📊 Setting up WandB (Weights & Biases)...")
    
    if WANDB_API_KEY == "apikeyhere":
        print("⚠️  Please update WANDB_API_KEY in this script!")
        print("   1. Get your API key from: https://wandb.ai/settings")
        print("   2. Replace 'apikeyhere' with your actual key")
        print("   3. Re-run this script")
        print("\n📴 Setting up offline mode for now...")
        
        # Set offline mode
        os.environ['WANDB_MODE'] = 'offline'
        
        # Test offline initialization
        with wandb.init(
            project="agi-core-hunter-colab",
            mode="offline",
            name="setup-test"
        ) as run:
            run.log({"setup": 1})
            print("✅ WandB offline mode configured!")
        
        return False
    
    else:
        print(f"🔐 Configuring WandB with provided API key...")
        
        try:
            # Set API key
            os.environ['WANDB_API_KEY'] = WANDB_API_KEY
            
            # Authenticate
            wandb.login(key=WANDB_API_KEY)
            
            # Test initialization
            with wandb.init(
                project="agi-core-hunter-colab",
                name="setup-test",
                tags=["colab", "setup"]
            ) as run:
                run.log({"setup_complete": 1})
                print(f"✅ WandB authenticated successfully!")
                print(f"🔗 Dashboard: {run.url}")
            
            return True
            
        except Exception as e:
            print(f"❌ WandB authentication failed: {e}")
            print("📴 Falling back to offline mode...")
            
            os.environ['WANDB_MODE'] = 'offline'
            return False

def print_wandb_info():
    """Print WandB configuration info"""
    print("\n📊 WandB Configuration Info:")
    print(f"   Mode: {os.environ.get('WANDB_MODE', 'online')}")
    print(f"   Project: agi-core-hunter-colab")
    
    if os.environ.get('WANDB_MODE') == 'offline':
        print("   📁 Logs: /content/wandb (offline)")
        print("   💡 Tip: Upload logs later with 'wandb sync'")
    else:
        print("   🌐 Dashboard: https://wandb.ai/")

if __name__ == "__main__":
    print("🚀 AGI Core Hunter - WandB Setup")
    print("=" * 40)
    
    success = setup_wandb()
    print_wandb_info()
    
    if not success and WANDB_API_KEY == "apikeyhere":
        print(f"\n🔧 To enable online tracking:")
        print(f"   1. Get API key: https://wandb.ai/settings")
        print(f"   2. Edit this file and replace 'apikeyhere'")
        print(f"   3. Run again: !python scripts/setup_wandb.py")
    
    print("\n✅ WandB setup complete!")