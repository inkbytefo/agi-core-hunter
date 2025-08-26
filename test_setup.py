#!/usr/bin/env python3
"""
Quick test script to verify the project setup works correctly
"""

import sys
from pathlib import Path
import jax
import jax.numpy as jnp

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        from agents.mdl_agent import MDLAgent
        print("✅ MDLAgent import successful")
    except Exception as e:
        print(f"❌ MDLAgent import failed: {e}")
        return False
    
    try:
        from envs.grid_world import GridWorld
        print("✅ GridWorld import successful")
    except Exception as e:
        print(f"❌ GridWorld import failed: {e}")
        return False
    
    try:
        from utils.metrics import MetricsTracker
        print("✅ MetricsTracker import successful")
    except Exception as e:
        print(f"❌ MetricsTracker import failed: {e}")
        return False
    
    return True


def test_environment():
    """Test basic environment functionality"""
    print("\n🌍 Testing environment...")
    
    try:
        from envs.grid_world import GridWorld
        
        env = GridWorld(height=5, width=5)
        rng_key = jax.random.PRNGKey(42)
        
        # Test reset
        state, obs = env.reset(rng_key)
        print(f"✅ Environment reset successful, obs shape: {obs.shape}")
        
        # Test step
        action = 1  # Right
        new_state, new_obs, reward, done, info = env.step(state, action)
        print(f"✅ Environment step successful, reward: {reward}")
        
        # Test ASCII rendering
        ascii_render = env.render_ascii(state)
        print("✅ ASCII rendering successful")
        print("Sample grid:")
        print(ascii_render[:100] + "..." if len(ascii_render) > 100 else ascii_render)
        
        return True
        
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        return False


def test_agent():
    """Test basic agent functionality"""
    print("\n🤖 Testing agent...")
    
    try:
        from agents.mdl_agent import MDLAgent
        from envs.grid_world import GridWorld
        
        # Create environment to get dimensions
        env = GridWorld(height=5, width=5)
        
        # Create agent
        config = {
            "obs_dim": env.obs_dim,
            "action_dim": env.action_dim,
            "latent_dim": 4,
            "beta": 1.0,
            "learning_rate": 3e-4
        }
        
        agent = MDLAgent(config)
        print("✅ Agent creation successful")
        
        # Setup agent
        rng_key = jax.random.PRNGKey(42)
        dummy_obs = jnp.zeros(env.obs_dim)
        agent.setup(rng_key, dummy_obs)
        print("✅ Agent setup successful")
        
        # Test action selection
        key1, key2 = jax.random.split(rng_key)
        state, obs = env.reset(key1)
        action, info = agent.act(obs, key2)
        print(f"✅ Action selection successful, action: {action}")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent test failed: {e}")
        return False


def test_metrics():
    """Test metrics utilities"""
    print("\n📊 Testing metrics...")
    
    try:
        from utils.metrics import MetricsTracker, calculate_ood_performance_ratio
        
        # Test metrics tracker
        tracker = MetricsTracker()
        tracker.update(reward=10.0, length=50, success=True)
        tracker.update(reward=8.0, length=45, success=False)
        
        summary = tracker.get_summary()
        print(f"✅ MetricsTracker successful, mean reward: {summary.get('mean_reward', 'N/A')}")
        
        # Test OOD ratio calculation
        ratio = calculate_ood_performance_ratio(0.8, 0.6)
        print(f"✅ OOD ratio calculation successful: {ratio}")
        
        return True
        
    except Exception as e:
        print(f"❌ Metrics test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🚀 AGI Core Hunter - Setup Test")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_environment,
        test_agent,
        test_metrics
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 40)
    print(f"📈 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! Project setup is working correctly.")
        print("\n🚀 Ready to run experiments:")
        print("   cd experiments/01_mdl_vs_ood")
        print("   python train.py")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())