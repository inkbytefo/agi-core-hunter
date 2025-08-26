#!/usr/bin/env python3
"""
Simple demo script to test the MDL agent without full training
"""

import sys
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from agents.mdl_agent import MDLAgent
from envs.grid_world import GridWorld


def run_demo():
    """Run a simple demo of the MDL agent"""
    print("🚀 AGI Core Hunter - MDL Agent Demo")
    print("=" * 40)
    
    # Create environment
    env = GridWorld(height=6, width=6, obstacle_prob=0.15)
    print(f"✅ Environment created: {env.height}x{env.width} grid")
    
    # Create agent
    config = {
        "obs_dim": env.obs_dim,
        "action_dim": env.action_dim,
        "latent_dim": 4,
        "beta": 1.0,
        "learning_rate": 3e-4
    }
    
    agent = MDLAgent(config)
    print(f"✅ Agent created with latent_dim={config['latent_dim']}, β={config['beta']}")
    
    # Setup agent
    rng_key = jax.random.PRNGKey(42)
    dummy_obs = jnp.zeros(env.obs_dim)
    agent.setup(rng_key, dummy_obs)
    print("✅ Agent setup complete")
    
    # Run a few episodes
    print("\n🎮 Running demo episodes...")
    
    for episode in range(3):
        print(f"\n--- Episode {episode + 1} ---")
        
        # Reset environment
        key, rng_key = jax.random.split(rng_key)
        state, obs = env.reset(key)
        
        print("Initial state:")
        print(env.render_ascii(state))
        
        total_reward = 0
        steps = 0
        max_steps = 20
        
        while not state.done and steps < max_steps:
            # Agent selects action
            key, rng_key = jax.random.split(rng_key)
            action, info = agent.act(obs, key)
            
            # Environment step
            state, obs, reward, done, env_info = env.step(state, int(action))
            
            total_reward += reward
            steps += 1
            
            # Print action and result
            action_names = ["Up", "Right", "Down", "Left"]
            print(f"Step {steps}: {action_names[int(action)]} -> Reward: {reward:.2f}")
            
            if env_info.get("goal_reached", False):
                print("🎯 Goal reached!")
                break
            elif env_info.get("hit_obstacle", False):
                print("💥 Hit obstacle!")
        
        print(f"Episode {episode + 1} finished:")
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Steps taken: {steps}")
        print(f"  Success: {'Yes' if env_info.get('goal_reached', False) else 'No'}")
        
        print("Final state:")
        print(env.render_ascii(state))
    
    # Test compression
    print("\n🗜️ Testing compression...")
    key, rng_key = jax.random.split(rng_key)
    state, obs = env.reset(key)
    
    key, rng_key = jax.random.split(rng_key)
    z, mean, log_var = agent.encode(obs, key)
    
    print(f"Original observation dim: {obs.shape[0]}")
    print(f"Compressed latent dim: {z.shape[0]}")
    print(f"Compression ratio: {obs.shape[0] / z.shape[0]:.1f}x")
    print(f"Latent representation: {z[:4]}...")  # Show first 4 values
    
    print("\n✅ Demo completed successfully!")
    print("\n🚀 Next steps:")
    print("  - Run full training: python train.py")
    print("  - Analyze results: jupyter notebook eval.ipynb")


if __name__ == "__main__":
    run_demo()