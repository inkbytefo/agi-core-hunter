#!/usr/bin/env python3
"""
Demo script for Free Energy Principle vs Exploration experiment

This script provides an interactive demonstration of how FEP agents
implement active inference, showing surprise minimization, exploration,
and belief updating in real-time.
"""

import sys
import time
from pathlib import Path
import argparse

import jax
import jax.numpy as jnp
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from agents.fep_agent import FEPAgent
from agents.mdl_agent import MDLAgent
from envs.active_inference_world import ActiveInferenceWorld


def demo_single_agent(agent_name: str, agent_config: dict, env_config: dict, episodes: int = 3):
    """Run demo for a single agent"""
    print(f"\n{'='*60}")
    print(f"🧠 DEMONSTRATING: {agent_name}")
    print(f"{'='*60}")
    
    # Create environment and agent
    env = ActiveInferenceWorld(**env_config)
    
    if "FEP" in agent_name:
        agent = FEPAgent(agent_config)
    else:
        agent = MDLAgent(agent_config)
    
    # Setup agent
    rng_key = jax.random.PRNGKey(42)
    dummy_obs = jnp.zeros(env.obs_dim)
    agent.setup(rng_key, dummy_obs)
    
    for episode in range(episodes):
        print(f"\n📺 Episode {episode + 1}/{episodes}")
        print("-" * 40)
        
        # Reset environment
        key, rng_key = jax.random.split(rng_key)
        env_state, observation = env.reset(key)
        agent.reset()
        
        episode_reward = 0.0
        step = 0
        
        print("🎬 Starting episode...")
        print(env.render_ascii(env_state))
        
        while not env_state.done and step < 50:  # Limit steps for demo
            # Agent acts
            action_key, key = jax.random.split(key)
            action, info = agent.act(observation, action_key)
            
            # Environment step
            step_key, key = jax.random.split(key)
            next_env_state, next_observation, reward, done, env_info = env.step(
                env_state, int(action), step_key
            )
            
            episode_reward += reward
            step += 1
            
            # Print step information
            if step % 10 == 0 or env_info.get("goal_reached", False) or step <= 5:
                print(f"\n⏰ Step {step}:")
                print(f"   Action: {int(action)} ({['Up', 'Right', 'Down', 'Left'][int(action)]}) → Reward: {reward:.2f}")
                print(f"   Goal reached: {env_info.get('goal_reached', False)}")
                print(f"   Exploration bonus: {env_info.get('exploration_bonus', 0):.2f}")
                print(f"   Surprise level: {env_info.get('surprise_level', 0):.2f}")
                print(f"   Exploration ratio: {env_info.get('exploration_ratio', 0):.1%}")
                
                # Show FEP-specific metrics if available
                if hasattr(agent, 'get_metrics'):
                    metrics = agent.get_metrics()
                    print(f"   💡 FEP Metrics:")
                    print(f"      VFE: {metrics.get('variational_free_energy', 0):.3f}")
                    print(f"      Epistemic: {metrics.get('epistemic_value', 0):.3f}")
                    print(f"      Pragmatic: {metrics.get('pragmatic_value', 0):.3f}")
                    print(f"      Surprise: {metrics.get('surprise', 0):.3f}")
                    print(f"      Belief entropy: {metrics.get('belief_entropy', 0):.3f}")
            
            # Update for next step
            observation = next_observation
            env_state = next_env_state
            
            if env_info.get("goal_reached", False):
                print("\n🎉 Goal reached!")
                break
        
        print(f"\n📊 Episode Summary:")
        print(f"   Total reward: {episode_reward:.2f}")
        print(f"   Steps taken: {step}")
        print(f"   Goal reached: {env_info.get('goal_reached', False)}")
        print(f"   Final exploration: {env_info.get('exploration_ratio', 0):.1%}")
        
        if episode < episodes - 1:
            print("\n⏳ Press Enter for next episode...")
            input()


def demo_comparison(env_config: dict):
    """Compare different agents side by side"""
    print(f"\n{'='*80}")
    print(f"🥊 AGENT COMPARISON DEMO")
    print(f"{'='*80}")
    
    agents_config = [
        {
            "name": "FEP_Explorer",
            "config": {
                "obs_dim": 0,  # Will be set
                "action_dim": 4,
                "hidden_dim": 16,
                "precision": 2.0,
                "epistemic_weight": 3.0,  # High exploration
                "learning_rate": 3e-4
            }
        },
        {
            "name": "FEP_Balanced", 
            "config": {
                "obs_dim": 0,  # Will be set
                "action_dim": 4,
                "hidden_dim": 16,
                "precision": 1.0,
                "epistemic_weight": 1.0,  # Balanced
                "learning_rate": 3e-4
            }
        },
        {
            "name": "MDL_Baseline",
            "config": {
                "obs_dim": 0,  # Will be set
                "action_dim": 4,
                "latent_dim": 16,
                "beta": 1.0,
                "learning_rate": 3e-4
            }
        }
    ]
    
    # Create environment
    env = ActiveInferenceWorld(**env_config)
    
    # Update obs_dim for all agents
    for agent_cfg in agents_config:
        agent_cfg["config"]["obs_dim"] = env.obs_dim
    
    results = {}
    
    for agent_cfg in agents_config:
        print(f"\n🤖 Testing {agent_cfg['name']}...")
        
        # Create and setup agent
        if "FEP" in agent_cfg["name"]:
            agent = FEPAgent(agent_cfg["config"])
        else:
            agent = MDLAgent(agent_cfg["config"])
        
        rng_key = jax.random.PRNGKey(42)
        dummy_obs = jnp.zeros(env.obs_dim)
        agent.setup(rng_key, dummy_obs)
        
        # Run test episode
        key, rng_key = jax.random.split(rng_key)
        env_state, observation = env.reset(key)
        agent.reset()
        
        episode_reward = 0.0
        episode_steps = 0
        exploration_total = 0.0
        surprise_total = 0.0
        
        while not env_state.done and episode_steps < 100:
            # Agent acts
            action_key, key = jax.random.split(key)
            action, info = agent.act(observation, action_key)
            
            # Environment step
            step_key, key = jax.random.split(key)
            next_env_state, next_observation, reward, done, env_info = env.step(
                env_state, int(action), step_key
            )
            
            episode_reward += reward
            episode_steps += 1
            exploration_total += env_info.get("exploration_bonus", 0)
            surprise_total += env_info.get("surprise_level", 0)
            
            # Update for next step
            observation = next_observation
            env_state = next_env_state
            
            if env_info.get("goal_reached", False):
                break
        
        # Store results
        results[agent_cfg["name"]] = {
            "reward": episode_reward,
            "steps": episode_steps,
            "success": env_info.get("goal_reached", False),
            "exploration": exploration_total / max(episode_steps, 1),
            "surprise": surprise_total / max(episode_steps, 1),
            "exploration_ratio": env_info.get("exploration_ratio", 0)
        }
        
        # Get FEP metrics if available
        if hasattr(agent, 'get_metrics'):
            fep_metrics = agent.get_metrics()
            results[agent_cfg["name"]].update(fep_metrics)
    
    # Print comparison table
    print(f"\n📊 COMPARISON RESULTS")
    print("-" * 80)
    print(f"{'Agent':<15} {'Success':<8} {'Reward':<10} {'Steps':<8} {'Explor.':<10} {'Surprise':<10}")
    print("-" * 80)
    
    for agent_name, result in results.items():
        success_icon = "✅" if result["success"] else "❌"
        print(f"{agent_name:<15} {success_icon:<8} {result['reward']:<10.1f} {result['steps']:<8} "
              f"{result['exploration']:<10.2f} {result['surprise']:<10.2f}")
    
    # Print FEP-specific metrics
    print(f"\n🧠 FEP-SPECIFIC METRICS")
    print("-" * 80)
    print(f"{'Agent':<15} {'VFE':<10} {'Epistemic':<12} {'Pragmatic':<12} {'Belief H':<10}")
    print("-" * 80)
    
    for agent_name, result in results.items():
        vfe = result.get('variational_free_energy', 0)
        epistemic = result.get('epistemic_value', 0)
        pragmatic = result.get('pragmatic_value', 0)
        belief_h = result.get('belief_entropy', 0)
        
        print(f"{agent_name:<15} {vfe:<10.3f} {epistemic:<12.3f} {pragmatic:<12.3f} {belief_h:<10.3f}")
    
    return results


def demo_uncertainty_handling():
    """Demo how FEP agents handle uncertainty"""
    print(f"\n{'='*80}")
    print(f"🌫️ UNCERTAINTY HANDLING DEMO")
    print(f"{'='*80}")
    
    # High uncertainty environment
    uncertain_config = {
        "height": 8,
        "width": 8,
        "uncertainty_prob": 0.6,  # High uncertainty
        "hidden_reward_prob": 0.15,
        "observation_noise_std": 0.4,  # High noise
        "max_steps": 100,
        "reward_goal": 20.0,
        "reward_hidden": 5.0,
        "reward_step": -0.1,
        "reward_exploration": 2.0,  # High exploration reward
        "surprise_threshold": 1.5,
        "goal_move_prob": 0.05,  # Dynamic goal
        "fep_strength": 2.0  # Strong FEP dynamics
    }
    
    # FEP agent optimized for uncertainty
    fep_config = {
        "obs_dim": 0,  # Will be set
        "action_dim": 4,
        "hidden_dim": 20,
        "precision": 1.5,
        "epistemic_weight": 2.5,  # Strong exploration drive
        "learning_rate": 3e-4
    }
    
    print("🔬 Testing FEP agent in high uncertainty environment...")
    print("   Features: High observation noise, dynamic goals, surprise events")
    
    demo_single_agent("FEP_UncertaintyExpert", fep_config, uncertain_config, episodes=2)


def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(description="FEP vs Exploration Demo")
    parser.add_argument(
        "--mode",
        choices=["single", "comparison", "uncertainty", "all"],
        default="comparison",
        help="Demo mode to run"
    )
    
    args = parser.parse_args()
    
    print("🎭 FREE ENERGY PRINCIPLE DEMO")
    print("=" * 50)
    print("This demo showcases active inference and FEP agents")
    print("in action, demonstrating surprise minimization and")
    print("intelligent exploration behaviors.")
    
    # Standard environment config
    env_config = {
        "height": 8,
        "width": 8,
        "uncertainty_prob": 0.3,
        "hidden_reward_prob": 0.1,
        "observation_noise_std": 0.2,
        "max_steps": 150,
        "reward_goal": 20.0,
        "reward_hidden": 5.0,
        "reward_step": -0.1,
        "reward_exploration": 1.0,
        "surprise_threshold": 2.0,
        "goal_move_prob": 0.02,
        "fep_strength": 1.0
    }
    
    if args.mode == "single" or args.mode == "all":
        # Demo single FEP agent
        fep_config = {
            "obs_dim": env_config["height"] * env_config["width"] * 3 + 8,
            "action_dim": 4,
            "hidden_dim": 16,
            "precision": 1.0,
            "epistemic_weight": 2.0,
            "learning_rate": 3e-4
        }
        demo_single_agent("FEP_Demo", fep_config, env_config)
    
    if args.mode == "comparison" or args.mode == "all":
        # Demo agent comparison
        demo_comparison(env_config)
    
    if args.mode == "uncertainty" or args.mode == "all":
        # Demo uncertainty handling
        demo_uncertainty_handling()
    
    print(f"\n🎉 Demo completed! Try running the full experiment with:")
    print(f"   python train.py")


if __name__ == "__main__":
    main()