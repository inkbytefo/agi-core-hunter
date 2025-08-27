#!/usr/bin/env python3
"""
Training script for Causality vs Correlation experiment

This script implements the core training loop for testing whether
causal reasoning improves adaptation to interventions and transfers
better than correlation-based learning.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import argparse

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
import wandb

# Configure JAX for optimal GPU usage
try:
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'
    jax.config.update('jax_enable_x64', False)
except Exception as e:
    print(f"⚠️ JAX config warning: {e}")
    print("🔄 Falling back to default JAX configuration...")

def setup_jax_devices():
    """Initialize and verify JAX devices for optimal performance"""
    devices = jax.devices()
    print(f"🔧 JAX Devices Available: {devices}")
    
    def is_gpu_device(device):
        """Check if device is a GPU using multiple detection methods"""
        return (device.platform == 'cuda' or device.platform == 'gpu' or 
                'cuda' in str(device).lower() or 
                (hasattr(device, 'device_kind') and device.device_kind == 'gpu'))
    
    gpu_devices = [d for d in devices if is_gpu_device(d)]
    print(f"🔍 GPU Detection: Found {len(gpu_devices)} GPU devices")
    
    if gpu_devices:
        print(f"✅ GPU devices found: {gpu_devices}")
        try:
            jax.config.update('jax_default_device', gpu_devices[0])
            print(f"✅ Default device set to: {gpu_devices[0]}")
        except Exception as e:
            print(f"⚠️ Could not set default device: {e}")
    else:
        print("⚠️ No GPU devices found, using CPU")
    
    return devices, gpu_devices

# Initialize JAX devices
_jax_devices, _gpu_devices = setup_jax_devices()

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from agents.causal_agent import CausalAgent
from agents.mdl_agent import MDLAgent
from envs.causal_grid_world import CausalGridWorld, CausalGridWorldState
from core.base_agent import BaseAgent


class CausalExperimentRunner:
    """Manages the complete causality experiment lifecycle"""
    
    def __init__(self, manifest_path: str):
        with open(manifest_path, 'r') as f:
            self.config = json.load(f)
        
        self.rng_key = jax.random.PRNGKey(42)
        self.setup_environment()
        self.setup_agents()
        self.setup_logging()
        
        # Track intervention history for analysis
        self.intervention_history = {}
        
    def setup_environment(self):
        """Initialize training and evaluation environments"""
        env_config = self.config["environment"]["config"]
        self.env = CausalGridWorld(**env_config)
        
        # Calculate observation dimension (includes causal variables)
        obs_dim = self.env.obs_dim
        
        # Update agent configs with environment info
        for agent_config in self.config["agents"]:
            agent_config["config"]["obs_dim"] = obs_dim
            agent_config["config"]["action_dim"] = self.env.action_dim
    
    def setup_agents(self):
        """Initialize all agents for comparison"""
        self.agents = {}
        
        for agent_config in self.config["agents"]:
            name = agent_config["name"]
            agent_type = agent_config["type"]
            config = agent_config["config"]
            
            if agent_type == "CausalAgent":
                agent = CausalAgent(config)
            elif agent_type == "MDLAgent":
                agent = MDLAgent(config)
            else:
                raise ValueError(f"Unknown agent type: {agent_type}")
            
            # Initialize agent networks
            key, self.rng_key = jax.random.split(self.rng_key)
            dummy_obs = jnp.zeros(config["obs_dim"])
            agent.setup(key, dummy_obs)
            
            self.agents[name] = agent
            self.intervention_history[name] = []
    
    def setup_logging(self):
        """Initialize experiment logging"""
        log_config = self.config["logging"]
        
        try:
            wandb.init(
                project=log_config["wandb_project"],
                name=f"{self.config['experiment']['name']}_{self.config['experiment']['version']}",
                tags=log_config["wandb_tags"],
                config=self.config
            )
            print("✅ WandB logging initialized successfully")
        except Exception as e:
            print(f"⚠️ WandB setup warning: {e}")
            print("   Continuing without WandB logging...")
            os.environ["WANDB_MODE"] = "offline"
            wandb.init(
                project=log_config["wandb_project"],
                name=f"{self.config['experiment']['name']}_{self.config['experiment']['version']}",
                tags=log_config["wandb_tags"],
                config=self.config,
                mode="offline"
            )
    
    def collect_episode(self, agent: BaseAgent, env: CausalGridWorld = None, 
                       max_steps: int = None) -> Dict[str, Any]:
        """Collect one episode of experience"""
        if env is None:
            env = self.env
        if max_steps is None:
            max_steps = self.config["environment"]["config"]["max_steps"]
        
        # Reset environment
        key, self.rng_key = jax.random.split(self.rng_key)
        state, obs = env.reset(key)
        
        episode_data = {
            "observations": [],
            "next_observations": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "infos": []
        }
        
        total_reward = 0
        steps = 0
        intervention_events = []
        
        while not state.done and steps < max_steps:
            # Agent selects action
            key, self.rng_key = jax.random.split(self.rng_key)
            action, agent_info = agent.act(obs, key)
            
            # Environment step
            next_state, next_obs, reward, done, env_info = env.step(state, int(action))
            
            # Track intervention events
            if jnp.any(state.active_interventions):
                intervention_events.append({
                    "step": steps,
                    "interventions": state.active_interventions.tolist(),
                    "values": state.intervention_values.tolist(),
                    "reward": reward
                })
            
            # Store experience
            episode_data["observations"].append(obs)
            episode_data["next_observations"].append(next_obs)
            episode_data["actions"].append(action)
            episode_data["rewards"].append(reward)
            episode_data["dones"].append(done)
            episode_data["infos"].append({**agent_info, **env_info})
            
            state = next_state
            obs = next_obs
            total_reward += reward
            steps += 1
        
        # Convert to arrays
        for key in ["observations", "next_observations", "actions", "rewards", "dones"]:
            episode_data[key] = jnp.array(episode_data[key])
        
        return {
            "episode_data": episode_data,
            "total_reward": total_reward,
            "steps": steps,
            "success": episode_data["infos"][-1].get("goal_reached", False) if episode_data["infos"] else False,
            "intervention_events": intervention_events,
            "final_causal_vars": {
                "wind_strength": float(state.wind_strength),
                "visibility": float(state.visibility),
                "goal_stability": float(state.goal_stability)
            }
        }
    
    def train_agent(self, agent: BaseAgent, episodes: int) -> List[Dict[str, float]]:
        """Train a single agent for specified number of episodes"""
        training_metrics = []
        
        for episode in tqdm(range(episodes), desc=f"Training {agent.name}"):
            # Collect episode
            episode_result = self.collect_episode(agent)
            
            # Prepare batch for training
            batch = {
                "observations": episode_result["episode_data"]["observations"],
                "next_observations": episode_result["episode_data"]["next_observations"],
                "actions": episode_result["episode_data"]["actions"],
                "rewards": episode_result["episode_data"]["rewards"],
                "dones": episode_result["episode_data"]["dones"]
            }
            
            # Update agent
            update_metrics = agent.update(batch)
            
            # Track intervention adaptation for causal agents
            if hasattr(agent, 'adapt_to_intervention') and episode_result["intervention_events"]:
                for event in episode_result["intervention_events"]:
                    agent.adapt_to_intervention(event)
            
            # Combine metrics
            metrics = {
                "episode": episode,
                "total_reward": float(episode_result["total_reward"]),
                "steps": episode_result["steps"],
                "success": episode_result["success"],
                "num_interventions": len(episode_result["intervention_events"]),
                **update_metrics
            }
            
            # Add causal-specific metrics for CausalAgent
            if isinstance(agent, CausalAgent):
                metrics.update({
                    "causal_accuracy": update_metrics.get("causal_accuracy", 0.0),
                    "intervention_adaptation": update_metrics.get("intervention_adaptation", 0.0)
                })
            
            training_metrics.append(metrics)
            
            # Log to wandb
            if episode % self.config["logging"]["log_frequency"] == 0:
                wandb.log({f"{agent.name}/{k}": v for k, v in metrics.items()})
                
                # Log causal variables if available
                if episode_result["final_causal_vars"]:
                    for var, value in episode_result["final_causal_vars"].items():
                        wandb.log({f"{agent.name}/causal_vars/{var}": value})
        
        return training_metrics
    
    def evaluate_intervention_adaptation(self, agent: BaseAgent, 
                                       intervention_type: str) -> Dict[str, float]:
        """Evaluate agent's adaptation to specific intervention"""
        # Create intervention environment
        ood_config = self.env.generate_causal_ood_config(
            self.rng_key, intervention_type
        )
        
        env_config = {**self.config["environment"]["config"], **ood_config}
        intervention_env = CausalGridWorld(**env_config)
        
        # Run episodes with intervention
        intervention_test = self.config["evaluation"]["intervention_tests"]
        test_config = next(t for t in intervention_test if t["type"] == intervention_type)
        episodes = test_config["episodes"]
        
        results = []
        for episode in tqdm(range(episodes), desc=f"Testing {intervention_type}"):
            episode_result = self.collect_episode(agent, intervention_env)
            results.append(episode_result)
        
        # Compute adaptation metrics
        success_rates = [r["success"] for r in results]
        rewards = [r["total_reward"] for r in results]
        steps = [r["steps"] for r in results]
        
        # Measure adaptation speed (improvement over time)
        window_size = 10
        early_performance = np.mean(success_rates[:window_size])
        late_performance = np.mean(success_rates[-window_size:])
        adaptation_speed = late_performance - early_performance
        
        return {
            "success_rate": np.mean(success_rates),
            "avg_reward": np.mean(rewards),
            "avg_steps": np.mean(steps),
            "adaptation_speed": adaptation_speed,
            "early_performance": early_performance,
            "late_performance": late_performance
        }
    
    def evaluate_causal_transfer(self, agent: BaseAgent) -> Dict[str, float]:
        """Evaluate causal transfer learning capabilities"""
        transfer_results = {}
        
        for test in self.config["evaluation"]["causal_transfer_tests"]:
            source_intervention = test["source_intervention"]
            target_intervention = test["target_intervention"]
            episodes = test["episodes"]
            
            # First train on source intervention
            source_metrics = self.evaluate_intervention_adaptation(agent, source_intervention)
            
            # Then test zero-shot transfer to target
            target_metrics = self.evaluate_intervention_adaptation(agent, target_intervention)
            
            # Calculate transfer efficiency
            baseline_success = 0.1  # Assumed random performance
            source_improvement = source_metrics["success_rate"] - baseline_success
            target_improvement = target_metrics["success_rate"] - baseline_success
            
            transfer_efficiency = target_improvement / max(source_improvement, 0.01)
            
            transfer_results[f"{source_intervention}_to_{target_intervention}"] = {
                "source_performance": source_metrics["success_rate"],
                "target_performance": target_metrics["success_rate"],
                "transfer_efficiency": transfer_efficiency,
                "adaptation_speed": target_metrics["adaptation_speed"]
            }
        
        return transfer_results
    
    def run_complete_evaluation(self):
        """Run complete evaluation suite for all agents"""
        print("\n🧪 Starting complete evaluation suite...")
        
        evaluation_results = {}
        
        for agent_name, agent in self.agents.items():
            print(f"\n📊 Evaluating {agent_name}...")
            agent_results = {}
            
            # Standard OOD tests
            for ood_test in self.config["evaluation"]["ood_tests"]:
                ood_type = ood_test["type"]
                print(f"   Testing OOD: {ood_type}")
                ood_metrics = self.evaluate_ood(agent, ood_type)
                agent_results[f"ood_{ood_type}"] = ood_metrics
            
            # Intervention adaptation tests
            for intervention_test in self.config["evaluation"]["intervention_tests"]:
                intervention_type = intervention_test["type"]
                print(f"   Testing intervention: {intervention_type}")
                intervention_metrics = self.evaluate_intervention_adaptation(agent, intervention_type)
                agent_results[f"intervention_{intervention_type}"] = intervention_metrics
            
            # Causal transfer tests (only for CausalAgent)
            if isinstance(agent, CausalAgent):
                print(f"   Testing causal transfer...")
                transfer_metrics = self.evaluate_causal_transfer(agent)
                agent_results["causal_transfer"] = transfer_metrics
            
            evaluation_results[agent_name] = agent_results
            
            # Log to wandb
            for test_name, metrics in agent_results.items():
                if isinstance(metrics, dict):
                    for metric_name, value in metrics.items():
                        wandb.log({f"eval_{agent_name}/{test_name}/{metric_name}": value})
                else:
                    wandb.log({f"eval_{agent_name}/{test_name}": metrics})
        
        return evaluation_results
    
    def evaluate_ood(self, agent: BaseAgent, ood_type: str) -> Dict[str, float]:
        """Evaluate agent on out-of-distribution scenarios"""
        key, self.rng_key = jax.random.split(self.rng_key)
        
        if ood_type in ["strong_wind", "low_visibility", "unstable_goal", "multiple_interventions"]:
            ood_config = self.env.generate_causal_ood_config(key, ood_type)
        else:
            ood_config = self.env.generate_ood_config(key, ood_type)
        
        env_config = {**self.config["environment"]["config"], **ood_config}
        ood_env = CausalGridWorld(**env_config)
        
        # Run evaluation episodes
        episodes = 100  # Default
        for test in self.config["evaluation"]["ood_tests"]:
            if test["type"] == ood_type:
                episodes = test["episodes"]
                break
        
        results = []
        for episode in range(episodes):
            episode_result = self.collect_episode(agent, ood_env)
            results.append(episode_result)
        
        # Compute metrics
        success_rates = [r["success"] for r in results]
        rewards = [r["total_reward"] for r in results]
        steps = [r["steps"] for r in results]
        
        return {
            "success_rate": np.mean(success_rates),
            "avg_reward": np.mean(rewards),
            "avg_steps": np.mean(steps),
            "std_reward": np.std(rewards)
        }
    
    def print_summary(self, evaluation_results: Dict[str, Any]):
        """Print experiment summary"""
        print("\n" + "="*50)
        print("🎯 CAUSALITY EXPERIMENT SUMMARY")
        print("="*50)
        
        print(f"\nExperiment: {self.config['experiment']['name']}")
        print(f"Hypothesis: {self.config['experiment']['hypothesis']}")
        
        for agent_name, results in evaluation_results.items():
            print(f"\n📊 {agent_name} Results:")
            
            # Intervention adaptation results
            intervention_results = {k: v for k, v in results.items() if k.startswith("intervention_")}
            if intervention_results:
                print("   Intervention Adaptation:")
                for test_name, metrics in intervention_results.items():
                    intervention_type = test_name.replace("intervention_", "")
                    print(f"     {intervention_type}: {metrics['success_rate']:.3f} success rate, "
                          f"{metrics['adaptation_speed']:.3f} adaptation speed")
            
            # Causal transfer results
            if "causal_transfer" in results:
                print("   Causal Transfer:")
                for transfer_name, metrics in results["causal_transfer"].items():
                    print(f"     {transfer_name}: {metrics['transfer_efficiency']:.3f} efficiency")
        
        # Print conclusion based on hypothesis
        print("\n" + "="*50)
        print("📋 CONCLUSION")
        print("="*50)
        
        # Compare causal agents vs baseline
        causal_agents = {k: v for k, v in evaluation_results.items() if "Causal" in k}
        baseline_agents = {k: v for k, v in evaluation_results.items() if "MDL" in k}
        
        if causal_agents and baseline_agents:
            # Calculate average intervention adaptation for causal vs baseline
            causal_avg_adaptation = np.mean([
                np.mean([metrics['adaptation_speed'] for test_name, metrics 
                        in agent_results.items() if test_name.startswith("intervention_")])
                for agent_results in causal_agents.values()
            ])
            
            baseline_avg_adaptation = np.mean([
                np.mean([metrics['adaptation_speed'] for test_name, metrics 
                        in agent_results.items() if test_name.startswith("intervention_")])
                for agent_results in baseline_agents.values()
            ])
            
            if causal_avg_adaptation > baseline_avg_adaptation * 1.1:
                print("✅ HYPOTHESIS SUPPORTED: Causal reasoning shows better intervention adaptation")
            else:
                print("❌ HYPOTHESIS NOT SUPPORTED: No clear advantage for causal reasoning")
            
            print(f"   Causal agents adaptation speed: {causal_avg_adaptation:.3f}")
            print(f"   Baseline agents adaptation speed: {baseline_avg_adaptation:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Run Causality vs Correlation experiment")
    parser.add_argument("--manifest", default="manifest.json", help="Path to experiment manifest")
    parser.add_argument("--episodes", type=int, help="Override number of training episodes")
    parser.add_argument("--eval-only", action="store_true", help="Run evaluation only")
    
    args = parser.parse_args()
    
    # Initialize experiment
    manifest_path = Path(__file__).parent / args.manifest
    runner = CausalExperimentRunner(str(manifest_path))
    
    if not args.eval_only:
        print("🚀 Starting training phase...")
        training_config = runner.config["training"]
        total_episodes = args.episodes or training_config["total_episodes"]
        
        # Train all agents
        for agent_name, agent in runner.agents.items():
            print(f"\n🎯 Training {agent_name} for {total_episodes} episodes...")
            runner.train_agent(agent, total_episodes)
    
    # Run complete evaluation
    evaluation_results = runner.run_complete_evaluation()
    
    # Print summary
    runner.print_summary(evaluation_results)
    
    # Save results
    results_path = Path(__file__).parent / "results.json"
    with open(results_path, 'w') as f:
        json.dump(evaluation_results, f, indent=2, default=str)
    print(f"\n💾 Results saved to {results_path}")
    
    wandb.finish()
    print("\n✅ Experiment completed successfully!")


if __name__ == "__main__":
    main()