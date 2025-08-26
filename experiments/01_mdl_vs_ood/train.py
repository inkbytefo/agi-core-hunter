#!/usr/bin/env python3
"""
Training script for MDL vs OOD experiment

This script implements the core training loop for testing whether
MDL regularization improves out-of-distribution generalization.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List
import argparse

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
import wandb

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from agents.mdl_agent import MDLAgent
from envs.grid_world import GridWorld, reset_batch, step_batch
from core.base_agent import BaseAgent


class ExperimentRunner:
    """Manages the complete experiment lifecycle"""
    
    def __init__(self, manifest_path: str):
        with open(manifest_path, 'r') as f:
            self.config = json.load(f)
        
        self.rng_key = jax.random.PRNGKey(42)
        self.setup_environment()
        self.setup_agents()
        self.setup_logging()
    
    def setup_environment(self):
        """Initialize training and evaluation environments"""
        env_config = self.config["environment"]["config"]
        self.env = GridWorld(**env_config)
        
        # Calculate observation dimension
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
            
            if agent_type == "MDLAgent":
                agent = MDLAgent(config)
            else:
                raise ValueError(f"Unknown agent type: {agent_type}")
            
            # Initialize agent networks
            key, self.rng_key = jax.random.split(self.rng_key)
            dummy_obs = jnp.zeros(config["obs_dim"])
            agent.setup(key, dummy_obs)
            
            self.agents[name] = agent
    
    def setup_logging(self):
        """Initialize experiment logging"""
        log_config = self.config["logging"]
        
        wandb.init(
            project=log_config["wandb_project"],
            name=f"{self.config['experiment']['name']}_{self.config['experiment']['version']}",
            tags=log_config["wandb_tags"],
            config=self.config
        )
    
    def collect_episode(self, agent: BaseAgent, max_steps: int = None) -> Dict[str, Any]:
        """Collect one episode of experience"""
        if max_steps is None:
            max_steps = self.config["environment"]["config"]["max_steps"]
        
        # Reset environment
        key, self.rng_key = jax.random.split(self.rng_key)
        state, obs = self.env.reset(key)
        
        episode_data = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "infos": []
        }
        
        total_reward = 0
        steps = 0
        
        while not state.done and steps < max_steps:
            # Agent selects action
            key, self.rng_key = jax.random.split(self.rng_key)
            action, agent_info = agent.act(obs, key)
            
            # Environment step
            state, next_obs, reward, done, env_info = self.env.step(state, int(action))
            
            # Store experience
            episode_data["observations"].append(obs)
            episode_data["actions"].append(action)
            episode_data["rewards"].append(reward)
            episode_data["dones"].append(done)
            episode_data["infos"].append({**agent_info, **env_info})
            
            obs = next_obs
            total_reward += reward
            steps += 1
        
        # Convert to arrays
        for key in ["observations", "actions", "rewards", "dones"]:
            episode_data[key] = jnp.array(episode_data[key])
        
        return {
            "episode_data": episode_data,
            "total_reward": total_reward,
            "steps": steps,
            "success": episode_data["infos"][-1].get("goal_reached", False) if episode_data["infos"] else False
        }
    
    def train_agent(self, agent: BaseAgent, episodes: int) -> List[Dict[str, float]]:
        """Train a single agent for specified number of episodes"""
        training_metrics = []
        
        for episode in tqdm(range(episodes), desc=f"Training {agent.name}"):
            # Collect episode
            episode_result = self.collect_episode(agent)
            
            # Prepare batch for training (simplified - single episode)
            batch = {
                "observations": episode_result["episode_data"]["observations"],
                "actions": episode_result["episode_data"]["actions"],
                "rewards": episode_result["episode_data"]["rewards"],
                "returns": self._compute_returns(episode_result["episode_data"]["rewards"])
            }
            
            # Update agent
            update_metrics = agent.update(batch)
            
            # Combine metrics
            metrics = {
                "episode": episode,
                "total_reward": float(episode_result["total_reward"]),
                "steps": episode_result["steps"],
                "success": episode_result["success"],
                **update_metrics
            }
            
            training_metrics.append(metrics)
            
            # Log to wandb
            if episode % self.config["logging"]["log_frequency"] == 0:
                wandb.log({f"{agent.name}/{k}": v for k, v in metrics.items()})
        
        return training_metrics
    
    def _compute_returns(self, rewards: jnp.ndarray, gamma: float = 0.99) -> jnp.ndarray:
        """Compute discounted returns"""
        returns = jnp.zeros_like(rewards)
        running_return = 0
        
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + gamma * running_return
            returns = returns.at[t].set(running_return)
        
        return returns
    
    def evaluate_ood(self, agent: BaseAgent) -> Dict[str, float]:
        """Evaluate agent on out-of-distribution scenarios"""
        ood_results = {}
        
        for ood_test in self.config["evaluation"]["ood_tests"]:
            test_name = ood_test["name"]
            ood_type = ood_test["type"]
            episodes = ood_test["episodes"]
            
            # Create OOD environment
            key, self.rng_key = jax.random.split(self.rng_key)
            ood_config = self.env.generate_ood_config(key, ood_type)
            
            # Temporarily modify environment
            original_config = {}
            for param, value in ood_config.items():
                if hasattr(self.env, param):
                    original_config[param] = getattr(self.env, param)
                    setattr(self.env, param, value)
            
            # Run evaluation episodes
            successes = 0
            total_rewards = []
            total_steps = []
            
            for _ in range(episodes):
                result = self.collect_episode(agent)
                if result["success"]:
                    successes += 1
                total_rewards.append(result["total_reward"])
                total_steps.append(result["steps"])
            
            # Restore original environment
            for param, value in original_config.items():
                setattr(self.env, param, value)
            
            # Calculate metrics
            ood_results[f"{test_name}_success_rate"] = successes / episodes
            ood_results[f"{test_name}_avg_reward"] = np.mean(total_rewards)
            ood_results[f"{test_name}_avg_steps"] = np.mean(total_steps)
        
        return ood_results
    
    def run_experiment(self):
        """Run the complete experiment"""
        print(f"Starting experiment: {self.config['experiment']['name']}")
        print(f"Training {len(self.agents)} agents for {self.config['training']['total_episodes']} episodes each")
        
        all_results = {}
        
        for agent_name, agent in self.agents.items():
            print(f"\n--- Training {agent_name} ---")
            
            # Train agent
            training_metrics = self.train_agent(
                agent, 
                self.config["training"]["total_episodes"]
            )
            
            # Evaluate on standard environment
            print(f"Evaluating {agent_name} on standard environment...")
            standard_results = []
            for _ in range(100):  # 100 evaluation episodes
                result = self.collect_episode(agent)
                standard_results.append(result)
            
            standard_success_rate = np.mean([r["success"] for r in standard_results])
            standard_avg_reward = np.mean([r["total_reward"] for r in standard_results])
            
            # Evaluate on OOD scenarios
            print(f"Evaluating {agent_name} on OOD scenarios...")
            ood_results = self.evaluate_ood(agent)
            
            # Calculate OOD performance ratios
            ood_performance_ratios = {}
            for ood_test in self.config["evaluation"]["ood_tests"]:
                test_name = ood_test["name"]
                ood_success = ood_results[f"{test_name}_success_rate"]
                ratio = ood_success / standard_success_rate if standard_success_rate > 0 else 0
                ood_performance_ratios[f"{test_name}_performance_ratio"] = ratio
            
            # Store results
            all_results[agent_name] = {
                "training_metrics": training_metrics,
                "standard_success_rate": standard_success_rate,
                "standard_avg_reward": standard_avg_reward,
                "ood_results": ood_results,
                "ood_performance_ratios": ood_performance_ratios,
                "agent_config": agent.config
            }
            
            # Log final results
            final_metrics = {
                f"{agent_name}/final_standard_success_rate": standard_success_rate,
                f"{agent_name}/final_standard_avg_reward": standard_avg_reward,
                **{f"{agent_name}/{k}": v for k, v in ood_results.items()},
                **{f"{agent_name}/{k}": v for k, v in ood_performance_ratios.items()}
            }
            wandb.log(final_metrics)
        
        # Save results
        results_path = Path(__file__).parent / "results.json"
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=float)
        
        print(f"\nExperiment completed! Results saved to {results_path}")
        
        # Print summary
        self.print_summary(all_results)
        
        wandb.finish()
        return all_results
    
    def print_summary(self, results: Dict[str, Any]):
        """Print experiment summary"""
        print("\n" + "="*60)
        print("EXPERIMENT SUMMARY")
        print("="*60)
        
        print(f"Hypothesis: {self.config['experiment']['hypothesis']}")
        print()
        
        # Sort agents by beta value for comparison
        agent_results = []
        for agent_name, result in results.items():
            beta = result["agent_config"]["beta"]
            standard_success = result["standard_success_rate"]
            
            # Average OOD performance ratio
            ood_ratios = [v for k, v in result["ood_performance_ratios"].items() if k.endswith("_performance_ratio")]
            avg_ood_ratio = np.mean(ood_ratios) if ood_ratios else 0
            
            agent_results.append({
                "name": agent_name,
                "beta": beta,
                "standard_success": standard_success,
                "avg_ood_ratio": avg_ood_ratio
            })
        
        agent_results.sort(key=lambda x: x["beta"])
        
        print("Agent Performance Summary:")
        print("-" * 60)
        print(f"{'Agent':<15} {'Beta':<8} {'Standard':<12} {'Avg OOD Ratio':<15}")
        print("-" * 60)
        
        for result in agent_results:
            print(f"{result['name']:<15} {result['beta']:<8.1f} {result['standard_success']:<12.3f} {result['avg_ood_ratio']:<15.3f}")
        
        print()
        
        # Check hypothesis
        if len(agent_results) >= 2:
            # Simple check: does higher beta lead to better OOD performance?
            correlation = np.corrcoef(
                [r["beta"] for r in agent_results],
                [r["avg_ood_ratio"] for r in agent_results]
            )[0, 1]
            
            print(f"Beta-OOD Correlation: {correlation:.3f}")
            
            if correlation > 0.3:
                print("✅ HYPOTHESIS SUPPORTED: Higher β appears to improve OOD generalization")
            elif correlation < -0.3:
                print("❌ HYPOTHESIS REJECTED: Higher β appears to hurt OOD generalization")
            else:
                print("❓ HYPOTHESIS INCONCLUSIVE: No clear relationship found")


def main():
    parser = argparse.ArgumentParser(description="Run MDL vs OOD experiment")
    parser.add_argument(
        "--manifest", 
        default="manifest.json",
        help="Path to experiment manifest file"
    )
    
    args = parser.parse_args()
    
    # Run experiment
    runner = ExperimentRunner(args.manifest)
    results = runner.run_experiment()
    
    return results


if __name__ == "__main__":
    main()