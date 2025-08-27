#!/usr/bin/env python3
"""
Training script for FEP vs Exploration experiment

This script implements training and evaluation for testing whether
Free Energy Principle agents show better active inference and exploration
efficiency compared to other approaches.
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

# Configure JAX for optimal performance
try:
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'
    jax.config.update('jax_enable_x64', False)  # Use float32 for better GPU performance
except Exception as e:
    print(f"⚠️ JAX config warning: {e}")

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from agents.fep_agent import FEPAgent
from agents.mdl_agent import MDLAgent
from envs.active_inference_world import ActiveInferenceWorld, ActiveInferenceState
from core.base_agent import BaseAgent


class ReplayBuffer:
    """Simple replay buffer for experience storage"""
    
    def __init__(self, capacity: int, obs_dim: int, action_dim: int):
        self.capacity = capacity
        self.size = 0
        self.index = 0
        
        self.observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.bool_)
    
    def add(self, obs, action, reward, next_obs, done):
        """Add experience to buffer"""
        self.observations[self.index] = obs
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.next_observations[self.index] = next_obs
        self.dones[self.index] = done
        
        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Dict[str, jnp.ndarray]:
        """Sample batch of experiences"""
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        return {
            "observations": jnp.array(self.observations[indices]),
            "actions": jnp.array(self.actions[indices]),
            "rewards": jnp.array(self.rewards[indices]),
            "next_observations": jnp.array(self.next_observations[indices]),
            "dones": jnp.array(self.dones[indices])
        }
    
    def can_sample(self, batch_size: int) -> bool:
        """Check if buffer has enough experiences for sampling"""
        return self.size >= batch_size


class FEPExperimentRunner:
    """Manages the Free Energy Principle experiment lifecycle"""
    
    def __init__(self, manifest_path: str):
        with open(manifest_path, 'r') as f:
            self.config = json.load(f)
        
        self.rng_key = jax.random.PRNGKey(42)
        self.setup_environment()
        self.setup_agents()
        self.setup_replay_buffers()
        self.setup_logging()
        
        # Results tracking
        self.results = {agent_name: [] for agent_name in self.agents.keys()}
        self.fep_metrics = {agent_name: [] for agent_name in self.agents.keys()}
    
    def setup_environment(self):
        """Initialize training and evaluation environments"""
        env_config = self.config["environment"]["config"]
        self.env = ActiveInferenceWorld(**env_config)
        
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
            
            if agent_type == "FEPAgent":
                agent = FEPAgent(config)
            elif agent_type == "MDLAgent":
                agent = MDLAgent(config)
            else:
                raise ValueError(f"Unknown agent type: {agent_type}")
            
            # Initialize agent networks
            key, self.rng_key = jax.random.split(self.rng_key)
            dummy_obs = jnp.zeros(config["obs_dim"])
            agent.setup(key, dummy_obs)
            
            self.agents[name] = agent
    
    def setup_replay_buffers(self):
        """Initialize replay buffers for each agent"""
        buffer_size = self.config["training"]["buffer_size"]
        obs_dim = self.config["agents"][0]["config"]["obs_dim"]
        action_dim = self.config["agents"][0]["config"]["action_dim"]
        
        self.replay_buffers = {}
        for agent_name in self.agents.keys():
            self.replay_buffers[agent_name] = ReplayBuffer(buffer_size, obs_dim, action_dim)
    
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
            self.use_wandb = True
        except Exception as e:
            print(f"⚠️ WandB setup warning: {e}")
            self.use_wandb = False
    
    def collect_episode(self, agent_name: str, agent: BaseAgent) -> Dict[str, Any]:
        """Collect one episode of experience for an agent"""
        key, self.rng_key = jax.random.split(self.rng_key)
        
        # Reset environment and agent
        env_state, observation = self.env.reset(key)
        agent.reset()
        
        episode_data = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "infos": [],
            "fep_metrics": []
        }
        
        episode_reward = 0.0
        episode_steps = 0
        
        while not env_state.done:
            # Agent acts
            action_key, key = jax.random.split(key)
            action, info = agent.act(observation, action_key)
            
            # Environment step
            step_key, key = jax.random.split(key)
            next_env_state, next_observation, reward, done, env_info = self.env.step(
                env_state, int(action), step_key
            )
            
            # Store experience in replay buffer
            self.replay_buffers[agent_name].add(
                observation, int(action), reward, next_observation, done
            )
            
            # Store episode data
            episode_data["observations"].append(observation)
            episode_data["actions"].append(action)
            episode_data["rewards"].append(reward)
            episode_data["infos"].append(env_info)
            
            # Store FEP-specific metrics if available
            if hasattr(agent, 'get_metrics'):
                fep_metrics = agent.get_metrics()
                episode_data["fep_metrics"].append(fep_metrics)
            
            episode_reward += reward
            episode_steps += 1
            
            # Update for next step
            observation = next_observation
            env_state = next_env_state
        
        return {
            "episode_reward": episode_reward,
            "episode_steps": episode_steps,
            "episode_data": episode_data,
            "final_info": env_info
        }
    
    def train_agent(self, agent_name: str, agent: BaseAgent) -> Dict[str, float]:
        """Train agent using replay buffer"""
        if not self.replay_buffers[agent_name].can_sample(self.config["training"]["batch_size"]):
            return {}
        
        # Sample batch and train
        batch = self.replay_buffers[agent_name].sample(self.config["training"]["batch_size"])
        metrics = agent.update(batch)
        
        return metrics
    
    def evaluate_agent(self, agent_name: str, agent: BaseAgent, test_config: Dict) -> Dict[str, float]:
        """Evaluate agent on specific FEP test"""
        test_type = test_config["type"]
        n_episodes = test_config["episodes"]
        
        # Generate test environment configuration
        key, self.rng_key = jax.random.split(self.rng_key)
        test_env_config = self.env.generate_fep_config(key, test_type)
        
        # Create test environment
        test_env = ActiveInferenceWorld(**{**self.env.__dict__, **test_env_config})
        
        results = {
            "success_rate": 0.0,
            "average_reward": 0.0,
            "average_steps": 0.0,
            "exploration_efficiency": 0.0,
            "surprise_minimization": 0.0,
            "adaptation_speed": 0.0,
            "variational_free_energy": 0.0,
            "epistemic_value": 0.0,
            "pragmatic_value": 0.0
        }
        
        successful_episodes = 0
        total_rewards = []
        total_steps = []
        exploration_scores = []
        surprise_scores = []
        adaptation_scores = []
        fep_scores = []
        
        for episode in range(n_episodes):
            key, self.rng_key = jax.random.split(self.rng_key)
            
            # Reset environment and agent
            env_state, observation = test_env.reset(key)
            agent.reset()
            
            episode_reward = 0.0
            episode_steps = 0
            episode_exploration = 0.0
            episode_surprise = 0.0
            episode_fep_metrics = []
            
            while not env_state.done and episode_steps < test_env.max_steps:
                # Agent acts
                action_key, key = jax.random.split(key)
                action, info = agent.act(observation, action_key)
                
                # Environment step
                step_key, key = jax.random.split(key)
                next_env_state, next_observation, reward, done, env_info = test_env.step(
                    env_state, int(action), step_key
                )
                
                episode_reward += reward
                episode_steps += 1
                
                # Track FEP-specific metrics
                if hasattr(agent, 'get_metrics'):
                    fep_metrics = agent.get_metrics()
                    episode_fep_metrics.append(fep_metrics)
                
                # Track exploration
                episode_exploration += env_info.get("exploration_bonus", 0.0)
                
                # Track surprise
                episode_surprise += env_info.get("surprise_level", 0.0)
                
                # Update for next step
                observation = next_observation
                env_state = next_env_state
            
            # Episode results
            if env_info.get("goal_reached", False):
                successful_episodes += 1
            
            total_rewards.append(episode_reward)
            total_steps.append(episode_steps)
            exploration_scores.append(episode_exploration / max(episode_steps, 1))
            surprise_scores.append(episode_surprise / max(episode_steps, 1))
            
            # Calculate adaptation score (performance improvement over episode)
            if len(total_rewards) >= 10:
                recent_performance = np.mean(total_rewards[-10:])
                early_performance = np.mean(total_rewards[:10])
                adaptation_score = recent_performance - early_performance
            else:
                adaptation_score = 0.0
            adaptation_scores.append(adaptation_score)
            
            # Average FEP metrics over episode
            if episode_fep_metrics:
                avg_fep = {}
                for key in episode_fep_metrics[0].keys():
                    avg_fep[key] = np.mean([m[key] for m in episode_fep_metrics])
                fep_scores.append(avg_fep)
        
        # Calculate final results
        results["success_rate"] = successful_episodes / n_episodes
        results["average_reward"] = np.mean(total_rewards)
        results["average_steps"] = np.mean(total_steps)
        results["exploration_efficiency"] = np.mean(exploration_scores)
        results["surprise_minimization"] = -np.mean(surprise_scores)  # Lower surprise is better
        results["adaptation_speed"] = np.mean(adaptation_scores)
        
        # Average FEP metrics
        if fep_scores:
            for key in fep_scores[0].keys():
                results[key] = np.mean([score[key] for score in fep_scores])
        
        return results
    
    def run_training(self):
        """Run the complete training process"""
        total_episodes = self.config["training"]["total_episodes"]
        eval_frequency = self.config["training"]["eval_frequency"]
        log_frequency = self.config["logging"]["log_frequency"]
        
        print(f"🚀 Starting FEP vs Exploration experiment")
        print(f"   Agents: {list(self.agents.keys())}")
        print(f"   Episodes: {total_episodes}")
        print(f"   Environment: {self.config['environment']['type']}")
        
        for episode in tqdm(range(total_episodes), desc="Training"):
            # Collect experiences for each agent
            for agent_name, agent in self.agents.items():
                episode_result = self.collect_episode(agent_name, agent)
                
                # Store results
                self.results[agent_name].append({
                    "episode": episode,
                    "reward": episode_result["episode_reward"],
                    "steps": episode_result["episode_steps"],
                    "info": episode_result["final_info"]
                })
                
                # Store FEP metrics
                if episode_result["episode_data"]["fep_metrics"]:
                    avg_metrics = {}
                    for key in episode_result["episode_data"]["fep_metrics"][0].keys():
                        avg_metrics[key] = np.mean([
                            m[key] for m in episode_result["episode_data"]["fep_metrics"]
                        ])
                    self.fep_metrics[agent_name].append(avg_metrics)
                
                # Train agent
                training_metrics = self.train_agent(agent_name, agent)
                
                # Log metrics
                if self.use_wandb and episode % log_frequency == 0:
                    log_data = {
                        f"{agent_name}/episode_reward": episode_result["episode_reward"],
                        f"{agent_name}/episode_steps": episode_result["episode_steps"],
                        f"{agent_name}/exploration_ratio": episode_result["final_info"].get("exploration_ratio", 0.0),
                        f"{agent_name}/surprise_level": episode_result["final_info"].get("surprise_level", 0.0)
                    }
                    
                    # Add FEP metrics
                    if hasattr(agent, 'get_metrics'):
                        agent_metrics = agent.get_metrics()
                        for key, value in agent_metrics.items():
                            log_data[f"{agent_name}/{key}"] = value
                    
                    # Add training metrics
                    for key, value in training_metrics.items():
                        log_data[f"{agent_name}/train_{key}"] = value
                    
                    wandb.log(log_data, step=episode)
            
            # Evaluation
            if episode % eval_frequency == 0 and episode > 0:
                print(f"\n📊 Evaluation at episode {episode}")
                self.run_evaluation(episode)
        
        # Final evaluation
        print(f"\n🎯 Final evaluation")
        self.run_evaluation(total_episodes)
        
        # Print summary
        self.print_summary()
    
    def run_evaluation(self, episode: int):
        """Run evaluation on all FEP test scenarios"""
        eval_results = {}
        
        for test_config in self.config["evaluation"]["fep_tests"]:
            test_name = test_config["name"]
            print(f"   Running {test_name} test...")
            
            test_results = {}
            for agent_name, agent in self.agents.items():
                agent_results = self.evaluate_agent(agent_name, agent, test_config)
                test_results[agent_name] = agent_results
            
            eval_results[test_name] = test_results
            
            # Log evaluation results
            if self.use_wandb:
                for agent_name, results in test_results.items():
                    log_data = {}
                    for metric, value in results.items():
                        log_data[f"eval_{test_name}/{agent_name}_{metric}"] = value
                    wandb.log(log_data, step=episode)
            
            # Print results
            print(f"     {test_name} Results:")
            for agent_name, results in test_results.items():
                print(f"       {agent_name}: Success Rate = {results['success_rate']:.3f}, "
                      f"Exploration = {results['exploration_efficiency']:.3f}, "
                      f"VFE = {results.get('variational_free_energy', 0):.3f}")
        
        return eval_results
    
    def print_summary(self):
        """Print experiment summary and hypothesis test results"""
        print("\n" + "="*80)
        print("🧠 FREE ENERGY PRINCIPLE vs EXPLORATION - EXPERIMENT SUMMARY")
        print("="*80)
        
        # Calculate overall performance metrics
        summary_table = []
        
        for agent_name in self.agents.keys():
            if not self.results[agent_name]:
                continue
                
            # Recent performance (last 20% of episodes)
            recent_episodes = int(len(self.results[agent_name]) * 0.2)
            recent_rewards = [r["reward"] for r in self.results[agent_name][-recent_episodes:]]
            
            avg_reward = np.mean(recent_rewards)
            avg_exploration = np.mean([
                r["info"].get("exploration_ratio", 0) for r in self.results[agent_name][-recent_episodes:]
            ])
            
            # FEP-specific metrics
            if self.fep_metrics[agent_name]:
                recent_fep = self.fep_metrics[agent_name][-recent_episodes:]
                avg_vfe = np.mean([m.get("variational_free_energy", 0) for m in recent_fep])
                avg_epistemic = np.mean([m.get("epistemic_value", 0) for m in recent_fep])
            else:
                avg_vfe = 0.0
                avg_epistemic = 0.0
            
            summary_table.append({
                "Agent": agent_name,
                "Avg Reward": avg_reward,
                "Exploration %": avg_exploration * 100,
                "VFE": avg_vfe,
                "Epistemic Value": avg_epistemic
            })
        
        # Print summary table
        print("📊 PERFORMANCE SUMMARY:")
        print("-" * 80)
        print(f"{'Agent':<15} {'Avg Reward':<12} {'Exploration %':<14} {'VFE':<10} {'Epistemic':<12}")
        print("-" * 80)
        
        for row in summary_table:
            print(f"{row['Agent']:<15} {row['Avg Reward']:<12.2f} {row['Exploration %']:<14.1f} "
                  f"{row['VFE']:<10.3f} {row['Epistemic']:<12.3f}")
        
        # Test hypothesis
        print("\n🔬 HYPOTHESIS TESTING:")
        print("-" * 80)
        
        hypothesis = self.config["experiment"]["hypothesis"]
        print(f"Hypothesis: {hypothesis}")
        
        # Find FEP agents and compare by epistemic weight
        fep_agents = [name for name in self.agents.keys() if "FEP" in name]
        
        if len(fep_agents) >= 2:
            # Sort by epistemic weight
            fep_performance = []
            for agent_name in fep_agents:
                agent_config = next(c for c in self.config["agents"] if c["name"] == agent_name)
                epistemic_weight = agent_config["config"].get("epistemic_weight", 0)
                
                recent_episodes = int(len(self.results[agent_name]) * 0.2)
                recent_rewards = [r["reward"] for r in self.results[agent_name][-recent_episodes:]]
                avg_performance = np.mean(recent_rewards)
                
                recent_exploration = np.mean([
                    r["info"].get("exploration_ratio", 0) for r in self.results[agent_name][-recent_episodes:]
                ])
                
                fep_performance.append({
                    "agent": agent_name,
                    "epistemic_weight": epistemic_weight,
                    "performance": avg_performance,
                    "exploration": recent_exploration
                })
            
            # Calculate correlation
            weights = [p["epistemic_weight"] for p in fep_performance]
            performances = [p["performance"] for p in fep_performance]
            explorations = [p["exploration"] for p in fep_performance]
            
            perf_correlation = np.corrcoef(weights, performances)[0, 1] if len(weights) > 1 else 0
            expl_correlation = np.corrcoef(weights, explorations)[0, 1] if len(weights) > 1 else 0
            
            print(f"\nCorrelation between epistemic_weight and performance: {perf_correlation:.3f}")
            print(f"Correlation between epistemic_weight and exploration: {expl_correlation:.3f}")
            
            # Hypothesis verdict
            if perf_correlation > 0.3 and expl_correlation > 0.3:
                verdict = "✅ SUPPORTED"
                explanation = "Higher epistemic weight correlates with better performance and exploration"
            elif perf_correlation < -0.3:
                verdict = "❌ REJECTED"
                explanation = "Higher epistemic weight actually hurts performance"
            else:
                verdict = "🤔 INCONCLUSIVE"
                explanation = "No clear relationship found between epistemic weight and performance"
            
            print(f"\n🎯 HYPOTHESIS VERDICT: {verdict}")
            print(f"   {explanation}")
        
        else:
            print("   Insufficient FEP agents for correlation analysis")
        
        print("\n" + "="*80)


def main():
    """Main training script"""
    parser = argparse.ArgumentParser(description="Train FEP vs Exploration experiment")
    parser.add_argument(
        "--manifest", 
        default="manifest.json",
        help="Path to experiment manifest file"
    )
    
    args = parser.parse_args()
    
    # Run experiment
    runner = FEPExperimentRunner(args.manifest)
    runner.run_training()


if __name__ == "__main__":
    main()