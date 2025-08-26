"""
Utility functions for calculating AGI principle-specific metrics
"""

import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Any


def calculate_ood_performance_ratio(
    standard_performance: float, 
    ood_performance: float
) -> float:
    """Calculate OOD performance as ratio of standard performance"""
    if standard_performance == 0:
        return 0.0
    return ood_performance / standard_performance


def calculate_sample_efficiency(
    performance_history: List[float], 
    target_performance: float = 0.8
) -> int:
    """Calculate number of episodes needed to reach target performance"""
    for i, perf in enumerate(performance_history):
        if perf >= target_performance:
            return i + 1
    return len(performance_history)  # Never reached target


def calculate_compression_ratio(obs_dim: int, latent_dim: int) -> float:
    """Calculate compression ratio"""
    return obs_dim / latent_dim


def calculate_transfer_speed(
    baseline_episodes: int,
    transfer_episodes: int
) -> float:
    """Calculate transfer learning speedup"""
    if transfer_episodes == 0:
        return float('inf')
    return baseline_episodes / transfer_episodes


def calculate_robustness_score(
    performances: Dict[str, float],
    baseline_key: str = "standard"
) -> float:
    """Calculate overall robustness as average performance ratio"""
    baseline = performances.get(baseline_key, 1.0)
    if baseline == 0:
        return 0.0
    
    ratios = []
    for key, perf in performances.items():
        if key != baseline_key:
            ratios.append(perf / baseline)
    
    return np.mean(ratios) if ratios else 0.0


def calculate_consistency_score(performance_history: List[float]) -> float:
    """Calculate performance consistency (1 - coefficient of variation)"""
    if len(performance_history) < 2:
        return 1.0
    
    mean_perf = np.mean(performance_history)
    if mean_perf == 0:
        return 0.0
    
    std_perf = np.std(performance_history)
    cv = std_perf / mean_perf
    
    return max(0.0, 1.0 - cv)


def calculate_learning_curve_auc(
    episodes: List[int], 
    performances: List[float]
) -> float:
    """Calculate area under learning curve (normalized)"""
    if len(episodes) != len(performances) or len(episodes) < 2:
        return 0.0
    
    # Normalize to [0, 1] range
    max_episodes = max(episodes)
    max_performance = max(performances) if max(performances) > 0 else 1.0
    
    normalized_episodes = [e / max_episodes for e in episodes]
    normalized_performances = [p / max_performance for p in performances]
    
    # Calculate AUC using trapezoidal rule
    auc = np.trapz(normalized_performances, normalized_episodes)
    
    return auc


class MetricsTracker:
    """Utility class for tracking and calculating metrics during experiments"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all tracked metrics"""
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rates = []
        self.losses = {}
        self.custom_metrics = {}
    
    def update(self, 
               reward: float = None,
               length: int = None, 
               success: bool = None,
               losses: Dict[str, float] = None,
               custom: Dict[str, float] = None):
        """Update tracked metrics"""
        
        if reward is not None:
            self.episode_rewards.append(reward)
        
        if length is not None:
            self.episode_lengths.append(length)
        
        if success is not None:
            self.success_rates.append(float(success))
        
        if losses:
            for key, value in losses.items():
                if key not in self.losses:
                    self.losses[key] = []
                self.losses[key].append(value)
        
        if custom:
            for key, value in custom.items():
                if key not in self.custom_metrics:
                    self.custom_metrics[key] = []
                self.custom_metrics[key].append(value)
    
    def get_summary(self, window: int = 100) -> Dict[str, float]:
        """Get summary statistics for recent performance"""
        summary = {}
        
        if self.episode_rewards:
            recent_rewards = self.episode_rewards[-window:]
            summary.update({
                'mean_reward': np.mean(recent_rewards),
                'std_reward': np.std(recent_rewards),
                'min_reward': np.min(recent_rewards),
                'max_reward': np.max(recent_rewards)
            })
        
        if self.episode_lengths:
            recent_lengths = self.episode_lengths[-window:]
            summary.update({
                'mean_length': np.mean(recent_lengths),
                'std_length': np.std(recent_lengths)
            })
        
        if self.success_rates:
            recent_success = self.success_rates[-window:]
            summary.update({
                'success_rate': np.mean(recent_success),
                'consistency_score': calculate_consistency_score(recent_success)
            })
        
        # Add loss summaries
        for loss_name, loss_values in self.losses.items():
            if loss_values:
                recent_losses = loss_values[-window:]
                summary[f'{loss_name}_mean'] = np.mean(recent_losses)
                summary[f'{loss_name}_std'] = np.std(recent_losses)
        
        # Add custom metric summaries
        for metric_name, metric_values in self.custom_metrics.items():
            if metric_values:
                recent_values = metric_values[-window:]
                summary[f'{metric_name}_mean'] = np.mean(recent_values)
                summary[f'{metric_name}_std'] = np.std(recent_values)
        
        return summary
    
    def get_learning_curve_auc(self) -> float:
        """Calculate AUC of the learning curve"""
        if not self.success_rates:
            return 0.0
        
        episodes = list(range(len(self.success_rates)))
        return calculate_learning_curve_auc(episodes, self.success_rates)