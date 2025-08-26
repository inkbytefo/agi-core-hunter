"""
Base agent interface for all AGI principle implementations
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
import jax.numpy as jnp
import chex


class BaseAgent(ABC):
    """Abstract base class for all agents in AGI Core Hunter experiments"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = self.__class__.__name__
        
    @abstractmethod
    def act(self, observation: chex.Array, rng_key: chex.PRNGKey) -> Tuple[chex.Array, Dict[str, Any]]:
        """
        Select action given observation
        
        Args:
            observation: Current environment observation
            rng_key: JAX random key
            
        Returns:
            action: Selected action
            info: Additional information (e.g., value estimates, attention weights)
        """
        pass
    
    @abstractmethod
    def update(self, batch: Dict[str, chex.Array]) -> Dict[str, float]:
        """
        Update agent parameters given a batch of experience
        
        Args:
            batch: Dictionary containing observations, actions, rewards, etc.
            
        Returns:
            metrics: Training metrics (loss, accuracy, etc.)
        """
        pass
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, float]:
        """Return current agent metrics for logging"""
        pass
    
    def save_checkpoint(self, path: str) -> None:
        """Save agent state to disk"""
        raise NotImplementedError("Checkpoint saving not implemented")
    
    def load_checkpoint(self, path: str) -> None:
        """Load agent state from disk"""
        raise NotImplementedError("Checkpoint loading not implemented")
    
    def reset(self) -> None:
        """Reset agent state (e.g., for new episode)"""
        pass