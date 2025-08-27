"""
Causal Agent Implementation

This agent implements the Causality principle by learning causal relationships
between variables in the environment and using causal reasoning for action selection.
The agent builds a Structural Causal Model (SCM) and can adapt to interventions.
"""

from typing import Any, Dict, Tuple, Optional
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import chex
from flax.training.train_state import TrainState
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.base_agent import BaseAgent


class CausalEncoder(nn.Module):
    """Encodes observations into causal variables"""
    causal_dim: int
    hidden_dims: Tuple[int, ...] = (64, 32)
    
    @nn.compact
    def __call__(self, x):
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)
        
        # Output causal variables (continuous representations)
        causal_vars = nn.Dense(self.causal_dim)(x)
        return causal_vars


class CausalGraphNetwork(nn.Module):
    """Learns the causal graph structure between variables"""
    causal_dim: int
    
    @nn.compact
    def __call__(self, causal_vars):
        # Predict causal relationships (adjacency matrix)
        # Each variable can influence others
        batch_size = causal_vars.shape[0]
        
        # Create pairwise features for causal prediction
        expanded_vars = jnp.expand_dims(causal_vars, axis=2)  # [batch, causal_dim, 1]
        repeated_vars = jnp.expand_dims(causal_vars, axis=1)  # [batch, 1, causal_dim]
        
        # Broadcast to create all pairs
        expanded_vars = jnp.broadcast_to(expanded_vars, (batch_size, self.causal_dim, self.causal_dim))
        repeated_vars = jnp.broadcast_to(repeated_vars, (batch_size, self.causal_dim, self.causal_dim))
        
        # Concatenate for pairwise features
        pairwise_features = jnp.concatenate([expanded_vars, repeated_vars], axis=-1)
        pairwise_features = pairwise_features.reshape(batch_size, self.causal_dim * self.causal_dim, 2)
        
        # Predict causal strengths
        causal_logits = nn.Dense(32)(pairwise_features)
        causal_logits = nn.relu(causal_logits)
        causal_strengths = nn.Dense(1)(causal_logits)
        
        # Reshape back to adjacency matrix
        causal_matrix = causal_strengths.reshape(batch_size, self.causal_dim, self.causal_dim)
        
        return causal_matrix


class InterventionPredictor(nn.Module):
    """Predicts outcomes under interventions using do-calculus"""
    causal_dim: int
    hidden_dims: Tuple[int, ...] = (32, 32)
    
    @nn.compact
    def __call__(self, causal_vars, causal_matrix, intervention_mask=None, intervention_values=None):
        """
        Predict outcomes under interventions
        
        Args:
            causal_vars: Current causal variable values
            causal_matrix: Learned causal adjacency matrix
            intervention_mask: Binary mask indicating which variables are intervened on
            intervention_values: Values to set for intervened variables
        """
        if intervention_mask is not None and intervention_values is not None:
            # Apply intervention: do(X_i = x_i)
            intervened_vars = jnp.where(intervention_mask, intervention_values, causal_vars)
        else:
            intervened_vars = causal_vars
        
        # Predict next state using causal relationships
        # Simple linear causal model: X_{t+1} = f(parents(X_t))
        influenced = jnp.einsum('bij,bj->bi', causal_matrix, intervened_vars)
        
        # Add non-linear transformation
        for dim in self.hidden_dims:
            influenced = nn.Dense(dim)(influenced)
            influenced = nn.relu(influenced)
        
        predicted_vars = nn.Dense(self.causal_dim)(influenced)
        return predicted_vars


class CausalPolicyNetwork(nn.Module):
    """Policy network that reasons about causal effects of actions"""
    action_dim: int
    causal_dim: int
    hidden_dims: Tuple[int, ...] = (32, 32)
    
    @nn.compact
    def __call__(self, causal_vars, causal_matrix):
        # Combine causal variables and graph structure for action selection
        # Flatten causal matrix and concatenate with variables
        flattened_matrix = causal_matrix.reshape(causal_vars.shape[0], -1)
        combined_features = jnp.concatenate([causal_vars, flattened_matrix], axis=-1)
        
        x = combined_features
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)
        
        logits = nn.Dense(self.action_dim)(x)
        return logits


class ValueNetwork(nn.Module):
    """Value network for critic"""
    hidden_dims: Tuple[int, ...] = (32, 32)
    
    @nn.compact
    def __call__(self, causal_vars):
        x = causal_vars
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)
        
        value = nn.Dense(1)(x)
        return jnp.squeeze(value, axis=-1)


class CausalAgent(BaseAgent):
    """
    Agent implementing Causality principle through causal reasoning
    
    The agent learns causal relationships between environmental variables
    and uses causal inference (including do-calculus) for planning and
    adaptation to interventions.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Extract configuration
        self.obs_dim = config["obs_dim"]
        self.action_dim = config["action_dim"]
        self.causal_dim = config.get("causal_dim", 8)
        self.causal_strength = config.get("causal_strength", 1.0)  # Strength of causal regularization
        self.learning_rate = config.get("learning_rate", 3e-4)
        
        # Initialize networks
        self.encoder = CausalEncoder(causal_dim=self.causal_dim)
        self.causal_graph = CausalGraphNetwork(causal_dim=self.causal_dim)
        self.intervention_predictor = InterventionPredictor(causal_dim=self.causal_dim)
        self.policy = CausalPolicyNetwork(action_dim=self.action_dim, causal_dim=self.causal_dim)
        self.value = ValueNetwork()
        
        # Training states
        self.causal_state = None
        self.policy_state = None
        self.value_state = None
        
        # Track intervention history for adaptation
        self.intervention_history = []
        
        # Metrics tracking
        self.metrics = {
            "causal_loss": 0.0,
            "intervention_loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "causal_accuracy": 0.0,
            "intervention_adaptation": 0.0
        }
    
    def setup(self, rng_key: chex.PRNGKey, dummy_obs: chex.Array):
        """Initialize network parameters and optimizers"""
        key1, key2, key3, key4, key5 = jax.random.split(rng_key, 5)
        
        # Initialize causal networks
        encoder_params = self.encoder.init(key1, dummy_obs)
        dummy_causal_vars = jax.random.normal(key2, (self.causal_dim,))
        causal_graph_params = self.causal_graph.init(key2, dummy_causal_vars[None, :])
        intervention_params = self.intervention_predictor.init(
            key3, dummy_causal_vars[None, :], 
            jax.random.normal(key3, (1, self.causal_dim, self.causal_dim))
        )
        
        causal_params = {
            "encoder": encoder_params,
            "causal_graph": causal_graph_params,
            "intervention_predictor": intervention_params
        }
        
        causal_optimizer = optax.adam(self.learning_rate)
        self.causal_state = TrainState.create(
            apply_fn=None,
            params=causal_params,
            tx=causal_optimizer
        )
        
        # Initialize policy and value networks
        dummy_matrix = jax.random.normal(key4, (1, self.causal_dim, self.causal_dim))
        policy_params = self.policy.init(key4, dummy_causal_vars[None, :], dummy_matrix)
        value_params = self.value.init(key5, dummy_causal_vars[None, :])
        
        policy_optimizer = optax.adam(self.learning_rate)
        value_optimizer = optax.adam(self.learning_rate)
        
        self.policy_state = TrainState.create(
            apply_fn=self.policy.apply,
            params=policy_params,
            tx=policy_optimizer
        )
        
        self.value_state = TrainState.create(
            apply_fn=self.value.apply,
            params=value_params,
            tx=value_optimizer
        )
    
    def encode_causal_variables(self, observation: chex.Array) -> chex.Array:
        """Extract causal variables from observation"""
        return self.encoder.apply(
            self.causal_state.params["encoder"], observation
        )
    
    def get_causal_graph(self, causal_vars: chex.Array) -> chex.Array:
        """Get causal adjacency matrix"""
        return self.causal_graph.apply(
            self.causal_state.params["causal_graph"], causal_vars
        )
    
    def predict_intervention(self, causal_vars: chex.Array, causal_matrix: chex.Array,
                           intervention_mask: Optional[chex.Array] = None,
                           intervention_values: Optional[chex.Array] = None) -> chex.Array:
        """Predict outcome under intervention"""
        return self.intervention_predictor.apply(
            self.causal_state.params["intervention_predictor"],
            causal_vars, causal_matrix, intervention_mask, intervention_values
        )
    
    def act(self, observation: chex.Array, rng_key: chex.PRNGKey) -> Tuple[chex.Array, Dict[str, Any]]:
        """Select action using causal reasoning"""
        key1, key2 = jax.random.split(rng_key)
        
        # Extract causal variables
        causal_vars = self.encode_causal_variables(observation[None, :])
        
        # Get causal graph
        causal_matrix = self.get_causal_graph(causal_vars)
        
        # Get action logits using causal reasoning
        logits = self.policy_state.apply_fn(self.policy_state.params, causal_vars, causal_matrix)
        
        # Sample action
        action = jax.random.categorical(key2, logits[0])
        
        # Get value estimate
        value = self.value_state.apply_fn(self.value_state.params, causal_vars)
        
        info = {
            "causal_variables": causal_vars[0],
            "causal_matrix": causal_matrix[0],
            "value_estimate": value[0],
            "action_logits": logits[0]
        }
        
        return action, info
    
    def update(self, batch: Dict[str, chex.Array]) -> Dict[str, float]:
        """Update agent parameters using causal learning"""
        # Extract batch data
        observations = batch["observations"]
        next_observations = batch["next_observations"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        dones = batch["dones"]
        
        # Update causal networks
        causal_loss, causal_metrics = self._update_causal_model(
            observations, next_observations, actions
        )
        
        # Update policy and value networks
        policy_loss, value_loss, rl_metrics = self._update_policy_value(
            observations, actions, rewards, next_observations, dones
        )
        
        # Combine metrics
        metrics = {
            "causal_loss": causal_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            **causal_metrics,
            **rl_metrics
        }
        
        self.metrics.update(metrics)
        return metrics
    
    def _update_causal_model(self, observations, next_observations, actions):
        """Update causal model components"""
        def causal_loss_fn(params):
            # Encode current and next causal variables
            current_causal = self.encoder.apply(params["encoder"], observations)
            next_causal = self.encoder.apply(params["encoder"], next_observations)
            
            # Get causal graph
            causal_matrix = self.causal_graph.apply(params["causal_graph"], current_causal)
            
            # Predict next causal variables
            predicted_next = self.intervention_predictor.apply(
                params["intervention_predictor"], current_causal, causal_matrix
            )
            
            # Causal consistency loss
            causal_prediction_loss = jnp.mean((predicted_next - next_causal) ** 2)
            
            # Sparsity regularization for causal graph
            sparsity_loss = jnp.mean(jnp.abs(causal_matrix))
            
            total_loss = causal_prediction_loss + self.causal_strength * sparsity_loss
            
            return total_loss, {
                "causal_prediction_loss": causal_prediction_loss,
                "sparsity_loss": sparsity_loss
            }
        
        # Compute gradients and update
        (loss, metrics), grads = jax.value_and_grad(causal_loss_fn, has_aux=True)(
            self.causal_state.params
        )
        
        self.causal_state = self.causal_state.apply_gradients(grads=grads)
        
        return loss, metrics
    
    def _update_policy_value(self, observations, actions, rewards, next_observations, dones):
        """Update policy and value networks using causal features"""
        # Policy update
        def policy_loss_fn(params):
            causal_vars = self.encoder.apply(
                self.causal_state.params["encoder"], observations
            )
            causal_matrix = self.causal_graph.apply(
                self.causal_state.params["causal_graph"], causal_vars
            )
            
            logits = self.policy.apply(params, causal_vars, causal_matrix)
            log_probs = jax.nn.log_softmax(logits)
            
            # Simple policy gradient (can be enhanced with advantages)
            action_log_probs = jnp.take_along_axis(
                log_probs, actions[:, None], axis=1
            ).squeeze()
            
            policy_loss = -jnp.mean(action_log_probs * rewards)
            return policy_loss
        
        policy_grads = jax.grad(policy_loss_fn)(self.policy_state.params)
        self.policy_state = self.policy_state.apply_gradients(grads=policy_grads)
        policy_loss = policy_loss_fn(self.policy_state.params)
        
        # Value update
        def value_loss_fn(params):
            causal_vars = self.encoder.apply(
                self.causal_state.params["encoder"], observations
            )
            values = self.value.apply(params, causal_vars)
            
            # Simple TD error
            next_causal_vars = self.encoder.apply(
                self.causal_state.params["encoder"], next_observations
            )
            next_values = self.value.apply(params, next_causal_vars)
            targets = rewards + 0.99 * next_values * (1 - dones)
            
            value_loss = jnp.mean((values - targets) ** 2)
            return value_loss
        
        value_grads = jax.grad(value_loss_fn)(self.value_state.params)
        self.value_state = self.value_state.apply_gradients(grads=value_grads)
        value_loss = value_loss_fn(self.value_state.params)
        
        return policy_loss, value_loss, {}
    
    def get_metrics(self) -> Dict[str, float]:
        """Return current agent metrics"""
        return self.metrics.copy()
    
    def adapt_to_intervention(self, intervention_data: Dict[str, Any]):
        """Adapt agent to detected interventions in the environment"""
        self.intervention_history.append(intervention_data)
        
        # Update intervention adaptation metric
        self.metrics["intervention_adaptation"] = len(self.intervention_history) / 100.0
    
    def reset(self) -> None:
        """Reset agent state for new episode"""
        # Clear intervention history for new episode
        pass