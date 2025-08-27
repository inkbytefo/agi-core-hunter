"""
FEP (Free Energy Principle) Agent Implementation

This agent implements the Free Energy Principle through active inference,
minimizing variational free energy via both perception (belief updating)
and action (environmental manipulation). The agent maintains generative
models of the world and acts to minimize surprise.
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


class GenerativeModel(nn.Module):
    """
    Generative model implementing the core components of active inference:
    - A matrix: P(observation | hidden_state) 
    - B matrix: P(hidden_state_t+1 | hidden_state_t, action)
    """
    obs_dim: int
    hidden_dim: int
    action_dim: int
    
    @nn.compact
    def __call__(self, hidden_state, action=None):
        # A matrix: observation likelihood
        a_logits = nn.Dense(self.obs_dim * self.hidden_dim, name='a_matrix')(
            jnp.ones((1, 1))  # learnable parameters
        )
        a_matrix = nn.softmax(a_logits.reshape(self.hidden_dim, self.obs_dim))
        
        # B matrix: transition likelihood 
        if action is not None:
            # Create state-action pairs for transition learning
            state_action = jnp.concatenate([hidden_state, action], axis=-1)
            b_logits = nn.Dense(self.hidden_dim, name='b_matrix')(state_action)
            b_transition = nn.softmax(b_logits)
        else:
            b_transition = None
            
        return a_matrix, b_transition


class BeliefNetwork(nn.Module):
    """Network for updating posterior beliefs over hidden states"""
    hidden_dim: int
    
    @nn.compact
    def __call__(self, observation, prior_belief):
        # Combine observation with prior belief for posterior update
        obs_encoded = nn.Dense(32)(observation)
        obs_encoded = nn.relu(obs_encoded)
        
        belief_encoded = nn.Dense(32)(prior_belief)
        belief_encoded = nn.relu(belief_encoded)
        
        combined = jnp.concatenate([obs_encoded, belief_encoded], axis=-1)
        
        # Output posterior belief (log probabilities)
        hidden = nn.Dense(64)(combined)
        hidden = nn.relu(hidden)
        
        posterior_logits = nn.Dense(self.hidden_dim)(hidden)
        posterior_belief = nn.softmax(posterior_logits)
        
        return posterior_belief, posterior_logits


class PolicyNetwork(nn.Module):
    """Policy network for action selection based on expected free energy"""
    action_dim: int
    hidden_dim: int
    
    @nn.compact
    def __call__(self, belief, preferences, epistemic_value, pragmatic_value):
        # Encode current belief
        belief_encoded = nn.Dense(32)(belief)
        belief_encoded = nn.relu(belief_encoded)
        
        # Encode preferences 
        pref_encoded = nn.Dense(16)(preferences)
        pref_encoded = nn.relu(pref_encoded)
        
        # Combine with epistemic and pragmatic values
        values_combined = jnp.concatenate([epistemic_value, pragmatic_value], axis=-1)
        values_encoded = nn.Dense(16)(values_combined)
        values_encoded = nn.relu(values_encoded)
        
        # Combine all features
        combined = jnp.concatenate([belief_encoded, pref_encoded, values_encoded], axis=-1)
        
        # Output action logits
        hidden = nn.Dense(64)(combined)
        hidden = nn.relu(hidden)
        
        action_logits = nn.Dense(self.action_dim)(hidden)
        
        return action_logits


class ExpectedFreeEnergyCalculator(nn.Module):
    """Calculate expected free energy components"""
    obs_dim: int
    hidden_dim: int
    action_dim: int
    
    @nn.compact
    def __call__(self, belief, action_onehot):
        # Epistemic value: information gain from action
        belief_encoded = nn.Dense(32)(belief)
        action_encoded = nn.Dense(16)(action_onehot)
        
        combined = jnp.concatenate([belief_encoded, action_encoded], axis=-1)
        hidden = nn.Dense(32)(combined)
        hidden = nn.relu(hidden)
        
        # Predict information gain (epistemic value)
        epistemic_value = nn.Dense(1, name='epistemic')(hidden)
        
        # Pragmatic value: expected reward/preference satisfaction
        pragmatic_value = nn.Dense(1, name='pragmatic')(hidden)
        
        return epistemic_value, pragmatic_value


class FEPAgent(BaseAgent):
    """
    Agent implementing Free Energy Principle through active inference
    
    The agent minimizes variational free energy by:
    1. Updating beliefs about hidden states (perception)
    2. Selecting actions to minimize expected free energy (action)
    3. Maintaining preferences over observations (goals)
    4. Balancing exploration (epistemic value) and exploitation (pragmatic value)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Extract configuration
        self.obs_dim = config["obs_dim"]
        self.action_dim = config["action_dim"]
        self.hidden_dim = config.get("hidden_dim", 16)
        self.precision = config.get("precision", 1.0)  # Precision parameter (inverse temperature)
        self.epistemic_weight = config.get("epistemic_weight", 1.0)  # Weight for exploration
        self.learning_rate = config.get("learning_rate", 3e-4)
        
        # Initialize networks
        self.generative_model = GenerativeModel(
            obs_dim=self.obs_dim,
            hidden_dim=self.hidden_dim, 
            action_dim=self.action_dim
        )
        self.belief_network = BeliefNetwork(hidden_dim=self.hidden_dim)
        self.policy_network = PolicyNetwork(
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim
        )
        self.efe_calculator = ExpectedFreeEnergyCalculator(
            obs_dim=self.obs_dim,
            hidden_dim=self.hidden_dim,
            action_dim=self.action_dim
        )
        
        # Training states
        self.generative_state = None
        self.belief_state = None
        self.policy_state = None
        self.efe_state = None
        
        # Agent state
        self.current_belief = None
        self.preferences = None  # Will be initialized in setup
        
        # Metrics tracking
        self.metrics = {
            "variational_free_energy": 0.0,
            "expected_free_energy": 0.0,
            "epistemic_value": 0.0,
            "pragmatic_value": 0.0,
            "surprise": 0.0,
            "belief_entropy": 0.0,
            "exploration_bonus": 0.0
        }
    
    def setup(self, rng_key: chex.PRNGKey, dummy_obs: chex.Array):
        """Initialize network parameters and agent state"""
        key1, key2, key3, key4, key5 = jax.random.split(rng_key, 5)
        
        # Initialize generative model
        dummy_hidden = jnp.ones((self.hidden_dim,)) / self.hidden_dim  # uniform belief
        dummy_action = jnp.ones((self.action_dim,)) / self.action_dim  # uniform action
        
        generative_params = self.generative_model.init(key1, dummy_hidden, dummy_action)
        generative_optimizer = optax.adam(self.learning_rate)
        
        self.generative_state = TrainState.create(
            apply_fn=self.generative_model.apply,
            params=generative_params,
            tx=generative_optimizer
        )
        
        # Initialize belief network
        belief_params = self.belief_network.init(key2, dummy_obs, dummy_hidden)
        belief_optimizer = optax.adam(self.learning_rate)
        
        self.belief_state = TrainState.create(
            apply_fn=self.belief_network.apply,
            params=belief_params, 
            tx=belief_optimizer
        )
        
        # Initialize policy network
        dummy_preferences = jnp.ones((self.obs_dim,)) / self.obs_dim
        dummy_epistemic = jnp.ones((self.action_dim,)) * 0.5  # Array for each action
        dummy_pragmatic = jnp.ones((self.action_dim,)) * 0.5  # Array for each action
        
        policy_params = self.policy_network.init(
            key3, dummy_hidden, dummy_preferences, dummy_epistemic, dummy_pragmatic
        )
        policy_optimizer = optax.adam(self.learning_rate)
        
        self.policy_state = TrainState.create(
            apply_fn=self.policy_network.apply,
            params=policy_params,
            tx=policy_optimizer
        )
        
        # Initialize expected free energy calculator
        dummy_action_onehot = jnp.ones((self.action_dim,)) / self.action_dim
        efe_params = self.efe_calculator.init(key4, dummy_hidden, dummy_action_onehot)
        efe_optimizer = optax.adam(self.learning_rate)
        
        self.efe_state = TrainState.create(
            apply_fn=self.efe_calculator.apply,
            params=efe_params,
            tx=efe_optimizer
        )
        
        # Initialize agent state
        self.current_belief = dummy_hidden
        
        # Initialize preferences (can be learned or set based on task)
        self.preferences = jnp.ones((self.obs_dim,)) / self.obs_dim
    
    def update_beliefs(self, observation: chex.Array, rng_key: chex.PRNGKey) -> chex.Array:
        """Update posterior beliefs given new observation"""
        posterior_belief, posterior_logits = self.belief_state.apply_fn(
            self.belief_state.params, observation, self.current_belief
        )
        
        # Update current belief state
        self.current_belief = posterior_belief
        
        return posterior_belief, posterior_logits
    
    def calculate_variational_free_energy(self, observation: chex.Array, 
                                        belief: chex.Array) -> float:
        """Calculate variational free energy = Accuracy + Complexity"""
        # Get observation likelihood from generative model
        a_matrix, _ = self.generative_state.apply_fn(
            self.generative_state.params, belief
        )
        
        # Accuracy: log P(observation | belief, A)
        obs_likelihood = jnp.sum(a_matrix * belief[:, None], axis=0)
        accuracy = jnp.sum(observation * jnp.log(obs_likelihood + 1e-8))
        
        # Complexity: KL divergence from prior (uniform)
        prior_belief = jnp.ones_like(belief) / len(belief)
        complexity = jnp.sum(belief * jnp.log(belief / prior_belief + 1e-8))
        
        # Variational free energy
        vfe = -accuracy + complexity
        
        return vfe, accuracy, complexity
    
    def calculate_expected_free_energy(self, belief: chex.Array) -> Tuple[chex.Array, chex.Array]:
        """Calculate expected free energy for each possible action"""
        epistemic_values = []
        pragmatic_values = []
        
        for action_idx in range(self.action_dim):
            # Create one-hot action vector
            action_onehot = jnp.zeros(self.action_dim)
            action_onehot = action_onehot.at[action_idx].set(1.0)
            
            # Calculate epistemic and pragmatic values
            epistemic_val, pragmatic_val = self.efe_state.apply_fn(
                self.efe_state.params, belief, action_onehot
            )
            
            epistemic_values.append(epistemic_val[0])
            pragmatic_values.append(pragmatic_val[0])
        
        epistemic_values = jnp.array(epistemic_values)
        pragmatic_values = jnp.array(pragmatic_values)
        
        # Expected free energy = -epistemic_value - pragmatic_value
        # (negative because we want to maximize these values)
        expected_free_energy = -(self.epistemic_weight * epistemic_values + pragmatic_values)
        
        return expected_free_energy, epistemic_values, pragmatic_values
    
    def act(self, observation: chex.Array, rng_key: chex.PRNGKey) -> Tuple[chex.Array, Dict[str, Any]]:
        """Select action using active inference"""
        key1, key2 = jax.random.split(rng_key)
        
        # Update beliefs given observation
        posterior_belief, posterior_logits = self.update_beliefs(observation, key1)
        
        # Calculate variational free energy
        vfe, accuracy, complexity = self.calculate_variational_free_energy(
            observation, posterior_belief
        )
        
        # Calculate expected free energy for action selection
        efe, epistemic_values, pragmatic_values = self.calculate_expected_free_energy(
            posterior_belief
        )
        
        # Get action logits from policy network
        action_logits = self.policy_state.apply_fn(
            self.policy_state.params, 
            posterior_belief, 
            self.preferences,
            epistemic_values,
            pragmatic_values
        )
        
        # Apply precision (inverse temperature) to action selection
        scaled_logits = action_logits * self.precision
        
        # Sample action
        action = jax.random.categorical(key2, scaled_logits)
        
        # Calculate metrics
        surprise = -jnp.sum(observation * jnp.log(
            jnp.sum(posterior_belief[:, None] * self.generative_state.apply_fn(
                self.generative_state.params, posterior_belief
            )[0], axis=0) + 1e-8
        ))
        
        belief_entropy = -jnp.sum(posterior_belief * jnp.log(posterior_belief + 1e-8))
        
        # Update metrics
        self.metrics.update({
            "variational_free_energy": float(vfe),
            "expected_free_energy": float(jnp.mean(efe)),
            "epistemic_value": float(jnp.mean(epistemic_values)),
            "pragmatic_value": float(jnp.mean(pragmatic_values)),
            "surprise": float(surprise),
            "belief_entropy": float(belief_entropy),
            "exploration_bonus": float(self.epistemic_weight * jnp.mean(epistemic_values))
        })
        
        info = {
            "belief_state": posterior_belief,
            "variational_free_energy": vfe,
            "expected_free_energy": efe,
            "epistemic_values": epistemic_values,
            "pragmatic_values": pragmatic_values,
            "action_logits": action_logits,
            "surprise": surprise,
            "belief_entropy": belief_entropy
        }
        
        return action, info
    
    def update(self, batch: Dict[str, chex.Array]) -> Dict[str, float]:
        """Update agent parameters given a batch of experience"""
        observations = batch["observations"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_observations = batch["next_observations"]
        
        def loss_fn(params_dict):
            total_loss = 0.0
            metrics = {}
            
            batch_size = observations.shape[0]
            
            for i in range(batch_size):
                obs = observations[i]
                action = actions[i]
                reward = rewards[i]
                next_obs = next_observations[i]
                
                # Update beliefs
                belief, belief_logits = self.belief_state.apply_fn(
                    params_dict["belief"], obs, self.current_belief
                )
                
                # Calculate VFE loss
                vfe, accuracy, complexity = self.calculate_variational_free_energy(obs, belief)
                
                # Calculate EFE loss for policy learning
                action_onehot = jnp.zeros(self.action_dim)
                action_onehot = action_onehot.at[action].set(1.0)
                
                epistemic_val, pragmatic_val = self.efe_state.apply_fn(
                    params_dict["efe"], belief, action_onehot
                )
                
                # Policy loss: minimize negative expected reward
                policy_logits = self.policy_state.apply_fn(
                    params_dict["policy"], belief, self.preferences, 
                    epistemic_val, pragmatic_val
                )
                
                action_prob = nn.softmax(policy_logits)[action]
                policy_loss = -reward * jnp.log(action_prob + 1e-8)
                
                # Generative model loss: predict next observation
                a_matrix, b_transition = self.generative_state.apply_fn(
                    params_dict["generative"], belief, action_onehot
                )
                
                if b_transition is not None:
                    # Prediction loss
                    next_belief, _ = self.belief_state.apply_fn(
                        params_dict["belief"], next_obs, belief
                    )
                    prediction_loss = jnp.mean((b_transition - next_belief) ** 2)
                else:
                    prediction_loss = 0.0
                
                # Combine losses
                total_loss += vfe + policy_loss + prediction_loss
                
            # Average over batch
            total_loss /= batch_size
            
            metrics.update({
                "total_loss": total_loss,
                "vfe_loss": vfe,
                "policy_loss": policy_loss,
                "prediction_loss": prediction_loss
            })
            
            return total_loss, metrics
        
        # Get gradients and update parameters
        params_dict = {
            "generative": self.generative_state.params,
            "belief": self.belief_state.params,
            "policy": self.policy_state.params,
            "efe": self.efe_state.params
        }
        
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params_dict)
        
        # Update each component
        self.generative_state = self.generative_state.apply_gradients(
            grads=grads["generative"]
        )
        self.belief_state = self.belief_state.apply_gradients(
            grads=grads["belief"]
        )
        self.policy_state = self.policy_state.apply_gradients(
            grads=grads["policy"]
        )
        self.efe_state = self.efe_state.apply_gradients(
            grads=grads["efe"]
        )
        
        return metrics
    
    def get_metrics(self) -> Dict[str, float]:
        """Return current agent metrics for logging"""
        return self.metrics.copy()
    
    def set_preferences(self, preferences: chex.Array):
        """Set agent preferences over observations"""
        self.preferences = preferences
    
    def reset(self):
        """Reset agent state for new episode"""
        if self.current_belief is not None:
            # Reset to uniform belief
            self.current_belief = jnp.ones(self.hidden_dim) / self.hidden_dim