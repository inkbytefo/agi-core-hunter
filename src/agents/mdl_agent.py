"""
MDL (Minimum Description Length) Agent Implementation

This agent implements the MDL principle by using a β-VAE to compress
state representations and training a policy on the compressed latent space.
"""

from typing import Any, Dict, Tuple
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import chex
from flax.training.train_state import TrainState

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.base_agent import BaseAgent


class VAEEncoder(nn.Module):
    """Variational Autoencoder Encoder for state compression"""
    latent_dim: int
    hidden_dims: Tuple[int, ...] = (64, 32)
    
    @nn.compact
    def __call__(self, x):
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)
        
        # Output mean and log variance for latent distribution
        mean = nn.Dense(self.latent_dim)(x)
        log_var = nn.Dense(self.latent_dim)(x)
        
        return mean, log_var


class VAEDecoder(nn.Module):
    """Variational Autoencoder Decoder for reconstruction"""
    output_dim: int
    hidden_dims: Tuple[int, ...] = (32, 64)
    
    @nn.compact
    def __call__(self, z):
        for dim in self.hidden_dims:
            z = nn.Dense(dim)(z)
            z = nn.relu(z)
        
        reconstruction = nn.Dense(self.output_dim)(z)
        return reconstruction


class PolicyNetwork(nn.Module):
    """Policy network operating on compressed latent space"""
    action_dim: int
    hidden_dims: Tuple[int, ...] = (32, 32)
    
    @nn.compact
    def __call__(self, z):
        x = z
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)
        
        logits = nn.Dense(self.action_dim)(x)
        return logits


class ValueNetwork(nn.Module):
    """Value network for critic"""
    hidden_dims: Tuple[int, ...] = (32, 32)
    
    @nn.compact
    def __call__(self, z):
        x = z
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)
        
        value = nn.Dense(1)(x)
        return jnp.squeeze(value, axis=-1)


class MDLAgent(BaseAgent):
    """
    Agent implementing MDL principle through β-VAE compression
    
    The agent learns to compress observations into a minimal latent representation
    while maintaining task-relevant information. The compression pressure (β parameter)
    implements the MDL principle by penalizing complex representations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Extract configuration
        self.obs_dim = config["obs_dim"]
        self.action_dim = config["action_dim"]
        self.latent_dim = config.get("latent_dim", 8)
        self.beta = config.get("beta", 1.0)  # β parameter for VAE
        self.learning_rate = config.get("learning_rate", 3e-4)
        
        # Initialize networks
        self.encoder = VAEEncoder(latent_dim=self.latent_dim)
        self.decoder = VAEDecoder(output_dim=self.obs_dim)
        self.policy = PolicyNetwork(action_dim=self.action_dim)
        self.value = ValueNetwork()
        
        # Initialize training states (will be set in setup)
        self.vae_state = None
        self.policy_state = None
        self.value_state = None
        
        # Metrics tracking
        self.metrics = {
            "reconstruction_loss": 0.0,
            "kl_loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "compression_ratio": 0.0
        }
    
    def setup(self, rng_key: chex.PRNGKey, dummy_obs: chex.Array):
        """Initialize network parameters and optimizers"""
        key1, key2, key3, key4 = jax.random.split(rng_key, 4)
        
        # Initialize VAE
        encoder_params = self.encoder.init(key1, dummy_obs)
        dummy_z = jax.random.normal(key2, (self.latent_dim,))
        decoder_params = self.decoder.init(key3, dummy_z)
        
        vae_params = {
            "encoder": encoder_params,
            "decoder": decoder_params
        }
        
        vae_optimizer = optax.adam(self.learning_rate)
        self.vae_state = TrainState.create(
            apply_fn=None,  # We'll handle apply manually
            params=vae_params,
            tx=vae_optimizer
        )
        
        # Initialize policy and value networks
        policy_params = self.policy.init(key3, dummy_z)
        value_params = self.value.init(key4, dummy_z)
        
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
    
    def encode(self, observation: chex.Array, rng_key: chex.PRNGKey) -> chex.Array:
        """Encode observation to latent space using reparameterization trick"""
        mean, log_var = self.encoder.apply(
            self.vae_state.params["encoder"], observation
        )
        
        # Reparameterization trick
        std = jnp.exp(0.5 * log_var)
        eps = jax.random.normal(rng_key, mean.shape)
        z = mean + std * eps
        
        return z, mean, log_var
    
    def act(self, observation: chex.Array, rng_key: chex.PRNGKey) -> Tuple[chex.Array, Dict[str, Any]]:
        """Select action using compressed representation"""
        key1, key2 = jax.random.split(rng_key)
        
        # Encode observation
        z, mean, log_var = self.encode(observation, key1)
        
        # Get action logits from policy
        logits = self.policy_state.apply_fn(self.policy_state.params, z)
        
        # Sample action
        action = jax.random.categorical(key2, logits)
        
        # Get value estimate
        value = self.value_state.apply_fn(self.value_state.params, z)
        
        info = {
            "latent_representation": z,
            "value_estimate": value,
            "action_logits": logits,
            "latent_mean": mean,
            "latent_log_var": log_var
        }
        
        return action, info
    
    def update(self, batch: Dict[str, chex.Array]) -> Dict[str, float]:
        """Update all network parameters"""
        # Update VAE
        vae_metrics = self._update_vae(batch)
        
        # Update policy and value networks
        rl_metrics = self._update_rl(batch)
        
        # Combine metrics
        self.metrics.update(vae_metrics)
        self.metrics.update(rl_metrics)
        
        # Calculate compression ratio
        self.metrics["compression_ratio"] = self.obs_dim / self.latent_dim
        
        return self.metrics
    
    def _update_vae(self, batch: Dict[str, chex.Array]) -> Dict[str, float]:
        """Update VAE parameters"""
        def vae_loss_fn(params, observations, rng_key):
            # Encode
            mean, log_var = self.encoder.apply(params["encoder"], observations)
            
            # Reparameterization
            std = jnp.exp(0.5 * log_var)
            eps = jax.random.normal(rng_key, mean.shape)
            z = mean + std * eps
            
            # Decode
            reconstruction = self.decoder.apply(params["decoder"], z)
            
            # Reconstruction loss (MSE)
            recon_loss = jnp.mean((observations - reconstruction) ** 2)
            
            # KL divergence loss
            kl_loss = -0.5 * jnp.mean(1 + log_var - mean**2 - jnp.exp(log_var))
            
            # Total VAE loss with β weighting
            total_loss = recon_loss + self.beta * kl_loss
            
            return total_loss, {
                "reconstruction_loss": recon_loss,
                "kl_loss": kl_loss
            }
        
        # Compute gradients and update
        rng_key = jax.random.PRNGKey(0)  # In practice, use proper key management
        grad_fn = jax.value_and_grad(vae_loss_fn, has_aux=True)
        (loss, metrics), grads = grad_fn(
            self.vae_state.params, batch["observations"], rng_key
        )
        
        self.vae_state = self.vae_state.apply_gradients(grads=grads)
        
        return metrics
    
    def _update_rl(self, batch: Dict[str, chex.Array]) -> Dict[str, float]:
        """Update policy and value networks (simplified A2C-style update)"""
        # This is a simplified implementation
        # In practice, you'd want proper advantage estimation, etc.
        
        def policy_loss_fn(params, latent_states, actions, advantages):
            logits = self.policy.apply(params, latent_states)
            log_probs = jax.nn.log_softmax(logits)
            action_log_probs = log_probs[jnp.arange(len(actions)), actions]
            
            policy_loss = -jnp.mean(action_log_probs * advantages)
            return policy_loss
        
        def value_loss_fn(params, latent_states, returns):
            values = self.value.apply(params, latent_states)
            value_loss = jnp.mean((values - returns) ** 2)
            return value_loss
        
        # Extract latent states (simplified - in practice, re-encode observations)
        latent_states = batch.get("latent_states", jnp.zeros((len(batch["observations"]), self.latent_dim)))
        
        # Compute simple advantages (returns - values)
        returns = batch.get("returns", jnp.zeros(len(batch["observations"])))
        values = self.value_state.apply_fn(self.value_state.params, latent_states)
        advantages = returns - values
        
        # Update policy
        policy_grad_fn = jax.grad(policy_loss_fn)
        policy_grads = policy_grad_fn(
            self.policy_state.params, latent_states, batch["actions"], advantages
        )
        self.policy_state = self.policy_state.apply_gradients(grads=policy_grads)
        
        # Update value function
        value_grad_fn = jax.grad(value_loss_fn)
        value_grads = value_grad_fn(
            self.value_state.params, latent_states, returns
        )
        self.value_state = self.value_state.apply_gradients(grads=value_grads)
        
        return {
            "policy_loss": policy_loss_fn(self.policy_state.params, latent_states, batch["actions"], advantages),
            "value_loss": value_loss_fn(self.value_state.params, latent_states, returns)
        }
    
    def get_metrics(self) -> Dict[str, float]:
        """Return current metrics"""
        return self.metrics.copy()