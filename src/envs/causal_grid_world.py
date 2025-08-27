"""
Causal Grid World Environment for testing Causality principle

Extended grid world that supports interventions on environmental variables
to test causal reasoning and adaptation capabilities.
"""

from typing import Tuple, Dict, Any, Optional, List
import jax
import jax.numpy as jnp
import chex
from flax import struct

from .grid_world import GridWorld, GridWorldState


@struct.dataclass
class CausalGridWorldState:
    """Extended state for causal grid world with intervention tracking"""
    agent_pos: chex.Array  # (2,) array with [x, y] position
    goal_pos: chex.Array   # (2,) array with [x, y] position
    grid: chex.Array       # (height, width) array with obstacles
    step_count: int
    done: bool
    
    # Causal variables
    wind_strength: float   # Environmental force affecting movement
    visibility: float      # How well agent can see (affects observations)
    goal_stability: float  # How stable the goal position is
    
    # Intervention tracking
    active_interventions: chex.Array  # Binary mask for active interventions
    intervention_values: chex.Array   # Values for intervened variables


class CausalGridWorld(GridWorld):
    """
    Grid world environment with causal variables and intervention support
    
    This environment extends the basic GridWorld with causal variables:
    - Wind: Affects movement reliability
    - Visibility: Affects observation quality  
    - Goal Stability: Goal can shift based on environmental factors
    
    Interventions can be applied to test causal reasoning:
    - do(wind = strong): Force high wind conditions
    - do(visibility = low): Force poor visibility
    - do(goal_stability = low): Make goal position unstable
    """
    
    def __init__(
        self,
        height: int = 8,
        width: int = 8,
        obstacle_prob: float = 0.2,
        max_steps: int = 100,
        reward_goal: float = 10.0,
        reward_step: float = -0.1,
        reward_obstacle: float = -1.0,
        enable_interventions: bool = True,
        intervention_probability: float = 0.1,
        **kwargs
    ):
        super().__init__(height, width, obstacle_prob, max_steps, 
                        reward_goal, reward_step, reward_obstacle)
        
        self.enable_interventions = enable_interventions
        self.intervention_probability = intervention_probability
        
        # Causal variable ranges
        self.wind_range = (0.0, 1.0)
        self.visibility_range = (0.3, 1.0)  # Never completely blind
        self.goal_stability_range = (0.7, 1.0)  # Goal mostly stable
        
        # Intervention types
        self.causal_variables = ["wind_strength", "visibility", "goal_stability"]
        self.num_causal_vars = len(self.causal_variables)
        
        # Update observation dimension to include causal variables
        self.obs_dim = height * width + 4 + self.num_causal_vars
    
    def reset(self, rng_key: chex.PRNGKey) -> Tuple[CausalGridWorldState, chex.Array]:
        """Reset environment with causal variables"""
        key1, key2, key3, key4, key5 = jax.random.split(rng_key, 5)
        
        # Generate basic grid world state
        basic_state, _ = super().reset(key1)
        
        # Initialize causal variables
        wind_strength = jax.random.uniform(key2, (), *self.wind_range)
        visibility = jax.random.uniform(key3, (), *self.visibility_range)
        goal_stability = jax.random.uniform(key4, (), *self.goal_stability_range)
        
        # Initialize no interventions
        active_interventions = jnp.zeros(self.num_causal_vars, dtype=bool)
        intervention_values = jnp.zeros(self.num_causal_vars)
        
        # Sample potential interventions
        if self.enable_interventions:
            intervention_mask = jax.random.bernoulli(
                key5, self.intervention_probability, (self.num_causal_vars,)
            )
            active_interventions = intervention_mask
            
            # Set intervention values
            intervention_wind = jax.random.choice(key5, jnp.array([0.0, 0.9]))
            intervention_visibility = jax.random.choice(key5, jnp.array([0.3, 0.9]))
            intervention_goal_stability = jax.random.choice(key5, jnp.array([0.7, 0.95]))
            
            intervention_values = jnp.array([
                intervention_wind, intervention_visibility, intervention_goal_stability
            ])
            
            # Apply interventions
            wind_strength = jnp.where(active_interventions[0], intervention_values[0], wind_strength)
            visibility = jnp.where(active_interventions[1], intervention_values[1], visibility)
            goal_stability = jnp.where(active_interventions[2], intervention_values[2], goal_stability)
        
        state = CausalGridWorldState(
            agent_pos=basic_state.agent_pos,
            goal_pos=basic_state.goal_pos,
            grid=basic_state.grid,
            step_count=0,
            done=False,
            wind_strength=wind_strength,
            visibility=visibility,
            goal_stability=goal_stability,
            active_interventions=active_interventions,
            intervention_values=intervention_values
        )
        
        observation = self._get_causal_observation(state)
        return state, observation
    
    def step(
        self, 
        state: CausalGridWorldState, 
        action: int
    ) -> Tuple[CausalGridWorldState, chex.Array, float, bool, Dict[str, Any]]:
        """Take step with causal effects"""
        
        # Apply wind effect to movement
        action_delta = self.actions[action]
        
        # Wind can cause movement to fail or deflect
        wind_effect = jax.random.bernoulli(
            jax.random.PRNGKey(state.step_count), 1.0 - state.wind_strength
        )
        effective_delta = jnp.where(wind_effect, action_delta, jnp.zeros(2, dtype=jnp.int32))
        
        new_pos = state.agent_pos + effective_delta
        
        # Check bounds
        new_pos = jnp.clip(new_pos, 0, jnp.array([self.height - 1, self.width - 1]))
        
        # Check obstacles
        is_obstacle = state.grid[new_pos[0], new_pos[1]]
        agent_pos = jnp.where(is_obstacle, state.agent_pos, new_pos)
        
        # Goal instability - goal might shift
        goal_shift_prob = 1.0 - state.goal_stability
        goal_shifts = jax.random.bernoulli(
            jax.random.PRNGKey(state.step_count + 1), goal_shift_prob
        )
        
        if goal_shifts:
            # Find new random free position for goal
            free_positions = jnp.argwhere(state.grid == 0, size=self.height * self.width, fill_value=-1)
            free_positions = free_positions[free_positions[:, 0] >= 0]
            new_goal_idx = jax.random.randint(
                jax.random.PRNGKey(state.step_count + 2), (), 0, len(free_positions)
            )
            goal_pos = free_positions[new_goal_idx]
        else:
            goal_pos = state.goal_pos
        
        # Calculate reward
        reward = self.reward_step
        
        goal_reached = jnp.array_equal(agent_pos, goal_pos)
        reward = jnp.where(goal_reached, self.reward_goal, reward)
        reward = jnp.where(is_obstacle, self.reward_obstacle, reward)
        
        # Wind penalty if movement was blocked
        wind_penalty = jnp.where(~wind_effect, -0.05, 0.0)
        reward += wind_penalty
        
        # Update causal variables (they can evolve)
        new_wind = state.wind_strength + jax.random.normal(
            jax.random.PRNGKey(state.step_count + 3), ()
        ) * 0.01
        new_wind = jnp.clip(new_wind, *self.wind_range)
        
        new_visibility = state.visibility + jax.random.normal(
            jax.random.PRNGKey(state.step_count + 4), ()
        ) * 0.01
        new_visibility = jnp.clip(new_visibility, *self.visibility_range)
        
        new_goal_stability = state.goal_stability + jax.random.normal(
            jax.random.PRNGKey(state.step_count + 5), ()
        ) * 0.005
        new_goal_stability = jnp.clip(new_goal_stability, *self.goal_stability_range)
        
        # Apply ongoing interventions
        new_wind = jnp.where(state.active_interventions[0], state.intervention_values[0], new_wind)
        new_visibility = jnp.where(state.active_interventions[1], state.intervention_values[1], new_visibility)
        new_goal_stability = jnp.where(state.active_interventions[2], state.intervention_values[2], new_goal_stability)
        
        new_step_count = state.step_count + 1
        done = goal_reached | (new_step_count >= self.max_steps)
        
        new_state = CausalGridWorldState(
            agent_pos=agent_pos,
            goal_pos=goal_pos,
            grid=state.grid,
            step_count=new_step_count,
            done=done,
            wind_strength=new_wind,
            visibility=new_visibility,
            goal_stability=new_goal_stability,
            active_interventions=state.active_interventions,
            intervention_values=state.intervention_values
        )
        
        observation = self._get_causal_observation(new_state)
        
        info = {
            "goal_reached": goal_reached,
            "hit_obstacle": is_obstacle,
            "distance_to_goal": jnp.linalg.norm(agent_pos - goal_pos),
            "wind_blocked": ~wind_effect,
            "goal_shifted": goal_shifts,
            "active_interventions": state.active_interventions,
            "causal_variables": {
                "wind_strength": new_wind,
                "visibility": new_visibility,
                "goal_stability": new_goal_stability
            }
        }
        
        return new_state, observation, reward, done, info
    
    def _get_causal_observation(self, state: CausalGridWorldState) -> chex.Array:
        """Get observation including causal variables and visibility effects"""
        # Base observation
        grid_flat = state.grid.flatten()
        agent_pos_norm = state.agent_pos / jnp.array([self.height, self.width])
        goal_pos_norm = state.goal_pos / jnp.array([self.height, self.width])
        
        # Apply visibility effect to grid observation
        noise_scale = (1.0 - state.visibility) * 0.3
        grid_noise = jax.random.normal(
            jax.random.PRNGKey(state.step_count), grid_flat.shape
        ) * noise_scale
        noisy_grid = jnp.clip(grid_flat + grid_noise, 0.0, 1.0)
        
        # Add causal variables to observation
        causal_vars = jnp.array([
            state.wind_strength,
            state.visibility,
            state.goal_stability
        ])
        
        observation = jnp.concatenate([
            noisy_grid,
            agent_pos_norm,
            goal_pos_norm,
            causal_vars
        ])
        
        return observation
    
    def apply_intervention(
        self, 
        state: CausalGridWorldState, 
        variable: str, 
        value: float
    ) -> CausalGridWorldState:
        """Apply intervention do(variable = value)"""
        var_idx = self.causal_variables.index(variable)
        
        new_interventions = state.active_interventions.at[var_idx].set(True)
        new_values = state.intervention_values.at[var_idx].set(value)
        
        # Apply intervention immediately
        if variable == "wind_strength":
            new_wind = value
            new_visibility = state.visibility
            new_goal_stability = state.goal_stability
        elif variable == "visibility":
            new_wind = state.wind_strength
            new_visibility = value
            new_goal_stability = state.goal_stability
        elif variable == "goal_stability":
            new_wind = state.wind_strength
            new_visibility = state.visibility
            new_goal_stability = value
        else:
            raise ValueError(f"Unknown causal variable: {variable}")
        
        return state.replace(
            wind_strength=new_wind,
            visibility=new_visibility,
            goal_stability=new_goal_stability,
            active_interventions=new_interventions,
            intervention_values=new_values
        )
    
    def remove_intervention(
        self, 
        state: CausalGridWorldState, 
        variable: str
    ) -> CausalGridWorldState:
        """Remove intervention on a variable"""
        var_idx = self.causal_variables.index(variable)
        new_interventions = state.active_interventions.at[var_idx].set(False)
        
        return state.replace(active_interventions=new_interventions)
    
    def generate_causal_ood_config(
        self, 
        rng_key: chex.PRNGKey, 
        ood_type: str = "strong_wind"
    ) -> Dict[str, Any]:
        """Generate causal out-of-distribution scenarios"""
        if ood_type == "strong_wind":
            return {
                "enable_interventions": True,
                "intervention_probability": 0.8,
                "forced_interventions": [("wind_strength", 0.9)]
            }
        elif ood_type == "low_visibility":
            return {
                "enable_interventions": True,
                "intervention_probability": 0.8,
                "forced_interventions": [("visibility", 0.3)]
            }
        elif ood_type == "unstable_goal":
            return {
                "enable_interventions": True,
                "intervention_probability": 0.8,
                "forced_interventions": [("goal_stability", 0.7)]
            }
        elif ood_type == "multiple_interventions":
            return {
                "enable_interventions": True,
                "intervention_probability": 0.9,
                "forced_interventions": [
                    ("wind_strength", 0.8),
                    ("visibility", 0.4)
                ]
            }
        else:
            # Fall back to base class OOD types
            return super().generate_ood_config(rng_key, ood_type)
    
    def render_ascii(self, state: CausalGridWorldState) -> str:
        """Render state with causal information"""
        base_render = super().render_ascii(state)
        
        causal_info = f"""
Causal Variables:
  Wind Strength: {state.wind_strength:.2f}
  Visibility: {state.visibility:.2f}
  Goal Stability: {state.goal_stability:.2f}
  
Active Interventions: {state.active_interventions}
"""
        
        return base_render + causal_info