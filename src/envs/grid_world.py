"""
Simple Grid World Environment for testing AGI principles

A minimal environment where an agent navigates a grid to reach a goal,
with configurable obstacles and dynamics. Perfect for testing OOD generalization.
"""

from typing import Tuple, Dict, Any, Optional
import jax
import jax.numpy as jnp
import chex
from flax import struct


@struct.dataclass
class GridWorldState:
    """State of the grid world environment"""
    agent_pos: chex.Array  # (2,) array with [x, y] position
    goal_pos: chex.Array   # (2,) array with [x, y] position
    grid: chex.Array       # (height, width) array with obstacles (1) and free space (0)
    step_count: int
    done: bool


class GridWorld:
    """
    Simple grid world navigation environment
    
    The agent (A) must navigate to the goal (G) while avoiding obstacles (X):
    
    . . . X . . G
    . X . X . . .
    . X . . . X .
    A . . X . . .
    
    Actions: 0=Up, 1=Right, 2=Down, 3=Left
    """
    
    def __init__(
        self,
        height: int = 8,
        width: int = 8,
        obstacle_prob: float = 0.2,
        max_steps: int = 100,
        reward_goal: float = 10.0,
        reward_step: float = -0.1,
        reward_obstacle: float = -1.0
    ):
        self.height = height
        self.width = width
        self.obstacle_prob = obstacle_prob
        self.max_steps = max_steps
        self.reward_goal = reward_goal
        self.reward_step = reward_step
        self.reward_obstacle = reward_obstacle
        
        # Action mappings
        self.actions = {
            0: jnp.array([-1, 0]),  # Up
            1: jnp.array([0, 1]),   # Right
            2: jnp.array([1, 0]),   # Down
            3: jnp.array([0, -1])   # Left
        }
        
        self.action_dim = 4
        self.obs_dim = height * width + 4  # Grid flattened + agent_pos + goal_pos
    
    def reset(self, rng_key: chex.PRNGKey) -> Tuple[GridWorldState, chex.Array]:
        """Reset environment to initial state"""
        key1, key2, key3 = jax.random.split(rng_key, 3)
        
        # Generate random grid with obstacles
        grid = jax.random.bernoulli(key1, self.obstacle_prob, (self.height, self.width))
        
        # Place agent at random free position
        free_positions = jnp.argwhere(grid == 0, size=self.height * self.width, fill_value=-1)
        free_positions = free_positions[free_positions[:, 0] >= 0]  # Remove fill values
        
        agent_idx = jax.random.randint(key2, (), 0, len(free_positions))
        agent_pos = free_positions[agent_idx]
        
        # Place goal at different random free position
        remaining_positions = free_positions[jnp.arange(len(free_positions)) != agent_idx]
        goal_idx = jax.random.randint(key3, (), 0, len(remaining_positions))
        goal_pos = remaining_positions[goal_idx]
        
        state = GridWorldState(
            agent_pos=agent_pos,
            goal_pos=goal_pos,
            grid=grid,
            step_count=0,
            done=False
        )
        
        observation = self._get_observation(state)
        return state, observation
    
    def step(
        self, 
        state: GridWorldState, 
        action: int
    ) -> Tuple[GridWorldState, chex.Array, float, bool, Dict[str, Any]]:
        """Take one step in the environment"""
        
        # Calculate new position
        action_delta = self.actions[action]
        new_pos = state.agent_pos + action_delta
        
        # Check bounds
        new_pos = jnp.clip(new_pos, 0, jnp.array([self.height - 1, self.width - 1]))
        
        # Check if new position is obstacle
        is_obstacle = state.grid[new_pos[0], new_pos[1]]
        
        # Update position only if not obstacle
        agent_pos = jnp.where(is_obstacle, state.agent_pos, new_pos)
        
        # Calculate reward
        reward = self.reward_step  # Base step penalty
        
        # Goal reached
        goal_reached = jnp.array_equal(agent_pos, state.goal_pos)
        reward = jnp.where(goal_reached, self.reward_goal, reward)
        
        # Hit obstacle
        reward = jnp.where(is_obstacle, self.reward_obstacle, reward)
        
        # Update state
        new_step_count = state.step_count + 1
        done = goal_reached | (new_step_count >= self.max_steps)
        
        new_state = GridWorldState(
            agent_pos=agent_pos,
            goal_pos=state.goal_pos,
            grid=state.grid,
            step_count=new_step_count,
            done=done
        )
        
        observation = self._get_observation(new_state)
        
        info = {
            "goal_reached": goal_reached,
            "hit_obstacle": is_obstacle,
            "distance_to_goal": jnp.linalg.norm(agent_pos - state.goal_pos)
        }
        
        return new_state, observation, reward, done, info
    
    def _get_observation(self, state: GridWorldState) -> chex.Array:
        """Convert state to observation vector"""
        # Flatten grid and concatenate with positions
        grid_flat = state.grid.flatten()
        agent_pos_norm = state.agent_pos / jnp.array([self.height, self.width])
        goal_pos_norm = state.goal_pos / jnp.array([self.height, self.width])
        
        observation = jnp.concatenate([
            grid_flat,
            agent_pos_norm,
            goal_pos_norm
        ])
        
        return observation
    
    def render_ascii(self, state: GridWorldState) -> str:
        """Render current state as ASCII art"""
        grid_str = ""
        for i in range(self.height):
            row = ""
            for j in range(self.width):
                if jnp.array_equal(jnp.array([i, j]), state.agent_pos):
                    row += "A "
                elif jnp.array_equal(jnp.array([i, j]), state.goal_pos):
                    row += "G "
                elif state.grid[i, j]:
                    row += "X "
                else:
                    row += ". "
            grid_str += row + "\n"
        
        return grid_str
    
    def generate_ood_config(self, rng_key: chex.PRNGKey, ood_type: str = "obstacles") -> Dict[str, Any]:
        """Generate out-of-distribution configuration for testing"""
        if ood_type == "obstacles":
            # Increase obstacle density
            return {"obstacle_prob": self.obstacle_prob * 2}
        elif ood_type == "size":
            # Change grid size
            return {"height": self.height + 2, "width": self.width + 2}
        elif ood_type == "rewards":
            # Change reward structure
            return {
                "reward_goal": self.reward_goal * 0.5,
                "reward_step": self.reward_step * 2
            }
        else:
            raise ValueError(f"Unknown OOD type: {ood_type}")


# Utility functions for batch processing
def reset_batch(env: GridWorld, rng_key: chex.PRNGKey, batch_size: int):
    """Reset multiple environments in parallel"""
    keys = jax.random.split(rng_key, batch_size)
    reset_fn = jax.vmap(env.reset)
    return reset_fn(keys)


def step_batch(env: GridWorld, states, actions):
    """Step multiple environments in parallel"""
    step_fn = jax.vmap(env.step, in_axes=(0, 0))
    return step_fn(states, actions)