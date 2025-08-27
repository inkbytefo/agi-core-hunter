"""
Active Inference Grid World Environment

This environment is specifically designed to test the Free Energy Principle
and active inference. It includes:
- Uncertainty regions with noisy observations
- Hidden rewards that require exploration to discover
- Dynamic goal locations that test adaptation
- Surprise-inducing environmental changes
"""

from typing import Tuple, Dict, Any, Optional
import jax
import jax.numpy as jnp
import chex
from flax import struct
import numpy as np


@struct.dataclass
class ActiveInferenceState:
    """State of the active inference environment"""
    agent_pos: chex.Array  # (2,) array with [x, y] position
    goal_pos: chex.Array   # (2,) array with [x, y] position
    grid: chex.Array       # (height, width) array with terrain types
    uncertainty_map: chex.Array  # (height, width) array with uncertainty levels
    hidden_rewards: chex.Array   # (height, width) array with hidden reward locations
    step_count: int
    done: bool
    surprise_events: chex.Array  # Track recent surprise events
    goal_discovered: bool        # Whether agent has discovered the goal
    exploration_bonus: float     # Current exploration bonus


class ActiveInferenceWorld:
    """
    Environment designed to test Active Inference and Free Energy Principle
    
    Key features for testing FEP:
    1. Uncertainty regions: Areas with noisy/partial observations
    2. Hidden information: Rewards and goals not immediately visible
    3. Surprise events: Unexpected environmental changes
    4. Exploration incentives: Information gain opportunities
    5. Dynamic goals: Goals that move to test adaptation
    
    Terrain types:
    0 = Free space (certain observations)
    1 = Obstacle (impassable)
    2 = Uncertainty zone (noisy observations)
    3 = Hidden reward location
    4 = Goal location
    """
    
    def __init__(
        self,
        height: int = 10,
        width: int = 10,
        uncertainty_prob: float = 0.3,
        hidden_reward_prob: float = 0.1,
        observation_noise_std: float = 0.2,
        max_steps: int = 150,
        reward_goal: float = 20.0,
        reward_hidden: float = 5.0,
        reward_step: float = -0.1,
        reward_exploration: float = 1.0,
        surprise_threshold: float = 2.0,
        goal_move_prob: float = 0.02,  # Probability goal moves each step
        fep_strength: float = 1.0      # Strength of FEP-specific dynamics
    ):
        self.height = height
        self.width = width
        self.uncertainty_prob = uncertainty_prob
        self.hidden_reward_prob = hidden_reward_prob
        self.observation_noise_std = observation_noise_std
        self.max_steps = max_steps
        self.reward_goal = reward_goal
        self.reward_hidden = reward_hidden
        self.reward_step = reward_step
        self.reward_exploration = reward_exploration
        self.surprise_threshold = surprise_threshold
        self.goal_move_prob = goal_move_prob
        self.fep_strength = fep_strength
        
        # Action mappings
        self.actions = {
            0: jnp.array([-1, 0]),  # Up
            1: jnp.array([0, 1]),   # Right
            2: jnp.array([1, 0]),   # Down
            3: jnp.array([0, -1])   # Left
        }
        
        self.action_dim = 4
        # Observation includes: grid state (noisy), uncertainty map, exploration map, positions, surprise features
        # grid_obs_noisy: height * width
        # uncertainty_obs: height * width  
        # exploration_obs: height * width
        # agent_pos_norm: 2
        # goal_pos_norm: 2
        # surprise_features: 4
        self.obs_dim = height * width * 3 + 2 + 2 + 4  # terrain + uncertainty + exploration + positions + features
        
        # Track visited locations for exploration bonus
        self.visited_map = jnp.zeros((height, width))
    
    def reset(self, rng_key: chex.PRNGKey) -> Tuple[ActiveInferenceState, chex.Array]:
        """Reset environment with uncertainty and hidden elements"""
        key1, key2, key3, key4, key5 = jax.random.split(rng_key, 5)
        
        # Generate base grid (0 = free, 1 = obstacle)
        obstacle_prob = 0.15  # Lower obstacle density to allow exploration
        grid = jax.random.bernoulli(key1, obstacle_prob, (self.height, self.width))
        
        # Add uncertainty zones (type 2)
        uncertainty_mask = jax.random.bernoulli(key2, self.uncertainty_prob, (self.height, self.width))
        grid = jnp.where((grid == 0) & uncertainty_mask, 2, grid)
        
        # Generate uncertainty map (how noisy observations are in each location)
        uncertainty_map = jax.random.uniform(key3, (self.height, self.width), minval=0.0, maxval=1.0)
        uncertainty_map = jnp.where(grid == 2, uncertainty_map * 0.8 + 0.2, uncertainty_map * 0.1)
        
        # Place hidden rewards (type 3)
        free_positions = jnp.argwhere(grid == 0, size=self.height * self.width, fill_value=-1)
        free_positions = free_positions[free_positions[:, 0] >= 0]
        
        # Select subset for hidden rewards
        n_hidden = max(1, int(len(free_positions) * self.hidden_reward_prob))
        hidden_indices = jax.random.choice(key4, len(free_positions), (n_hidden,), replace=False)
        
        hidden_rewards = jnp.zeros((self.height, self.width))
        for idx in hidden_indices:
            pos = free_positions[idx]
            hidden_rewards = hidden_rewards.at[pos[0], pos[1]].set(1.0)
            grid = grid.at[pos[0], pos[1]].set(3)  # Mark as hidden reward
        
        # Place agent at random free position
        available_positions = jnp.argwhere(grid == 0, size=self.height * self.width, fill_value=-1)
        available_positions = available_positions[available_positions[:, 0] >= 0]
        
        agent_idx = jax.random.randint(key5, (), 0, len(available_positions))
        agent_pos = available_positions[agent_idx]
        
        # Place goal at different position (initially hidden from agent)
        remaining_positions = available_positions[jnp.arange(len(available_positions)) != agent_idx]
        goal_idx = jax.random.randint(key5, (), 0, len(remaining_positions))
        goal_pos = remaining_positions[goal_idx]
        
        # Initialize state
        state = ActiveInferenceState(
            agent_pos=agent_pos,
            goal_pos=goal_pos,
            grid=grid,
            uncertainty_map=uncertainty_map,
            hidden_rewards=hidden_rewards,
            step_count=0,
            done=False,
            surprise_events=jnp.zeros(10),  # Track last 10 events
            goal_discovered=False,
            exploration_bonus=0.0
        )
        
        # Reset visited map
        self.visited_map = jnp.zeros((self.height, self.width))
        self.visited_map = self.visited_map.at[agent_pos[0], agent_pos[1]].set(1.0)
        
        observation = self._get_observation(state, key5)
        return state, observation
    
    def step(
        self, 
        state: ActiveInferenceState, 
        action: int,
        rng_key: chex.PRNGKey
    ) -> Tuple[ActiveInferenceState, chex.Array, float, bool, Dict[str, Any]]:
        """Take one step with FEP-specific dynamics"""
        key1, key2, key3 = jax.random.split(rng_key, 3)
        
        # Calculate new position
        action_delta = self.actions[action]
        new_pos = state.agent_pos + action_delta
        
        # Check bounds
        new_pos = jnp.clip(new_pos, 0, jnp.array([self.height - 1, self.width - 1]))
        
        # Check terrain at new position
        terrain_type = state.grid[new_pos[0], new_pos[1]]
        
        # Movement rules based on terrain
        can_move = terrain_type != 1  # Can't move into obstacles
        agent_pos = jnp.where(can_move, new_pos, state.agent_pos)
        
        # Update visited map for exploration tracking
        was_visited = self.visited_map[agent_pos[0], agent_pos[1]]
        self.visited_map = self.visited_map.at[agent_pos[0], agent_pos[1]].set(1.0)
        
        # Calculate exploration bonus (information gain)
        exploration_bonus = jnp.where(was_visited == 0, self.reward_exploration, 0.0)
        
        # Base reward
        reward = self.reward_step
        
        # Goal discovery and reaching
        goal_distance = jnp.linalg.norm(agent_pos - state.goal_pos)
        goal_reached = goal_distance < 1.0
        
        # Discover goal if close enough (active inference: reduce uncertainty about goal location)
        goal_discovered = state.goal_discovered | (goal_distance < 3.0)
        
        if goal_reached:
            reward += self.reward_goal
        
        # Hidden reward collection
        hidden_reward_collected = state.hidden_rewards[agent_pos[0], agent_pos[1]]
        if hidden_reward_collected > 0:
            reward += self.reward_hidden
            # Remove collected reward
            hidden_rewards = state.hidden_rewards.at[agent_pos[0], agent_pos[1]].set(0.0)
        else:
            hidden_rewards = state.hidden_rewards
        
        # Add exploration bonus
        reward += exploration_bonus * self.fep_strength
        
        # Calculate surprise: unexpected changes in environment
        surprise_level = 0.0
        
        # Goal movement (surprising event)
        if jax.random.uniform(key1) < self.goal_move_prob:
            # Move goal to nearby location
            goal_offset = jax.random.choice(key2, 9, shape=()) - 4  # -4 to 4 offset
            goal_offset_2d = jnp.array([goal_offset // 3 - 1, goal_offset % 3 - 1])
            new_goal_pos = jnp.clip(
                state.goal_pos + goal_offset_2d, 
                0, 
                jnp.array([self.height - 1, self.width - 1])
            )
            
            # Ensure new goal position is valid (not obstacle)
            if state.grid[new_goal_pos[0], new_goal_pos[1]] != 1:
                goal_pos = new_goal_pos
                surprise_level = 3.0  # High surprise from goal movement
                goal_discovered = False  # Need to rediscover moved goal
            else:
                goal_pos = state.goal_pos
        else:
            goal_pos = state.goal_pos
        
        # Environmental surprise based on uncertainty
        current_uncertainty = state.uncertainty_map[agent_pos[0], agent_pos[1]]
        if current_uncertainty > 0.5:
            # High uncertainty location - add random surprise
            surprise_level += jax.random.exponential(key3) * current_uncertainty
        
        # Update surprise events buffer
        surprise_events = jnp.roll(state.surprise_events, 1)
        surprise_events = surprise_events.at[0].set(surprise_level)
        
        # Apply surprise penalty (FEP: minimize surprise)
        if surprise_level > self.surprise_threshold:
            reward -= surprise_level * 0.5 * self.fep_strength
        
        # Update state
        new_step_count = state.step_count + 1
        done = goal_reached | (new_step_count >= self.max_steps)
        
        new_state = ActiveInferenceState(
            agent_pos=agent_pos,
            goal_pos=goal_pos,
            grid=state.grid,
            uncertainty_map=state.uncertainty_map,
            hidden_rewards=hidden_rewards,
            step_count=new_step_count,
            done=done,
            surprise_events=surprise_events,
            goal_discovered=goal_discovered,
            exploration_bonus=exploration_bonus
        )
        
        observation = self._get_observation(new_state, key3)
        
        info = {
            "goal_reached": goal_reached,
            "goal_discovered": goal_discovered,
            "hidden_reward_collected": hidden_reward_collected,
            "exploration_bonus": exploration_bonus,
            "surprise_level": surprise_level,
            "uncertainty_level": current_uncertainty,
            "distance_to_goal": goal_distance,
            "total_explored": jnp.sum(self.visited_map),
            "exploration_ratio": jnp.sum(self.visited_map) / (self.height * self.width)
        }
        
        return new_state, observation, reward, done, info
    
    def _get_observation(self, state: ActiveInferenceState, rng_key: chex.PRNGKey) -> chex.Array:
        """
        Get observation with uncertainty and noise
        
        FEP-specific observation features:
        - Noisy observations in uncertainty zones
        - Partial observability of goals until discovered
        - Exploration map showing visited areas
        """
        key1, key2 = jax.random.split(rng_key)
        
        # Base grid observation with terrain types
        grid_obs = state.grid.flatten()
        
        # Add observation noise based on uncertainty map
        noise = jax.random.normal(key1, grid_obs.shape) * self.observation_noise_std
        uncertainty_noise = state.uncertainty_map.flatten() * noise
        grid_obs_noisy = grid_obs + uncertainty_noise
        
        # Agent and goal positions (normalized)
        agent_pos_norm = state.agent_pos / jnp.array([self.height, self.width])
        
        # Goal position only observable if discovered or nearby
        goal_distance = jnp.linalg.norm(state.agent_pos - state.goal_pos)
        goal_visible = state.goal_discovered | (goal_distance < 2.0)
        
        goal_pos_norm = jnp.where(
            goal_visible,
            state.goal_pos / jnp.array([self.height, self.width]),
            jnp.array([0.5, 0.5])  # Default uncertain position
        )
        
        # Uncertainty map (flattened)
        uncertainty_obs = state.uncertainty_map.flatten()
        
        # Exploration map (visited locations)
        exploration_obs = self.visited_map.flatten()
        
        # Additional FEP-specific features
        surprise_features = jnp.array([
            jnp.mean(state.surprise_events),  # Recent surprise level
            state.exploration_bonus,          # Current exploration bonus
            float(state.goal_discovered),     # Goal discovery status
            goal_distance / (self.height + self.width)  # Normalized goal distance
        ])
        
        # Combine all observation components
        observation = jnp.concatenate([
            grid_obs_noisy,      # Noisy terrain observations
            uncertainty_obs,     # Uncertainty levels
            exploration_obs,     # Exploration map
            agent_pos_norm,      # Agent position
            goal_pos_norm,       # Goal position (if known)
            surprise_features    # FEP-specific features
        ])
        
        return observation
    
    def render_ascii(self, state: ActiveInferenceState) -> str:
        """Render current state as ASCII with FEP-specific information"""
        grid_str = "Active Inference World:\n"
        
        for i in range(self.height):
            row = ""
            for j in range(self.width):
                pos = jnp.array([i, j])
                
                if jnp.array_equal(pos, state.agent_pos):
                    row += "A "
                elif jnp.array_equal(pos, state.goal_pos):
                    if state.goal_discovered:
                        row += "G "
                    else:
                        row += "? "  # Hidden goal
                elif state.grid[i, j] == 1:
                    row += "X "  # Obstacle
                elif state.grid[i, j] == 2:
                    row += "~ "  # Uncertainty zone
                elif state.grid[i, j] == 3:
                    row += "H "  # Hidden reward
                elif self.visited_map[i, j] > 0:
                    row += ". "  # Visited
                else:
                    row += "  "  # Unvisited
            
            grid_str += row + "\n"
        
        # Add FEP-specific information
        grid_str += f"\nStep: {state.step_count}\n"
        grid_str += f"Goal discovered: {state.goal_discovered}\n"
        grid_str += f"Recent surprise: {jnp.mean(state.surprise_events):.2f}\n"
        grid_str += f"Exploration bonus: {state.exploration_bonus:.2f}\n"
        grid_str += f"Explored: {jnp.sum(self.visited_map)}/{self.height * self.width}\n"
        
        return grid_str
    
    def generate_fep_config(self, rng_key: chex.PRNGKey, test_type: str = "uncertainty") -> Dict[str, Any]:
        """Generate configurations for testing specific FEP aspects"""
        if test_type == "uncertainty":
            # High uncertainty environment
            return {
                "uncertainty_prob": 0.6,
                "observation_noise_std": 0.4,
                "fep_strength": 2.0
            }
        elif test_type == "exploration":
            # Exploration-heavy environment
            return {
                "hidden_reward_prob": 0.2,
                "reward_exploration": 2.0,
                "goal_move_prob": 0.0  # Stable goal for pure exploration
            }
        elif test_type == "surprise":
            # High surprise environment
            return {
                "goal_move_prob": 0.1,
                "surprise_threshold": 1.0,
                "fep_strength": 3.0
            }
        elif test_type == "adaptation":
            # Dynamic environment requiring adaptation
            return {
                "goal_move_prob": 0.05,
                "uncertainty_prob": 0.4,
                "hidden_reward_prob": 0.15
            }
        else:
            raise ValueError(f"Unknown FEP test type: {test_type}")
    
    def get_epistemic_value_map(self, state: ActiveInferenceState) -> chex.Array:
        """Calculate epistemic value (information gain potential) for each location"""
        # Epistemic value based on:
        # 1. Uncertainty level (high uncertainty = high info gain potential)
        # 2. Unexplored areas (unvisited = potential information)
        # 3. Distance from known areas (further = more uncertain)
        
        uncertainty_value = state.uncertainty_map
        exploration_value = 1.0 - self.visited_map
        
        # Combine values
        epistemic_map = uncertainty_value * 0.6 + exploration_value * 0.4
        
        return epistemic_map
    
    def get_pragmatic_value_map(self, state: ActiveInferenceState) -> chex.Array:
        """Calculate pragmatic value (goal achievement potential) for each location"""
        # Pragmatic value based on:
        # 1. Distance to goal (if discovered)
        # 2. Hidden reward locations
        # 3. Progress toward exploration completion
        
        pragmatic_map = jnp.zeros((self.height, self.width))
        
        # Goal attraction (if discovered)
        if state.goal_discovered:
            for i in range(self.height):
                for j in range(self.width):
                    distance = jnp.linalg.norm(jnp.array([i, j]) - state.goal_pos)
                    pragmatic_map = pragmatic_map.at[i, j].set(
                        jnp.exp(-distance / 3.0)  # Exponential decay with distance
                    )
        
        # Add hidden reward values
        pragmatic_map += state.hidden_rewards * 0.5
        
        return pragmatic_map