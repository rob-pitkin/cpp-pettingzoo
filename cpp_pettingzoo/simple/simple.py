"""PettingZoo-compatible wrapper for C++ Simple environment."""

import sys
from pathlib import Path
from typing import Optional
import functools

import gymnasium
from gymnasium.utils import EzPickle
from pettingzoo import ParallelEnv

# Add build directory to path
build_dir = Path(__file__).parent.parent.parent / "build"
sys.path.insert(0, str(build_dir))

import _simple_core


class raw_env(ParallelEnv, EzPickle):
    """PettingZoo ParallelEnv wrapper for C++ Simple environment.

    This is a thin wrapper that provides PettingZoo API compatibility.
    All physics computation happens in C++.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "simple_v3",
        "is_parallelizable": True,
        "render_fps": 10,
    }

    def __init__(
        self,
        max_cycles: int = 25,
        continuous_actions: bool = False,
        render_mode: Optional[str] = None,
        dynamic_rescaling: bool = False,
    ):
        """Initialize Simple environment.

        Args:
            max_cycles: Maximum number of timesteps per episode
            continuous_actions: If True, use continuous action space (NOT IMPLEMENTED)
            render_mode: Rendering mode (NOT IMPLEMENTED)
            dynamic_rescaling: Dynamic rescaling (NOT IMPLEMENTED)
        """
        EzPickle.__init__(
            self,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
            dynamic_rescaling=dynamic_rescaling,
        )

        if continuous_actions:
            raise NotImplementedError("Continuous actions not yet implemented in C++ version")
        if render_mode is not None:
            raise NotImplementedError("Rendering not yet implemented in C++ version")
        if dynamic_rescaling:
            raise NotImplementedError("Dynamic rescaling not yet implemented in C++ version")

        self.max_cycles = max_cycles
        self.render_mode = render_mode

        # C++ environment
        self._cpp_env = _simple_core.SimpleEnv(max_cycles=self.max_cycles)

        # Agent setup
        self.possible_agents = ["agent_0"]
        self.agents = self._cpp_env.get_agents()

        # Action and observation spaces
        self._action_spaces = {
            agent: gymnasium.spaces.Discrete(5) for agent in self.agents
        }
        self._observation_spaces = {
            agent: gymnasium.spaces.Box(
                low=-float("inf"),
                high=float("inf"),
                shape=(4,),
                dtype="float32",
            )
            for agent in self.agents
        }

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        """Return observation space for agent."""
        return self._observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        """Return action space for agent."""
        return self._action_spaces[agent]

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment.

        Args:
            seed: Random seed
            options: Additional options (ignored)

        Returns:
            observations: Dict of observations for each agent
            infos: Dict of info dicts for each agent
        """
        result = self._cpp_env.reset(seed=seed)

        self.agents = self._cpp_env.get_agents()

        return result

    def step(self, actions):
        """Step the environment.

        Args:
            actions: Dict mapping agent names to actions

        Returns:
            observations: Dict of observations
            rewards: Dict of rewards
            terminations: Dict of termination flags
            truncations: Dict of truncation flags
            infos: Dict of info dicts
        """
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        # Forward to C++ (already returns correct format)
        observations, rewards, terminations, truncations, infos = self._cpp_env.step(actions)

        # Get updated agents list from C++ (will be empty if episode done)
        self.agents = self._cpp_env.get_agents()

        return observations, rewards, terminations, truncations, infos

    def render(self):
        """Render the environment (not implemented)."""
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return
        raise NotImplementedError("Rendering not yet implemented in C++ version")

    def close(self):
        """Close the environment."""
        self._cpp_env = None

    def state(self):
        """Return the global state (same as observation for single agent)."""
        if self._cpp_env is None:
            return None
        # For single agent, state is just the observation
        # This is a bit of a hack - we'd need to expose state from C++ properly
        return None


# Expose as both raw_env and parallel_env (they're the same since C++ is already parallel)
parallel_env = raw_env

# Note: We don't provide 'env' (the AEC wrapper) since our C++ implementation
# is natively parallel. Users should use parallel_env directly.
