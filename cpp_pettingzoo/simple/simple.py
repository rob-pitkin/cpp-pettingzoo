"""PettingZoo-compatible wrapper for C++ Simple environment."""

import sys
from pathlib import Path
from typing import Optional
import functools

import gymnasium
from gymnasium.utils import EzPickle
from pettingzoo import ParallelEnv
import numpy as np
import pygame

# Add build directory to path
build_dir = Path(__file__).parent.parent.parent / "build"
sys.path.insert(0, str(build_dir))

import _simple_core
from cpp_pettingzoo.core import Agent, Landmark, World

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

        self.max_cycles = max_cycles
        self.render_mode = render_mode
        self.dynamic_rescaling = dynamic_rescaling
        self.continuous_actions = continuous_actions
        if render_mode is not None and render_mode not in ["human", "rgb_array"]:
            raise ValueError(f"Invalid render_mode: {render_mode}. Must be 'human' or 'rgb_array'")
        self.render_mode = render_mode

        # C++ environment
        self._cpp_env = _simple_core.SimpleEnv(max_cycles=self.max_cycles, dynamic_rescaling=self.dynamic_rescaling, continuous_actions=self.continuous_actions)

        # Agent setup
        self.possible_agents = ["agent_0"]
        self.agents = self._cpp_env.get_agents()

        # Action and observation spaces
        if (self.continuous_actions):
          self._action_spaces = {
              agent: gymnasium.spaces.Box(low=0, high=1, shape=(5,), dtype="float32") for agent in self.agents
          }
        else:
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

        if self.render_mode is not None:
            pygame.init()
            self.width = 700
            self.height = 700
            self.screen = pygame.Surface([self.width, self.height])
            self.game_font = pygame.font.Font(None, 24)
            self.viewer = None
            self.renderOn = False

            self.world = World()
            self.world.agents = [Agent("agent_0", silent=True)]
            self.world.landmarks = [Landmark("landmark_0")]

            self.world.agents[0].color = np.array([0.25, 0.25, 0.25])
            self.world.landmarks[0].color = np.array([0.75, 0.25, 0.25])

            self.original_cam_range = 1.0

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
        observations, infos = self._cpp_env.reset(seed=seed)

        # Convert observations to numpy arrays for MPE2 API compatibility
        observations = {agent: np.array(obs, dtype=np.float32) for agent, obs in observations.items()}

        self.agents = self._cpp_env.get_agents()

        if self.render_mode is not None:
            render_state = self._cpp_env.get_render_state()

            self.world.agents[0].state.p_pos = np.array(render_state["agent_0_pos"])
            self.world.agents[0].state.p_vel = np.array(render_state["agent_0_vel"])
            self.world.agents[0].color = np.array(render_state["agent_0_color"])
            self.world.landmarks[0].state.p_pos = np.array(render_state["landmark_0_pos"])
            self.world.landmarks[0].state.p_vel = np.zeros(2)

            all_poses = [entity.state.p_pos for entity in self.world.entities]
            self.original_cam_range = np.max(np.abs(np.array(all_poses)))

        return observations, infos

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
        try:
            # Convert actions to an array of floats if continuous actions is false
            if not self.continuous_actions:
                actions = {agent: [float(action)] for agent, action in actions.items()}
            observations, rewards, terminations, truncations, infos = self._cpp_env.step(actions)
        except RuntimeError as e:
            if "reset() must be called before step()" in str(e):
                raise AttributeError("agents cannot be accessed before reset")
            raise

        # Convert observations to numpy arrays for MPE2 API compatibility
        observations = {agent: np.array(obs, dtype=np.float32) for agent, obs in observations.items()}

        # Get updated agents list from C++ (will be empty if episode done)
        self.agents = self._cpp_env.get_agents()

        if self.render_mode is not None:
            render_state = self._cpp_env.get_render_state()

            self.world.agents[0].state.p_pos = np.array(render_state["agent_0_pos"])
            self.world.agents[0].state.p_vel = np.array(render_state["agent_0_vel"])
            self.world.agents[0].color = np.array(render_state["agent_0_color"])
            self.world.landmarks[0].state.p_pos = np.array(render_state["landmark_0_pos"])
            self.world.landmarks[0].state.p_vel = np.zeros(2)

        return observations, rewards, terminations, truncations, infos

    def enable_render(self, mode="human"):
        """Enable rendering by creating the pygame display window."""
        if not self.renderOn and mode == "human":
            self.screen = pygame.display.set_mode(self.screen.get_size())
            self.clock = pygame.time.Clock()
            self.renderOn = True

    def render(self):
        """Render the environment."""
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        self.enable_render(self.render_mode)

        self.draw()
        if self.render_mode == "rgb_array":
            observation = np.array(pygame.surfarray.pixels3d(self.screen))
            return np.transpose(observation, axes=(1, 0, 2))
        elif self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            return

    def draw(self):
        """Draw the current state of the environment."""
        # clear screen
        self.screen.fill((255, 255, 255))

        # update bounds to center around agent
        all_poses = [entity.state.p_pos for entity in self.world.entities]
        cam_range = np.max(np.abs(np.array(all_poses)))

        # The scaling factor is used for dynamic rescaling of the rendering - a.k.a Zoom In/Zoom Out effect
        # The 0.9 is a factor to keep the entities from appearing "too" out-of-bounds
        scaling_factor = 0.9 * self.original_cam_range / cam_range

        # update geometry and text positions
        for e, entity in enumerate(self.world.entities):
            # geometry
            x, y = entity.state.p_pos
            y *= (
                -1
            )  # this makes the display mimic the old pyglet setup (ie. flips image)
            x = (
                (x / cam_range) * self.width // 2 * 0.9
            )  # the .9 is just to keep entities from appearing "too" out-of-bounds
            y = (y / cam_range) * self.height // 2 * 0.9
            x += self.width // 2
            y += self.height // 2

            # 350 is an arbitrary scale factor to get pygame to render similar sizes as pyglet
            if self.dynamic_rescaling:
                radius = entity.size * 350 * scaling_factor
            else:
                radius = entity.size * 350

            pygame.draw.circle(self.screen, entity.color * 200, (x, y), radius)
            pygame.draw.circle(self.screen, (0, 0, 0), (x, y), radius, 1)  # borders
            assert (
                0 < x < self.width and 0 < y < self.height
            ), f"Coordinates {(x, y)} are out of bounds."

    def close(self):
        """Close the environment and clean up pygame."""
        if hasattr(self, 'screen') and self.screen is not None:
            pygame.quit()
            self.screen = None

    def state(self):
        """Return the global state (same as observation for single agent)."""
        if self._cpp_env:
            try:
                return np.array(self._cpp_env.get_state(), dtype=np.float32)
            except RuntimeError as e:
                raise AssertionError(str(e))



# Expose parallel API
parallel_env = raw_env

# Provide AEC wrapper for compatibility with MPE2
def env(**kwargs):
    from pettingzoo.utils.conversions import parallel_to_aec_wrapper
    from pettingzoo.utils import wrappers

    parallel = parallel_env(**kwargs)
    aec = parallel_to_aec_wrapper(parallel)
    # Add the same wrappers MPE2 uses
    if parallel.continuous_actions:
        aec = wrappers.ClipOutOfBoundsWrapper(aec)
    else:
        aec = wrappers.AssertOutOfBoundsWrapper(aec)
    aec = wrappers.OrderEnforcingWrapper(aec)
    return aec
