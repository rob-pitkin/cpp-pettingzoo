"""SimpleAdversary environment with C++ backend."""

import sys
from pathlib import Path

import gymnasium
import numpy as np
import pygame
from gymnasium.utils import EzPickle
from pettingzoo import ParallelEnv
from pettingzoo.utils.conversions import parallel_to_aec_wrapper

# Add build directory to path
build_dir = Path(__file__).parent.parent.parent / "build"
sys.path.insert(0, str(build_dir))

import _simple_adversary
from cpp_mpe2._wrappers import check_and_maybe_clip_actions, make_aec_env
from cpp_mpe2.core import Agent, Landmark, World


class parallel_env(ParallelEnv, EzPickle):
    """SimpleAdversary parallel environment.

    1 adversary (red) tries to reach the goal landmark (green).
    N good agents (blue) try to keep the adversary away from the goal.
    The adversary doesn't know which landmark is the goal.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "simple_adversary_v3",
        "is_parallelizable": True,
        "render_fps": 10,
    }

    def __init__(
        self,
        N=2,
        max_cycles=25,
        continuous_actions=False,
        render_mode=None,
        dynamic_rescaling=False,
         benchmark_data=False,
    ):
        EzPickle.__init__(
            self,
            N=N,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
            dynamic_rescaling=dynamic_rescaling,
             benchmark_data=benchmark_data,
        )

        self.N = N
        self.max_cycles = max_cycles
        self.continuous_actions = continuous_actions
        self.render_mode = render_mode
        self.dynamic_rescaling = dynamic_rescaling

        # Create C++ environment
        self._cpp_env = _simple_adversary.SimpleAdversaryEnv(
            N=N,
            max_cycles=max_cycles,
            dynamic_rescaling=dynamic_rescaling,
            continuous_actions=continuous_actions,
        )

        # Agent names: adversary_0, agent_0, agent_1, ...
        self.possible_agents = ["adversary_0"] + [f"agent_{i}" for i in range(N)]
        self.agents = self.possible_agents[:]

        # Action spaces: all agents can only move (no communication)
        if self.continuous_actions:
            self._action_spaces = {
                agent: gymnasium.spaces.Box(low=0.0, high=1.0, shape=(5,), dtype="float32")
                for agent in self.possible_agents
            }
        else:
            self._action_spaces = {
                agent: gymnasium.spaces.Discrete(5) for agent in self.possible_agents
            }

        # Observation spaces: asymmetric
        # Adversary: landmarks + other agents = 2*N + 2*N = 4*N
        # Good agents: goal + landmarks + other agents = 2 + 2*N + 2*N = 2 + 4*N
        self._observation_spaces = {
            "adversary_0": gymnasium.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(4 * N,),
                dtype="float32",
            )
        }
        for i in range(N):
            self._observation_spaces[f"agent_{i}"] = gymnasium.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(2 + 4 * N,),
                dtype="float32",
            )

        # Initialize rendering
        if self.render_mode is not None:
            pygame.init()
            self.width = 700
            self.height = 700
            self.screen = pygame.Surface([self.width, self.height])
            self.game_font = pygame.font.Font(None, 24)
            self.viewer = None
            self.renderOn = False

            self.world = World()
            # Create agents (adversary + good agents)
            self.world.agents = [Agent(agent_name, silent=True) for agent_name in self.possible_agents]
            # Create landmarks
            self.world.landmarks = [Landmark(f"landmark {i}") for i in range(N)]

            # Set entity sizes to match C++ implementation
            for agent in self.world.agents:
                agent.size = 0.15  # SimpleAdversary uses 0.15 for agents

            for landmark in self.world.landmarks:
                landmark.size = 0.08  # SimpleAdversary uses 0.08 for landmarks

            # Set default colors (will be updated in reset)
            for agent in self.world.agents:
                agent.color = np.array([0.35, 0.35, 0.85])  # Default blue

            for landmark in self.world.landmarks:
                landmark.color = np.array([0.25, 0.25, 0.25])  # Default gray

            self.original_cam_range = 1.0

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        if seed is not None:
            observations, infos = self._cpp_env.reset(seed)
        else:
            observations, infos = self._cpp_env.reset(None)

        self.agents = self.possible_agents[:]

        # Convert observations to numpy arrays
        observations = {
            agent: np.array(obs, dtype=np.float32)
            for agent, obs in observations.items()
        }

        # Update render state if rendering
        if self.render_mode is not None:
            render_state = self._cpp_env.get_render_state()

            # Update agent positions and colors
            for i, agent_name in enumerate(self.possible_agents):
                self.world.agents[i].state.p_pos = np.array(render_state[f"{agent_name}_pos"])
                self.world.agents[i].state.p_vel = np.array(render_state[f"{agent_name}_vel"])
                self.world.agents[i].color = np.array(render_state[f"{agent_name}_color"])

            # Update landmark positions
            for i in range(self.N):
                self.world.landmarks[i].state.p_pos = np.array(render_state[f"landmark_{i}_pos"])
                self.world.landmarks[i].state.p_vel = np.zeros(2)

        return observations, infos

    def step(self, actions):
        """Execute one step."""
        actions = check_and_maybe_clip_actions(actions, self.action_space, self.continuous_actions)
        # Convert discrete actions to vectors
        if not self.continuous_actions:
            actions = {
                agent: np.array([action], dtype=np.float32)
                for agent, action in actions.items()
            }
        else:
            actions = {
                agent: np.array(action, dtype=np.float32)
                for agent, action in actions.items()
            }

        observations, rewards, terminations, truncations, infos = self._cpp_env.step(actions)

        # Convert observations to numpy arrays
        observations = {
            agent: np.array(obs, dtype=np.float32)
            for agent, obs in observations.items()
        }

        # Update agent list (remove terminated/truncated agents)
        if any(terminations.values()) or any(truncations.values()):
            self.agents = []

        # Update render state if rendering
        if self.render_mode is not None:
            render_state = self._cpp_env.get_render_state()

            # Update agent positions and colors
            for i, agent_name in enumerate(self.possible_agents):
                self.world.agents[i].state.p_pos = np.array(render_state[f"{agent_name}_pos"])
                self.world.agents[i].state.p_vel = np.array(render_state[f"{agent_name}_vel"])
                self.world.agents[i].color = np.array(render_state[f"{agent_name}_color"])

            # Update landmark positions
            for i in range(self.N):
                self.world.landmarks[i].state.p_pos = np.array(render_state[f"landmark_{i}_pos"])
                self.world.landmarks[i].state.p_vel = np.zeros(2)

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
        """Return global state."""
        return np.array(self._cpp_env.get_state(), dtype=np.float32)

    def observation_space(self, agent):
        """Return observation space for agent."""
        return self._observation_spaces[agent]

    def action_space(self, agent):
        """Return action space for agent."""
        return self._action_spaces[agent]


# AEC wrapper
def env(**kwargs):
    return make_aec_env(parallel_env(**kwargs))
