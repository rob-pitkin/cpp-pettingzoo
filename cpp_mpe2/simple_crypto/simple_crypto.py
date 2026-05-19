"""SimpleCrypto environment with C++ backend."""

import sys
from pathlib import Path

import gymnasium
import numpy as np
import pygame
from gymnasium.utils import EzPickle
from pettingzoo import ParallelEnv
from pettingzoo.utils.conversions import parallel_to_aec_wrapper

build_dir = Path(__file__).parent.parent.parent / "build"
sys.path.insert(0, str(build_dir))

import _simple_crypto
from cpp_mpe2._wrappers import check_and_maybe_clip_actions, make_aec_env
from cpp_mpe2.core import Agent, Landmark, World


# Observation sizes (with dim_c=4):
#   alice (speaker): goal(4) + key(4) = 8
#   bob   (good listener): key(4) + alice_comm(4) = 8
#   eve   (adversary): alice_comm(4) = 4
_OBS_SHAPES = {
    "alice_0": 8,
    "bob_0": 8,
    "eve_0": 4,
}


class parallel_env(ParallelEnv, EzPickle):
    """SimpleCrypto parallel environment.

    Alice sends a 1-bit private message to Bob over a public channel; Eve
    eavesdrops. Alice and Bob share a private key; they're rewarded when Bob
    can reconstruct the message and Eve cannot.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "simple_crypto_v3",
        "is_parallelizable": True,
        "render_fps": 10,
    }

    def __init__(
        self,
        max_cycles=25,
        continuous_actions=False,
        render_mode=None,
        dynamic_rescaling=False,
         benchmark_data=False,
    ):
        EzPickle.__init__(
            self,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
            dynamic_rescaling=dynamic_rescaling,
             benchmark_data=benchmark_data,
        )

        self.max_cycles = max_cycles
        self.continuous_actions = continuous_actions
        self.render_mode = render_mode
        self.dynamic_rescaling = dynamic_rescaling
        self.dim_c = 4

        self._cpp_env = _simple_crypto.SimpleCryptoEnv(
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            dynamic_rescaling=dynamic_rescaling,
        )

        # mpe2 names agents in C++-creation order: eve_0, bob_0, alice_0
        self.possible_agents = ["eve_0", "bob_0", "alice_0"]
        self.agents = self.possible_agents[:]

        # All agents: Discrete(4) or Box(4,) — pure communication, no movement
        if self.continuous_actions:
            self._action_spaces = {
                name: gymnasium.spaces.Box(low=0.0, high=1.0, shape=(self.dim_c,),
                                            dtype="float32")
                for name in self.possible_agents
            }
        else:
            self._action_spaces = {
                name: gymnasium.spaces.Discrete(self.dim_c)
                for name in self.possible_agents
            }

        self._observation_spaces = {
            name: gymnasium.spaces.Box(low=-np.inf, high=np.inf,
                                       shape=(_OBS_SHAPES[name],), dtype="float32")
            for name in self.possible_agents
        }

        if self.render_mode is not None:
            pygame.init()
            self.width = 700
            self.height = 700
            self.screen = pygame.Surface([self.width, self.height])
            self.game_font = pygame.font.Font(None, 24)
            self.viewer = None
            self.renderOn = False
            self.original_cam_range = 1.0

            self.world = World()
            self.world.agents = [Agent(name, silent=False) for name in self.possible_agents]
            for agent in self.world.agents:
                agent.size = 0.05
                agent.color = np.array([0.25, 0.25, 0.25])
            self.world.landmarks = [Landmark(f"landmark {i}") for i in range(2)]
            # Landmark colors are one-hot in dim_c=4 in C++ but rendered as RGB;
            # truncate to 3 channels: landmark 0 → red, landmark 1 → green.
            _lm_colors = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])]
            for lm, col in zip(self.world.landmarks, _lm_colors):
                lm.size = 0.05
                lm.color = col

    def observation_space(self, agent):
        return self._observation_spaces[agent]

    def action_space(self, agent):
        return self._action_spaces[agent]

    def reset(self, seed=None, options=None):
        observations, infos = self._cpp_env.reset(seed)
        self.agents = self.possible_agents[:]
        observations = {
            agent: np.array(obs, dtype=np.float32)
            for agent, obs in observations.items()
        }
        if self.render_mode is not None:
            self._sync_render_state()
        return observations, infos

    def step(self, actions):
        actions = check_and_maybe_clip_actions(actions, self.action_space, self.continuous_actions)
        if not self.continuous_actions:
            actions = {agent: np.array([action], dtype=np.float32)
                       for agent, action in actions.items()}
        else:
            actions = {agent: np.array(action, dtype=np.float32)
                       for agent, action in actions.items()}

        observations, rewards, terminations, truncations, infos = (
            self._cpp_env.step(actions)
        )
        observations = {agent: np.array(obs, dtype=np.float32)
                        for agent, obs in observations.items()}

        if any(terminations.values()) or any(truncations.values()):
            self.agents = []

        if self.render_mode is not None:
            self._sync_render_state()

        return observations, rewards, terminations, truncations, infos

    def _sync_render_state(self):
        render_state = self._cpp_env.get_render_state()
        for i, name in enumerate(self.possible_agents):
            self.world.agents[i].state.p_pos = np.array(render_state[f"{name}_pos"])
            self.world.agents[i].state.p_vel = np.array(render_state[f"{name}_vel"])
            self.world.agents[i].color = np.array(render_state[f"{name}_color"])
        for i in range(len(self.world.landmarks)):
            self.world.landmarks[i].state.p_pos = np.array(
                render_state[f"landmark_{i}_pos"]
            )
            self.world.landmarks[i].state.p_vel = np.zeros(2)

    def enable_render(self, mode="human"):
        if not self.renderOn and mode == "human":
            self.screen = pygame.display.set_mode(self.screen.get_size())
            self.clock = pygame.time.Clock()
            self.renderOn = True

    def render(self):
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

    def draw(self):
        self.screen.fill((255, 255, 255))

        all_poses = [entity.state.p_pos for entity in self.world.entities]
        cam_range = np.max(np.abs(np.array(all_poses)))
        scaling_factor = 0.9 * self.original_cam_range / cam_range

        for entity in self.world.entities:
            x, y = entity.state.p_pos
            y *= -1
            x = (x / cam_range) * self.width // 2 * 0.9
            y = (y / cam_range) * self.height // 2 * 0.9
            x += self.width // 2
            y += self.height // 2

            if self.dynamic_rescaling:
                radius = entity.size * 350 * scaling_factor
            else:
                radius = entity.size * 350

            pygame.draw.circle(self.screen, entity.color * 200, (x, y), radius)
            pygame.draw.circle(self.screen, (0, 0, 0), (x, y), radius, 1)

    def close(self):
        if hasattr(self, "screen") and self.screen is not None:
            pygame.quit()
            self.screen = None

    def state(self):
        return np.array(self._cpp_env.get_state(), dtype=np.float32)


raw_env = parallel_env


def env(**kwargs):
    return make_aec_env(parallel_env(**kwargs))
