from collections import defaultdict
from typing import Dict, NamedTuple, Any, Optional

import gym

from flatland.envs.rail_env import RailEnv, RailEnvActions
import numpy as np

from flatlander.envs.utils.gym_env import StepOutput


class SingleFlatlandGymEnv(gym.Env):
    def render(self, mode='human'):
        pass

    action_space = gym.spaces.MultiDiscrete(5)

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 10,
        'semantics.autoreset': True
    }

    def __init__(self,
                 rail_env: RailEnv,
                 observation_space: gym.spaces.Space,
                 regenerate_rail_on_reset: bool = True,
                 regenerate_schedule_on_reset: bool = True, **_) -> None:
        super().__init__()
        self._agent_score = 0
        self._agent_steps = 0
        self._regenerate_rail_on_reset = regenerate_rail_on_reset
        self._regenerate_schedule_on_reset = regenerate_schedule_on_reset
        self.rail_env = rail_env
        self.observation_space = observation_space

    def step(self, action_dict: Dict[int, RailEnvActions]) -> StepOutput:
        obs, rewards, dones, infos = self.rail_env.step(action_dict)
        done = dones["__all__"]
        reward = np.sum(list(rewards.values()))
        observation = obs

        self._agent_score += reward
        self._agent_steps += 1

        return StepOutput(obs=observation, reward=reward, done=done, info={
            'max_episode_steps': self.rail_env._max_episode_steps,
            'num_agents': self.rail_env.get_num_agents(),
            'agent_done': done,
            'agent_score': self._agent_score,
            'agent_step': self._agent_steps,
        })

    def reset(self, random_seed: Optional[int] = None) -> Dict[int, Any]:
        self._agent_score = 0
        self._agent_steps = 0
        obs, infos = self.rail_env.reset(regenerate_rail=self._regenerate_rail_on_reset,
                                         regenerate_schedule=self._regenerate_schedule_on_reset,
                                         random_seed=random_seed)
        return obs
