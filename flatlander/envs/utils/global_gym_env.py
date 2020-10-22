from collections import defaultdict
from typing import Dict, NamedTuple, Any, Optional, Callable

import gym
import numpy as np

from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatlander.envs.utils.gym_env import StepOutput


class GlobalFlatlandGymEnv(gym.Env):
    action_space = gym.spaces.Discrete(5)

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 10,
        'semantics.autoreset': True
    }

    def __init__(self,
                 rail_env: RailEnv,
                 observation_space: gym.spaces.Space,
                 regenerate_rail_on_reset: bool = True,
                 regenerate_schedule_on_reset: bool = True, config=None, **kwargs) -> None:
        super().__init__()
        self._agents_done = []
        self._agent_scores = defaultdict(float)
        self._agent_steps = defaultdict(int)
        self._regenerate_rail_on_reset = regenerate_rail_on_reset
        self._regenerate_schedule_on_reset = regenerate_schedule_on_reset
        self.rail_env = rail_env
        self.observation_space = observation_space
        self.exclude_done_agents = config.get("exclude_done_agents", True)
        self.fill_done_agents = config.get("fill_done_agents", True)
        self.global_done_signal = config.get("global_done_signal", False)
        self._step_out: Callable = self.get_independent_done_observations if self.exclude_done_agents \
            else self.get_global_done_observations

    def step(self, action_dict: Dict[int, RailEnvActions]) -> StepOutput:
        obs, rewards, dones, infos = self.rail_env.step(action_dict)

        o, r, d = self._step_out(obs, dones)

        assert len(obs) > 0
        assert all([x is not None for x in (dones, rewards, obs)])

        return StepOutput(obs=o, reward=r, done=d, info={agent: {
            'max_episode_steps': self.rail_env._max_episode_steps,
            'num_agents': self.rail_env.get_num_agents(),
            'agent_done': dones[agent] and agent not in self.rail_env.active_agents,
            'agent_score': self._agent_scores[agent],
            'agent_step': self._agent_steps[agent],
        } for agent in o.keys()})

    def get_independent_done_observations(self, obs, dones):
        o, r, d = {}, {}, {}
        for handle, done in dones.items():
            if handle != "__all__":
                if done and handle not in self._agents_done:
                    r[handle] = 0
                    o[handle] = obs[handle]
                    d[handle] = done
                    self._agents_done.append(handle)
                elif handle not in self._agents_done:
                    o[handle] = obs[handle]
                    r[handle] = -1
                    d[handle] = done
            else:
                d[handle] = done

        global_reward = np.sum(list(r.values()), dtype=np.float) if not d["__all__"] else 1.
        r = {handle: global_reward for handle in r.keys()}
        return o, r, d

    def get_global_done_observations(self, obs, dones):
        o, r, d = {}, {}, {}
        for handle, done in dones.items():
            if handle != "__all__":
                if done:
                    r[handle] = 0
                    o[handle] = np.zeros(shape=self.observation_space.shape) if self.fill_done_agents else obs[handle]
                else:
                    r[handle] = -1
                    o[handle] = obs[handle]
            if self.global_done_signal:
                d[handle] = dones["__all__"]
            else:
                d[handle] = done

        global_reward = np.mean(list(r.values()), dtype=np.float) if not d["__all__"] else 1.
        r = {handle: global_reward for handle in r.keys()}
        assert len(o.keys()) == self.rail_env.get_num_agents()
        return o, r, d

    def reset(self, random_seed: Optional[int] = None) -> Dict[int, Any]:
        self._agents_done = []
        self._agent_scores = defaultdict(float)
        self._agent_steps = defaultdict(int)
        obs, infos = self.rail_env.reset(regenerate_rail=self._regenerate_rail_on_reset,
                                         regenerate_schedule=self._regenerate_schedule_on_reset,
                                         random_seed=random_seed)
        return {k: o for k, o in obs.items() if not k == '__all__'}

    def render(self, mode='human'):
        return self.rail_env.render(mode)

    def close(self):
        self.rail_env.close()
