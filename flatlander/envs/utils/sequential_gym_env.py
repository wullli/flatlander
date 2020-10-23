from collections import defaultdict
from typing import Dict, Any, Optional

import gym
from flatland.envs.rail_env import RailEnv, RailEnvActions

from flatlander.envs.utils.gym_env import StepOutput


class SequentialFlatlandGymEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 10,
        'semantics.autoreset': True
    }

    def __init__(self,
                 rail_env: RailEnv,
                 observation_space: gym.spaces.Space,
                 regenerate_rail_on_reset: bool = True,
                 allow_noop: bool = True,
                 regenerate_schedule_on_reset: bool = True, **_) -> None:
        super().__init__()
        self._agents_done = []
        self._allow_noop = allow_noop
        self._agent_scores = defaultdict(float)
        self._agent_steps = defaultdict(int)
        self._regenerate_rail_on_reset = regenerate_rail_on_reset
        self._regenerate_schedule_on_reset = regenerate_schedule_on_reset
        self.rail_env = rail_env
        self.observation_space = observation_space
        self._current_handle = 0
        self._num_agents = self.rail_env.get_num_agents()
        if allow_noop:
            self.action_space = gym.spaces.Discrete(5)
        else:
            self.action_space = gym.spaces.Discrete(4)

    def step(self, action: Dict[int, RailEnvActions]) -> StepOutput:
        action_dict = {h: RailEnvActions.STOP_MOVING.value for h in range(self._num_agents)
                       if h not in self._agents_done}
        if self._allow_noop:
            action_dict[self._current_handle] = action[self._current_handle]
        else:
            action_dict[self._current_handle] = action[self._current_handle] + 1

        obs, rewards, dones, infos = self.rail_env.step(action_dict)

        new_dones = []
        for agent, done in dones.items():
            if agent not in self._agents_done and agent != '__all__' and done:
                new_dones.append(agent)

        if not dones['__all__']:
            self._current_handle = (self._current_handle + 1) % self._num_agents
            while self._current_handle in (self._agents_done + new_dones):
                self._current_handle = (self._current_handle + 1) % self._num_agents

        d, r, o = dict(), dict(), dict()
        for agent in new_dones + [self._current_handle]:
            o[agent] = obs[agent]
            r[agent] = rewards[agent]
            d[agent] = dones[agent]
            self._agent_scores[agent] += rewards[agent]
            if not d[agent]:
                self._agent_steps[agent] += 1

        d['__all__'] = dones['__all__']
        self._agents_done.extend(new_dones)

        i = {h: self.get_agent_info(h, d) for h, done in d.items() if not h == '__all__'}

        return StepOutput(obs=o, reward=r, done=d, info=i)

    def get_agent_info(self, handle, d):
        return {
            'max_episode_steps': self.rail_env._max_episode_steps / 5,
            'num_agents': self.rail_env.get_num_agents(),
            'agent_done': d[handle] and handle
                          not in self.rail_env.active_agents,
            'agent_score': self._agent_scores[handle],
            'agent_step': self._agent_steps[handle],
        }

    def reset(self, random_seed: Optional[int] = None) -> Dict[int, Any]:
        self._agents_done = []
        self._agent_scores = defaultdict(float)
        self._agent_steps = defaultdict(int)
        self._current_handle = 0
        self._num_agents = self.rail_env.get_num_agents()
        obs, infos = self.rail_env.reset(regenerate_rail=self._regenerate_rail_on_reset,
                                         regenerate_schedule=self._regenerate_schedule_on_reset,
                                         random_seed=random_seed)
        return {self._current_handle: obs[self._current_handle]}
