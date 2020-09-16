from collections import defaultdict
from typing import Dict, Any, Optional

import numpy as np

from flatland.envs.rail_env import RailEnvActions
from flatlander.envs.utils.gym_env import FlatlandGymEnv, StepOutput


class FillingFlatlandGymEnv(FlatlandGymEnv):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.missing_fill_value = np.full(shape=self.observation_space.shape,
                                          fill_value=config.get("missing_fill_value", -1))
        self.done_fill_value = np.full(shape=self.observation_space.shape, fill_value=config.get("done_fill_value", 1))
        self.agent_keys = list(range(config.get("n_agents", 5)))
        self.agent_done_independent = config.get("agents_done_independent", False)
        self.prev_obs = defaultdict(lambda : self.missing_fill_value)

    def step(self, action_dict: Dict[int, RailEnvActions]) -> StepOutput:
        d, r, o = None, None, None
        obs_or_done = False

        action_dict = {k: v for k, v in action_dict.items() if k not in self._agents_done}

        while not obs_or_done:
            # Perform envs steps as long as there is no observation (for all agents) or all agents are done
            # The observation is `None` if an agent is done or malfunctioning.

            obs, rewards, dones, infos = self.rail_env.step(action_dict)

            d, r, o = dict(), dict(), dict()
            for agent in self.agent_keys + ["__all__"]:
                if agent != '__all__':
                    if dones.get(agent, False):
                        if agent not in self._agents_done:
                            self._agents_done.append(agent)
                    if self.agent_done_independent and agent not in self._agents_done:
                        o[agent] = obs.get(agent, self.prev_obs[agent])
                        r[agent] = rewards.get(agent, 0)
                        self._agent_scores[agent] += rewards.get(agent, 0)
                        self._agent_steps[agent] += 1

                    elif not self.agent_done_independent:
                        o[agent] = obs.get(agent, self.prev_obs[agent])
                        r[agent] = rewards.get(agent, 0)
                        self._agent_scores[agent] += rewards.get(agent, 0)
                        self._agent_steps[agent] += 1

                if self.agent_done_independent:
                    d[agent] = dones[agent]
                else:
                    d[agent] = dones["__all__"]

            action_dict = {}  # reset action dict for cases where we do multiple envs steps
            obs_or_done = len(o) > 0 or d['__all__']  # step through envs as long as there are no obs/all agents done

        assert all([x is not None for x in (d, r, o)])

        self.prev_obs = o

        return StepOutput(obs=o, reward=r, done=d, info={agent: {
            'max_episode_steps': self.rail_env._max_episode_steps,
            'num_agents': self.rail_env.get_num_agents(),
            'agent_done': d.get(agent, False) and agent not in self.rail_env.active_agents,
            'agent_score': self._agent_scores.get(agent, 0),
            'agent_step': self._agent_steps.get(agent, 0),
        } for agent in o.keys()})

    def reset(self, random_seed: Optional[int] = None) -> Dict[int, Any]:
        self._agents_done = []
        self._agent_scores = defaultdict(float)
        self._agent_steps = defaultdict(int)
        obs, infos = self.rail_env.reset(regenerate_rail=self._regenerate_rail_on_reset,
                                         regenerate_schedule=self._regenerate_schedule_on_reset,
                                         random_seed=random_seed)
        return {k: obs.get(k, self.missing_fill_value)
                for k in self.agent_keys}
