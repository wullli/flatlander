from collections import defaultdict

import gym
from flatland.envs.agent_utils import RailAgentStatus

from flatlander.agents.shortest_path_agent import ShortestPathAgent
from flatlander.envs import get_generator_config
from flatlander.envs.flatland_sparse import FlatlandSparse
from flatlander.envs.observations import make_obs
from flatlander.envs.utils.gym_env import StepOutput
from flatlander.envs.utils.robust_gym_env import RobustFlatlandGymEnv
import numpy as np

from flatlander.utils.helper import is_done


class FlatlandMeta(FlatlandSparse):
    action_space = gym.spaces.Box(low=0, high=1, shape=(1,))

    def __init__(self, env_config, **kwargs):
        super(FlatlandMeta, self).__init__(env_config, **kwargs)
        assert env_config['generator'] == 'sparse_rail_generator'
        self._env_config = env_config

        self._observation = make_obs(env_config['observation'], env_config.get('observation_config'))
        self._config = get_generator_config(env_config['generator_config'])

        if env_config.get('number_of_agents', None) is not None:
            self._config['number_of_agents'] = env_config['number_of_agents']

        # Overwrites with env_config seed if it exists
        if env_config.get('seed'):
            self._config['seed'] = env_config.get('seed')

        if not hasattr(env_config, 'worker_index') or (env_config.worker_index == 0 and env_config.vector_index == 0):
            print("=" * 50)
            print(self._config)
            print("=" * 50)

        self._env = RobustFlatlandGymEnv(
            rail_env=self._launch(),
            observation_space=self._observation.observation_space(),
            render=env_config.get('render'),
            regenerate_rail_on_reset=self._config['regenerate_rail_on_reset'],
            regenerate_schedule_on_reset=self._config['regenerate_schedule_on_reset'],
            config=env_config,
            allow_noop=True
        )
        self.last_obs = None

    def reset(self):
        """
        Assume first index is meta obs
        :return:
        """
        self.last_obs = super(FlatlandMeta, self).reset()
        return self.last_obs

    def step(self, action_dict):
        return self._scheduling_step(action_dict)

    def _scheduling_step(self, action):
        norm_factor = self._env.rail_env._max_episode_steps * self._env.rail_env.get_num_agents()
        sorted_actions = {k: v for k, v in sorted(action.items(), key=lambda item: item[1], reverse=True)}
        self._env.sorted_handles = list(sorted_actions.keys())

        done = defaultdict(lambda: False)
        while not done['__all__']:
            actions = ShortestPathAgent().compute_actions(self.last_obs, self._env.rail_env)
            _, _, done, _ = self._env.step(actions)

        pc = np.sum(
            np.array([1 for a in self._env.rail_env.agents if is_done(a)])) / self._env.rail_env.get_num_agents()
        malf = np.sum([a.malfunction_data['nr_malfunctions'] for a in self._env.rail_env.agents])
        print("EPISODE PC:", pc, "NR MALFUNCTIONS:", malf)

        d = {a.handle: a.status == RailAgentStatus.DONE or a.status == RailAgentStatus.DONE_REMOVED
             for a in self._env.rail_env.agents}
        d['__all__'] = True

        r = {a.handle: self._env._agent_scores[a.handle] / norm_factor for a in self._env.rail_env.agents}
        o = self.last_obs
        return StepOutput(obs=o, reward=r, done=d, info={a.handle: {
            'max_episode_steps': self._env.rail_env._max_episode_steps,
            'num_agents': self._env.rail_env.get_num_agents(),
            'agent_done': d[a.handle],
            'agent_score': self._env._agent_scores[a.handle],
            'agent_step': self._env._agent_steps[a.handle],
        } for a in self._env.rail_env.agents})
