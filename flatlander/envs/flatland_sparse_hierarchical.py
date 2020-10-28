import gym

from flatlander.envs.flatland_sparse import FlatlandSparse
import numpy as np

from flatlander.envs.utils.gym_env import StepOutput
from flatlander.envs.utils.robust_gym_env import RobustFlatlandGymEnv


class FlatlandSparseHierarchical(FlatlandSparse):
    def __init__(self, env_config, **kwargs):
        super(FlatlandSparseHierarchical, self).__init__(env_config, **kwargs)
        assert isinstance(self.observation_space, gym.spaces.Tuple), "Need tuple observations for hierarchical env!"
        assert isinstance(self._env, RobustFlatlandGymEnv), "Need robust gym env to specify scheduling orders!"
        self._env: RobustFlatlandGymEnv
        self._low_level_reset_obs = None
        self._high_level_reset_obs = None

    def reset(self):
        """
        Assume first index is meta obs
        :return:
        """
        obs = super(FlatlandSparseHierarchical, self).reset()
        self._low_level_reset_obs = {f'agent_{h}': o[1] for h, o in obs.items()}
        self._high_level_reset_obs = {f'meta_{h}': o[0] for h, o in obs.items()}
        return self._high_level_reset_obs

    def step(self, action_dict):
        if np.any(['meta' in str(agent_id) for agent_id in list(action_dict.keys())]):
            return self._scheduling_step(action_dict)
        else:
            o, r, d, i = super(FlatlandSparseHierarchical, self).step(action_dict)
            obs, rew, dones, info = {}, {}, {}, {}
            for h, done in d.items():
                if h == '__all__':
                    dones['__all__'] = done
                    continue
                obs[f'agent_{h}'] = o[h][1]
                rew[f'agent_{h}'] = r[h]
                dones[f'agent_{h}'] = d[h]
                info[f'agent_{h}'] = i[h]
            if d['__all__']:
                ep_return = np.sum([ai['agent_score'] for ai in i.values()])
                for handle in range(len(self._env.rail_env.agents)):
                    agent_id = 'meta_' + str(handle)
                    d[agent_id] = True
                    r[agent_id] = ep_return
                    o[agent_id] = self._high_level_reset_obs[agent_id]
                    i[agent_id] = {}
            return StepOutput(obs=obs, reward=rew, done=dones, info=info)

    def _scheduling_step(self, action):
        sorted_actions = {int(k.replace('meta_', '')): v
                          for k, v in sorted(action.items(), key=lambda item: item[1], reverse=True)}
        self._env.sorted_handles = list(sorted_actions.keys())

        d = {h: False for h in self._low_level_reset_obs.keys()}
        d['__all__'] = False
        r = {h: 0 for h in self._low_level_reset_obs.keys()}
        return StepOutput(obs=self._low_level_reset_obs, reward=r, done=d, info={agent: {
            'max_episode_steps': self._env.rail_env._max_episode_steps,
            'num_agents': self._env.rail_env.get_num_agents(),
            'agent_done': d[agent] and int(agent.replace('agent_', '')) not in self._env.rail_env.active_agents,
            'agent_score': self._env._agent_scores[int(agent.replace('agent_', ''))],
            'agent_step': self._env._agent_steps[int(agent.replace('agent_', ''))],
        } for agent in self._low_level_reset_obs.keys()})
