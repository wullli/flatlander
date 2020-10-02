from typing import Optional, List

import gym
import numpy as np
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv

from flatlander.envs.observations import Observation, register_obs
from flatlander.envs.observations.builders.priority_tree import PriorityTreeObs
from flatlander.envs.observations.tree_obs import normalize_observation


@register_obs("priority_tree")
class PriorityTreeObservation(Observation):

    def __init__(self, config) -> None:
        super().__init__(config)
        self._builder = PriorityTreeObsWrapper(
            PriorityTreeObs(
                max_depth=config['max_depth'],
                predictor=ShortestPathPredictorForRailEnv(config['shortest_path_max_depth'])
            ),
            config.get('normalize_fixed', None),
        )

    def builder(self) -> ObservationBuilder:
        return self._builder

    def observation_space(self) -> gym.Space:
        num_features_per_node = self._builder.observation_dim
        nr_nodes = 0
        for i in range(self.config['max_depth'] + 1):
            nr_nodes += np.power(4, i)
        dim = num_features_per_node * nr_nodes
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(dim + 14,))  # 14 agent info fields


class PriorityTreeObsWrapper(ObservationBuilder):

    def __init__(self,
                 tree_obs_builder: TreeObsForRailEnv,
                 normalize_fixed=None):
        super().__init__()
        self._builder = tree_obs_builder
        self._normalize_fixed = normalize_fixed

    @property
    def observation_dim(self):
        return self._builder.observation_dim

    def reset(self):
        self._builder.reset()

    def get(self, handle: int = 0):
        obs, agent_info = self._builder.get(handle)

        obs_branches = []
        for k, node in obs:
            norm_obs = normalize_observation(node, self._builder.max_depth, observation_radius=10,
                                             normalize_fixed=self._normalize_fixed)
            obs_branches.append(norm_obs)

        obs = np.concatenate(obs_branches + [agent_info])
        return obs

    def get_many(self, handles: Optional[List[int]] = None):
        obs_dict = {}
        for handle, obs in self._builder.get_many(handles).items():
            obs_branches = []
            for k, node in obs[0].items():
                norm_obs = normalize_observation(node, self._builder.max_depth, observation_radius=10,
                                                 normalize_fixed=self._normalize_fixed)
                obs_branches.append(norm_obs)

            obs = np.concatenate(obs_branches + [obs[1]])
            obs_dict[handle] = obs

    def util_print_obs_subtree(self, tree):
        self._builder.util_print_obs_subtree(tree)

    def print_subtree(self, node, label, indent):
        self._builder.print_subtree(node, label, indent)

    def set_env(self, env):
        self._builder.set_env(env)
