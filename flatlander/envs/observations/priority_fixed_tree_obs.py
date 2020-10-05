from typing import Optional, List

import gym
import numpy as np
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.envs.observations import Node
from flatland.envs.predictions import ShortestPathPredictorForRailEnv

from flatlander.envs.observations import Observation, register_obs
from flatlander.envs.observations.builders.priority_tree import PriorityTreeObs
from flatlander.envs.observations.common.fixed_tree_flattener import FixedTreeFlattener


@register_obs("priority_fixed_tree")
class PriorityFixedTreeObservation(Observation):
    PAD_VALUE = -np.inf

    def __init__(self, config) -> None:
        super().__init__(config)
        self._builder = PriorityFixedTreeObsWrapper(
            PriorityTreeObs(
                max_depth=config['max_depth'],
                predictor=ShortestPathPredictorForRailEnv(config['shortest_path_max_depth'])
            ),
        )

    def builder(self) -> ObservationBuilder:
        return self._builder

    def observation_space(self) -> gym.Space:
        return gym.spaces.Tuple((gym.spaces.Box(low=-1, high=1, shape=(self._builder.max_nr_nodes,
                                                                       self._builder.observation_dim,)),
                                 gym.spaces.Box(low=-1, high=1, shape=(16,))))


class PriorityFixedTreeObsWrapper(ObservationBuilder):

    def __init__(self, tree_obs_builder: PriorityTreeObs):
        super().__init__()
        self._builder = tree_obs_builder
        self.max_nr_nodes = 0

        for i in range(self._builder.max_depth):
            self.max_nr_nodes += np.power(4, i) * 4

        self._flattener = FixedTreeFlattener(tree_depth=tree_obs_builder.max_depth,
                                             max_nr_nodes=self.max_nr_nodes,
                                             observation_dim=self._builder.observation_dim)

    @property
    def observation_dim(self):
        return self._flattener.observation_dim

    def reset(self):
        self._builder.reset()

    def get(self, handle: int = 0):
        obs: Node = self._builder.get(handle)
        return self._flattener.flatten(root=obs[0], agent_info=obs[1])

    def get_many(self, handles: Optional[List[int]] = None):
        result = {k: self._flattener.flatten(root=o[0], agent_info=o[1])
                  for k, o in self._builder.get_many(handles).items() if o is not None}
        return result

    def set_env(self, env):
        self._builder.set_env(env)
