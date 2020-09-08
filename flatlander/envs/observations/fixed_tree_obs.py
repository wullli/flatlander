from typing import Optional, List

import gym
import numpy as np

from flatland.core.env_observation_builder import ObservationBuilder
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnvActions
from flatlander.envs.observations import Observation, register_obs
from flatlander.envs.observations.utils import norm_obs_clip


@register_obs("fixed_tree")
class FixedTreeObservation(Observation):
    PAD_VALUE = -np.inf

    def __init__(self, config) -> None:
        super().__init__(config)
        self._builder = FixedTreeObsWrapper(
            TreeObsForRailEnv(
                max_depth=config['max_depth'],
                predictor=ShortestPathPredictorForRailEnv(config['shortest_path_max_depth'])
            )
        )

    def builder(self) -> ObservationBuilder:
        return self._builder

    def observation_space(self) -> gym.Space:
        return gym.spaces.Box(low=-1, high=1, shape=(self._builder.max_nr_nodes,
                                                     self._builder.observation_dim,))


class FixedTreeObsWrapper(ObservationBuilder):

    def __init__(self, tree_obs_builder: TreeObsForRailEnv):
        super().__init__()
        self._builder = tree_obs_builder
        self._max_nr_nodes = 0
        self._available_actions = [RailEnvActions.MOVE_FORWARD,
                                   RailEnvActions.DO_NOTHING,
                                   RailEnvActions.MOVE_LEFT,
                                   RailEnvActions.MOVE_RIGHT]
        for i in range(self._builder.max_depth + 1):
            self._max_nr_nodes += np.power(len(self._available_actions), i)

    @property
    def observation_dim(self):
        return self._builder.observation_dim - 4

    @property
    def max_nr_nodes(self):
        return self._max_nr_nodes

    def reset(self):
        self._builder.reset()

    def get(self, handle: int = 0):
        obs: TreeObsForRailEnv.Node = self._builder.get(handle)
        return self.build_obs(obs)

    def build_obs(self, obs_node: TreeObsForRailEnv.Node):
        padded_observations = np.full(shape=(self.max_nr_nodes, self.observation_dim,),
                                      fill_value=FixedTreeObservation.PAD_VALUE)
        self.dfs(obs_node, padded_observations)
        padded_observations = np.clip(padded_observations, -1, 1)
        return padded_observations

    def get_many(self, handles: Optional[List[int]] = None):
        result = {k: self.build_obs(o)
                  for k, o in self._builder.get_many(handles).items() if o is not None}
        return result

    def set_env(self, env):
        self._builder.set_env(env)

    @staticmethod
    def _get_node_feature_vector(node: TreeObsForRailEnv.Node) -> np.ndarray:

        data = np.zeros(3)
        distance = np.zeros(1)
        agent_data = np.zeros(3)

        data[0] = node.dist_own_target_encountered
        data[1] = node.dist_potential_conflict
        data[2] = node.dist_unusable_switch

        distance[0] = node.dist_min_to_target

        agent_data[0] = node.num_agents_opposite_direction
        agent_data[1] = node.num_agents_malfunctioning
        agent_data[2] = node.speed_min_fractional

        data = norm_obs_clip(data, fixed_radius=10)
        distance = norm_obs_clip(distance, fixed_radius=100)
        agent_data = np.clip(agent_data, -1, 1)
        normalized_obs = np.concatenate([data, distance, agent_data])

        return normalized_obs

    def dfs(self, node: TreeObsForRailEnv.Node,
            node_observations: np.ndarray, current_level=0, abs_pos=0):
        """
        Depth first search, as operation should be used the inference
        :param abs_pos: absolute index in flat obs vector
        :param current_level: current level of node in the tree (how deep)
        :param node_observations: accumulated obs vectors of nodes
        :param node: current node
        """
        for action in self._available_actions:
            filtered = list(filter(lambda k: k == RailEnvActions.to_char(action.value), node.childs.keys()))
            if len(filtered) == 1 and not isinstance(node.childs[filtered[0]], float):
                abs_pos = self.dfs(node.childs[filtered[0]],
                                   node_observations,
                                   current_level=current_level + 1,
                                   abs_pos=abs_pos)
            elif current_level != self._builder.max_depth:
                abs_pos += self._count_missing_nodes(current_level + 1)

        node_observations[abs_pos, :] = self._get_node_feature_vector(node)
        return abs_pos + 1

    def _count_missing_nodes(self, tree_level):
        missing_nodes = 0
        for i in range((self._builder.max_depth + 1) - tree_level):
            missing_nodes += np.power(len(self._available_actions), i)
        return missing_nodes
