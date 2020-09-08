from typing import Optional, List

import gym
import numpy as np

from flatland.core.env_observation_builder import ObservationBuilder
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnvActions
from flatlander.envs.observations import Observation, register_obs
from flatlander.envs.observations.utils import norm_obs_clip


@register_obs("positional_tree")
class PositionalTreeObservation(Observation):
    PAD_VALUE = -np.inf

    def __init__(self, config) -> None:
        super().__init__(config)
        self._builder = PositionalTreeObsWrapper(
            TreeObsForRailEnv(
                max_depth=config['max_depth'],
                predictor=ShortestPathPredictorForRailEnv(config['shortest_path_max_depth'])
            )
        )

    def builder(self) -> ObservationBuilder:
        return self._builder

    def observation_space(self) -> gym.Space:
        return gym.spaces.Tuple((gym.spaces.Box(low=-1, high=1, shape=(self._builder.max_nr_nodes,
                                                                       self._builder.observation_dim,)),
                                 gym.spaces.Box(low=0, high=1,
                                                shape=(self._builder.max_nr_nodes,
                                                       self._builder.positional_encoding_len))))


class PositionalTreeObsWrapper(ObservationBuilder):

    def __init__(self, tree_obs_builder: TreeObsForRailEnv):
        super().__init__()
        self._builder = tree_obs_builder
        self._max_nr_nodes = 0
        for i in range(self._builder.max_depth + 1):
            self._max_nr_nodes += np.power(4, i)
        self._available_actions = [RailEnvActions.MOVE_FORWARD,
                                   RailEnvActions.DO_NOTHING,
                                   RailEnvActions.MOVE_LEFT,
                                   RailEnvActions.MOVE_RIGHT]
        self._positional_encoding_len = self._builder.max_depth * len(self._available_actions)
        self._max_dist_seen = 100

    @property
    def observation_dim(self):
        return self._builder.observation_dim - 4

    @property
    def positional_encoding_len(self):
        return self._positional_encoding_len

    @property
    def max_nr_nodes(self):
        return self._max_nr_nodes

    def reset(self):
        self._builder.reset()

    def get(self, handle: int = 0):
        obs: TreeObsForRailEnv.Node = self._builder.get(handle)
        return self._build_pairs(obs)

    def _build_pairs(self, obs_node: TreeObsForRailEnv.Node):
        encodings = []
        node_observations = []
        self.dfs(obs_node, -1, [],
                 encodings,
                 node_observations)
        node_observations = np.array(node_observations)
        padded_encodings = np.full(shape=(self.max_nr_nodes, self.positional_encoding_len,), fill_value=0.)
        padded_observations = np.full(shape=(self.max_nr_nodes, self.observation_dim,),
                                      fill_value=PositionalTreeObservation.PAD_VALUE)
        padded_encodings[:len(encodings), :] = np.array(encodings)
        padded_observations[:len(node_observations), :] = np.array(node_observations)
        padded_observations = np.clip(padded_observations, -1, 1)
        return padded_observations, padded_encodings

    def get_many(self, handles: Optional[List[int]] = None):
        result = {k: self._build_pairs(o)
                  for k, o in self._builder.get_many(handles).items() if o is not None}
        return result

    def set_env(self, env):
        self._builder.set_env(env)

    def _get_node_feature_vector(self, node: TreeObsForRailEnv.Node) -> np.ndarray:
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
            node_pos: int,
            ancestry: list,
            encodings: list,
            node_observations: list):
        """
        Depth first search, as operation should be used the inference
        :param node_observations: accumulated obs vectors of nodes
        :param encodings: accumulated pos encodings of nodes
        :param node_pos: Position of node relative to parent node
        :param ancestry: previous node encodings
        :param node: current node
        """
        ancestry = ancestry.copy()
        node_encoding = np.zeros(len(self._available_actions))
        if node_pos != -1:
            node_encoding[node_pos] = 1
            ancestry.extend(node_encoding)

        for action in self._available_actions:
            filtered = list(filter(lambda k: k == RailEnvActions.to_char(action.value), node.childs.keys()))
            if len(filtered) == 1 and not isinstance(node.childs[filtered[0]], float):
                self.dfs(node.childs[filtered[0]], action.value, ancestry, encodings, node_observations)

        positional_encoding = np.zeros(self.positional_encoding_len)
        positional_encoding[:len(ancestry)] = np.array(ancestry)
        node_observations.append(self._get_node_feature_vector(node))
        encodings.append(positional_encoding)
