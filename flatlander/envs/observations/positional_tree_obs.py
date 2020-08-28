from typing import Optional, List

import gym
import numpy as np

from flatland.core.env_observation_builder import ObservationBuilder
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatlander.envs.observations import Observation, register_obs
from flatlander.envs.observations.utils import norm_obs_clip
from flatlander.envs.utils.const import NUMBER_ACTIONS


class GenericNode:
    def __init__(self, obs_vector: np.ndarray = None, children: list = None):
        self.obs_vector: np.ndarray = obs_vector
        self.children: List[GenericNode] = children


@register_obs("positional_tree")
class PositionalTreeObservation(Observation):

    def __init__(self, config) -> None:
        super().__init__(config)
        self._builder = PositionalTreeObsRLLibWrapper(
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


class PositionalTreeObsRLLibWrapper(ObservationBuilder):

    def __init__(self, tree_obs_builder: TreeObsForRailEnv):
        super().__init__()
        self._builder = tree_obs_builder
        self._positional_encoding_len = (self._builder.max_depth + 1) * NUMBER_ACTIONS
        self._max_nr_nodes = 0
        for i in range(self._builder.max_depth + 1):
            self._max_nr_nodes += np.power(4, i)

    @property
    def observation_dim(self):
        return self._builder.observation_dim

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
        root = self._build_tree(obs_node)
        encodings = []
        node_observations = []
        self.dfs(root, -1, [], encodings, node_observations)
        padded_encodings = np.full(shape=(self.max_nr_nodes, self.positional_encoding_len,), fill_value=-np.inf)
        padded_observations = np.full(shape=(self.max_nr_nodes, self.observation_dim,), fill_value=-np.inf)
        padded_observations[:len(node_observations), :] = np.array(node_observations)
        padded_encodings[:len(encodings), :] = np.array(encodings)

    def _build_tree(self, node: TreeObsForRailEnv.Node) -> GenericNode:
        new_children = []
        ordered_children = sorted(node.childs.items())
        for _, child in ordered_children:
            if child != -np.inf:
                new_children.append(self._build_tree(child))
        return GenericNode(self._get_node_feature_vector(node), children=new_children)

    def get_many(self, handles: Optional[List[int]] = None):
        result = {k: self._build_pairs(o)
                  for k, o in self._builder.get_many(handles).items() if o is not None}
        return result

    def util_print_obs_subtree(self, tree):
        self._builder.util_print_obs_subtree(tree)

    def print_subtree(self, node, label, indent):
        self._builder.print_subtree(node, label, indent)

    def set_env(self, env):
        self._builder.set_env(env)

    @staticmethod
    def _get_node_feature_vector(node: TreeObsForRailEnv.Node) -> np.ndarray:
        data = np.zeros(6)
        distance = np.zeros(1)
        agent_data = np.zeros(4)

        data[0] = node.dist_own_target_encountered
        data[1] = node.dist_other_target_encountered
        data[2] = node.dist_other_agent_encountered
        data[3] = node.dist_potential_conflict
        data[4] = node.dist_unusable_switch
        data[5] = node.dist_to_next_branch

        distance[0] = node.dist_min_to_target

        agent_data[0] = node.num_agents_same_direction
        agent_data[1] = node.num_agents_opposite_direction
        agent_data[2] = node.num_agents_malfunctioning
        agent_data[3] = node.speed_min_fractional

        data = norm_obs_clip(data, fixed_radius=10)
        distance = norm_obs_clip(distance, normalize_to_range=True)
        agent_data = np.clip(agent_data, -1, 1)
        normalized_obs = np.concatenate([data, distance, agent_data])

        return normalized_obs

    def dfs(self, node: GenericNode,
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
        node_encoding = np.zeros(NUMBER_ACTIONS)
        if node_pos != -1:
            node_encoding[node_pos] = 1
        ancestry.extend(node_encoding)

        for i, child in enumerate(node.children):
            self.dfs(child, i, ancestry, encodings, node_observations)

        positional_encoding = np.zeros(self.positional_encoding_len)
        positional_encoding[:len(ancestry)] = np.array(ancestry)
        node_observations.append(node.obs_vector)
        encodings.append(positional_encoding)
