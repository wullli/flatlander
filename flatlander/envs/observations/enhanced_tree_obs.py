from typing import Optional, List

import gym
import numpy as np
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.envs.observations import TreeObsForRailEnv, Node
from flatland.envs.predictions import ShortestPathPredictorForRailEnv

from flatlander.envs.observations import Observation, register_obs
from flatlander.envs.observations.common.malf_shortest_path_predictor import MalfShortestPathPredictorForRailEnv
from flatlander.envs.observations.common.utils import norm_obs_clip


@register_obs("enhanced_tree")
class TreeObservation(Observation):

    def __init__(self, config) -> None:
        super().__init__(config)
        self._concat_agent_id = config.get('concat_agent_id', False)
        self._num_agents = config.get('num_agents', 5)
        self._builder = TreeObsForRailEnvRLLibWrapper(
            TreeObsForRailEnv(
                max_depth=config['max_depth'],
                predictor=MalfShortestPathPredictorForRailEnv(config['shortest_path_max_depth'])
            ),
            config.get('normalize_fixed', None),
            self._concat_agent_id,
            self._num_agents
        )

    def builder(self) -> ObservationBuilder:
        return self._builder

    def observation_space(self) -> gym.Space:
        num_features_per_node = self._builder.observation_dim - 2
        nr_nodes = 0
        for i in range(self.config['max_depth'] + 1):
            nr_nodes += np.power(4, i)
        dim = num_features_per_node * nr_nodes
        if self._concat_agent_id:
            dim += self._num_agents
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(dim,))


def _is_root(node):
    return node.dist_own_target_encountered == 0 and \
           node.dist_other_target_encountered == 0 and \
           node.dist_other_agent_encountered == 0 and \
           node.dist_potential_conflict == 0


def _split_node_into_feature_groups(node: Node) -> (np.ndarray, np.ndarray, np.ndarray):
    data = np.zeros(4)
    flags = np.zeros(2)
    distance = np.zeros(1)
    agent_data = np.zeros(2)

    if not _is_root(node):
        flags[1] = float(node.dist_own_target_encountered == 0)
        flags[0] = float(node.dist_own_target_encountered != np.inf)

    data[0] = node.dist_other_target_encountered
    data[1] = node.dist_potential_conflict
    data[2] = node.dist_unusable_switch
    data[3] = node.dist_to_next_branch

    distance[0] = node.dist_min_to_target

    agent_data[0] = node.num_agents_opposite_direction
    agent_data[1] = node.num_agents_malfunctioning

    return flags, data, distance, agent_data


def _split_subtree_into_feature_groups(node: Node, current_tree_depth: int, max_tree_depth: int) -> (
        np.ndarray, np.ndarray, np.ndarray):
    if node == -np.inf:
        remaining_depth = max_tree_depth - current_tree_depth
        # reference: https://stackoverflow.com/questions/515214/total-number-of-nodes-in-a-tree-data-structure
        num_remaining_nodes = int((4 ** (remaining_depth + 1) - 1) / (4 - 1))
        return [-np.inf] * num_remaining_nodes * 2, \
               [-np.inf] * num_remaining_nodes * 4, \
               [-np.inf] * num_remaining_nodes, \
               [-np.inf] * num_remaining_nodes * 2

    flags, data, distance, agent_data = _split_node_into_feature_groups(node)

    if not node.childs:
        return flags, data, distance, agent_data

    for direction in TreeObsForRailEnv.tree_explored_actions_char:
        sub_flags, sub_data, sub_distance, sub_agent_data = _split_subtree_into_feature_groups(node.childs[direction],
                                                                                               current_tree_depth + 1,
                                                                                               max_tree_depth)
        flags = np.concatenate((flags, sub_flags))
        data = np.concatenate((data, sub_data))
        distance = np.concatenate((distance, sub_distance))
        agent_data = np.concatenate((agent_data, sub_agent_data))

    return flags, data, distance, agent_data


def split_tree_into_feature_groups(tree: Node, max_tree_depth: int) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    This function splits the tree into three difference arrays of values
    """
    flags, data, distance, agent_data = _split_node_into_feature_groups(tree)

    for direction in TreeObsForRailEnv.tree_explored_actions_char:
        sub_flags, sub_data, sub_distance, sub_agent_data = _split_subtree_into_feature_groups(tree.childs[direction],
                                                                                               1,
                                                                                               max_tree_depth)
        flags = np.concatenate((flags, sub_flags))
        data = np.concatenate((data, sub_data))
        distance = np.concatenate((distance, sub_distance))
        agent_data = np.concatenate((agent_data, sub_agent_data))

    return flags, data, distance, agent_data


def normalize_observation(observation: Node, tree_depth: int, observation_radius=0,
                          normalize_fixed=None):
    """
    This function normalizes the observation used by the RL algorithm
    """
    flags, data, distance, agent_data = split_tree_into_feature_groups(observation, tree_depth)
    flags = np.clip(flags, -1, 1)
    data = norm_obs_clip(data, fixed_radius=observation_radius)
    if normalize_fixed is not None:
        distance = norm_obs_clip(distance, fixed_radius=normalize_fixed)
    else:
        distance = norm_obs_clip(distance, normalize_to_range=True)
    agent_data = norm_obs_clip(agent_data, fixed_radius=5)
    normalized_obs = np.concatenate((flags, np.concatenate((data, distance)), agent_data))
    return normalized_obs


def normalize_observation_with_agent_id(observation: Node, tree_depth: int, observation_radius=0,
                                        normalize_fixed=None, handle=0, num_agents=5):
    """
    This function normalizes the observation used by the RL algorithm
    """
    flags, data, distance, agent_data = split_tree_into_feature_groups(observation, tree_depth)
    flags = np.clip(flags, -1, 1)
    data = norm_obs_clip(data, fixed_radius=observation_radius)
    if normalize_fixed is not None:
        distance = norm_obs_clip(distance, fixed_radius=normalize_fixed)
    else:
        distance = norm_obs_clip(distance, normalize_to_range=True)
    agent_data = norm_obs_clip(agent_data, fixed_radius=num_agents)
    agent_one_hot = np.zeros(num_agents)
    agent_one_hot[handle % num_agents] = 1
    normalized_obs = np.concatenate((flags, np.concatenate((data, distance)), agent_data, agent_one_hot))
    return normalized_obs


class TreeObsForRailEnvRLLibWrapper(ObservationBuilder):

    def __init__(self, tree_obs_builder: TreeObsForRailEnv, normalize_fixed=None, concat_agent_id=False, num_agents=5):
        super().__init__()
        self._builder = tree_obs_builder
        self._normalize_fixed = normalize_fixed
        self._concat_agent_id = concat_agent_id
        self._num_agents = num_agents
        self.last_obs = None

    @property
    def observation_dim(self):
        return self._builder.observation_dim

    def reset(self):
        self._builder.reset()

    def get(self, handle: int = 0):
        self.last_obs = self._builder.get(handle)
        norm_obs = normalize_observation(self.last_obs, self._builder.max_depth, observation_radius=10,
                                         normalize_fixed=self._normalize_fixed) if self.last_obs is not None else self.last_obs
        if self._concat_agent_id:
            norm_obs = normalize_observation_with_agent_id(self.last_obs, self._builder.max_depth,
                                                           observation_radius=10,
                                                           normalize_fixed=self._normalize_fixed,
                                                           handle=handle,
                                                           num_agents=self._num_agents) if self.last_obs is not None else self.last_obs
        return norm_obs

    def get_many(self, handles: Optional[List[int]] = None):
        self.last_obs = self._builder.get_many(handles)
        if self._concat_agent_id:
            return {k: normalize_observation_with_agent_id(o, self._builder.max_depth, observation_radius=10,
                                                           normalize_fixed=self._normalize_fixed,
                                                           handle=k,
                                                           num_agents=self._num_agents)
                    for k, o in self.last_obs.items() if o is not None}
        else:
            return {k: normalize_observation(o, self._builder.max_depth, observation_radius=10,
                                             normalize_fixed=self._normalize_fixed)
                    for k, o in self.last_obs.items() if o is not None}

    def util_print_obs_subtree(self, tree):
        self._builder.util_print_obs_subtree(tree)

    def print_subtree(self, node, label, indent):
        self._builder.print_subtree(node, label, indent)

    def set_env(self, env):
        self._builder.set_env(env)
