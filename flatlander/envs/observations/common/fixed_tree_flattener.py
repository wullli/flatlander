from typing import Any

import numpy as np
from flatland.envs.rail_env import RailEnvActions

from flatlander.envs.observations.common.tree_flatter import TreeFlattener
from flatlander.envs.observations.common.utils import norm_obs_clip
from flatlander.envs.observations.fixed_tree_obs import FixedTreeObservation


class FixedTreeFlattener(TreeFlattener):
    _pos_dist_keys = ['dist_target', 'agent_position', 'agent_target']
    _num_agents_keys = ['malfunctions']
    _max_branch_length = 25

    def __init__(self, tree_depth=2, max_nr_nodes=None, observation_dim=None):
        super().__init__()
        self.tree_depth = tree_depth
        self.max_nr_nodes = max_nr_nodes
        self.observation_dim = observation_dim
        self._available_actions = [RailEnvActions.MOVE_FORWARD,
                                   RailEnvActions.DO_NOTHING,
                                   RailEnvActions.MOVE_LEFT,
                                   RailEnvActions.MOVE_RIGHT]

    @staticmethod
    def _get_node_features(node: Any) -> (np.ndarray, np.ndarray, np.ndarray):
        data = np.zeros(6)
        distance = np.zeros(1)
        agent_data = np.zeros(5)

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
        agent_data[3] = node.own_target_encountered
        agent_data[4] = node.shortest_path_direction

        data = norm_obs_clip(data, fixed_radius=10)
        distance = norm_obs_clip(distance, fixed_radius=100)
        agent_data = np.clip(agent_data, -1, 1)
        normalized_obs = np.concatenate([data, distance, agent_data])

        return normalized_obs

    def dfs(self, node: Any,
            node_observations: np.ndarray, current_level=1, abs_pos=0):
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
            elif current_level != self.tree_depth:
                abs_pos += self._count_missing_nodes(current_level + 1)

        node_obs = self._get_node_features(node)
        node_observations[abs_pos, :] = node_obs
        return abs_pos + 1

    def _count_missing_nodes(self, tree_level):
        missing_nodes = 0
        for i in range((self.tree_depth + 1) - tree_level):
            missing_nodes += np.power(len(self._available_actions), i)
        return missing_nodes

    def normalize_agent_info(self, agent_info):
        positions_distances = norm_obs_clip(np.concatenate([v for k, v in agent_info.items()
                                                            if k in self._pos_dist_keys]),
                                            fixed_radius=100)
        num_agents = np.clip(np.concatenate([v for k, v in agent_info.items()
                                             if k in self._num_agents_keys]), -1, 1)
        remaining = np.concatenate([v for k, v in agent_info.items()
                                    if k not in self._num_agents_keys and k not in self._pos_dist_keys])
        return np.concatenate([positions_distances, num_agents, remaining])

    def flatten(self, root: Any, agent_info=None):
        data = []
        for k, node in root.items():
            padded_observations = np.full(shape=(int(self.max_nr_nodes / len(root.values())), self.observation_dim,),
                                          fill_value=FixedTreeObservation.PAD_VALUE)
            if node != -np.inf:
                self.dfs(node, padded_observations)

            padded_observations = np.clip(padded_observations, -1, np.inf)
            assert not np.any(padded_observations == np.inf)
            data.append(padded_observations)

        data = np.concatenate(data)

        if agent_info is not None:
            agent_info = self.normalize_agent_info(agent_info=agent_info)
            return data, agent_info

        return data
