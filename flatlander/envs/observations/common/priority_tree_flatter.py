from typing import Any

import numpy as np

from flatlander.envs.observations.builders.priority_tree import PriorityTreeObs
from flatlander.envs.observations.common.grouping_tree_flatter import GroupingTreeFlattener
from flatlander.envs.observations.common.utils import norm_obs_clip
from flatlander.envs.observations.common.utils import one_hot


class PriorityTreeFlattener(GroupingTreeFlattener):
    _pos_dist_keys = ['dist_target', 'agent_position', 'agent_target']
    _num_agents_keys = ['malfunctions']
    _max_branch_length = 25

    def __init__(self, tree_depth=2, normalize_fixed=True, num_agents=5):
        super().__init__(tree_depth, normalize_fixed, num_agents)
        self.tree_depth = tree_depth
        self.normalize_fixed = normalize_fixed
        self.num_agents = num_agents

    def _split_subtree_into_feature_groups(self, node: Any, current_tree_depth: int, max_tree_depth: int) -> (
            np.ndarray, np.ndarray, np.ndarray):
        if node == -np.inf:
            remaining_depth = max_tree_depth - current_tree_depth
            # reference: https://stackoverflow.com/questions/515214/total-number-of-nodes-in-a-tree-data-structure
            num_remaining_nodes = int((4 ** (remaining_depth + 1) - 1) / (4 - 1))
            return [-np.inf] * num_remaining_nodes * 6, [-np.inf] * num_remaining_nodes, [
                -np.inf] * num_remaining_nodes * 5

        data, distance, agent_data = self._split_node_into_feature_groups(node)

        if not node.childs:
            return data, distance, agent_data

        for direction in PriorityTreeObs.tree_explored_actions_char:
            sub_data, sub_distance, sub_agent_data = self._split_subtree_into_feature_groups(node.childs[direction],
                                                                                             current_tree_depth + 1,
                                                                                             max_tree_depth)
            data = np.concatenate((data, sub_data))
            distance = np.concatenate((distance, sub_distance))
            agent_data = np.concatenate((agent_data, sub_agent_data))

        return data, distance, agent_data

    @staticmethod
    def _split_node_into_feature_groups(node: Any) -> (np.ndarray, np.ndarray, np.ndarray):
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

        return data, distance, agent_data

    def normalize(self, data, distance, agent_data, observation_radius):
        """
        This function normalizes the observation used by the RL algorithm
        """

        data = norm_obs_clip(data, fixed_radius=observation_radius)
        if self.normalize_fixed is not None:
            distance = norm_obs_clip(distance, fixed_radius=self.normalize_fixed)
        else:
            distance = norm_obs_clip(distance, normalize_to_range=True)
        agent_data = np.clip(agent_data, -1, 1)
        normalized_obs = np.concatenate((np.concatenate((data, distance)), agent_data))
        return normalized_obs

    def normalize_with_agent_id(self, data, distance, agent_data,
                                observation_radius, handle=0):
        """
        This function normalizes the observation used by the RL algorithm
        """

        normalized_obs = self.normalize(data, distance, agent_data, observation_radius)
        agent_one_hot = one_hot(handle, self.num_agents)
        normalized_obs = np.concatenate([normalized_obs, agent_one_hot])

        return normalized_obs

    def normalize_agent_info(self, agent_info):
        positions_distances = norm_obs_clip(np.concatenate([v for k, v in agent_info.items()
                                                            if k in self._pos_dist_keys]),
                                            fixed_radius=self.normalize_fixed)
        num_agents = np.clip(np.concatenate([v for k, v in agent_info.items()
                                             if k in self._num_agents_keys]), -1, 1)
        remaining = np.concatenate([v for k, v in agent_info.items()
                                    if k not in self._num_agents_keys and k not in self._pos_dist_keys])
        return np.concatenate([positions_distances, num_agents, remaining])

    def flatten(self, root: Any, agent_info, handle, concat_agent_id, **kwargs):
        data = np.array([])
        distance = np.array([])
        agent_data = np.array([])
        for k, node in root.items():
            b_data, b_distance, b_agent_data = self._split_subtree_into_feature_groups(node=node,
                                                                                       current_tree_depth=1,
                                                                                       max_tree_depth=self.tree_depth)
            data = np.concatenate([data, b_data])
            distance = np.concatenate([distance, b_distance])
            agent_data = np.concatenate([agent_data, b_agent_data])

        if concat_agent_id:
            norm_obs = self.normalize_with_agent_id(data=data, distance=distance, agent_data=agent_data,
                                                    observation_radius=self._max_branch_length, handle=handle)
        else:
            norm_obs = self.normalize(data=data, distance=distance, agent_data=agent_data,
                                      observation_radius=self._max_branch_length)

        norm_agent_info = self.normalize_agent_info(agent_info=agent_info)
        obs = np.concatenate([norm_agent_info, norm_obs])
        return obs
