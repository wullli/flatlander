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
