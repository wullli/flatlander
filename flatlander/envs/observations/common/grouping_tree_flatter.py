from typing import Any, Optional

import numpy as np
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.envs.observations import TreeObsForRailEnv

from flatlander.envs.observations.common.tree_flatter import TreeFlattener
from flatlander.envs.observations.common.utils import norm_obs_clip


class GroupingTreeFlattener(TreeFlattener):

    def __init__(self, tree_depth=2, normalize_fixed=True, num_agents=5,
                 builder: Optional[ObservationBuilder] = None):
        self.tree_depth = tree_depth
        self.normalize_fixed = normalize_fixed
        self.num_agents = num_agents
        self.builder = builder

    @staticmethod
    def _split_node_into_feature_groups(node: Any) -> (np.ndarray, np.ndarray, np.ndarray):
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

        return data, distance, agent_data

    def _split_subtree_into_feature_groups(self, node: Any, current_tree_depth: int, max_tree_depth: int) -> (
            np.ndarray, np.ndarray, np.ndarray):
        if node == -np.inf:
            remaining_depth = max_tree_depth - current_tree_depth
            # reference: https://stackoverflow.com/questions/515214/total-number-of-nodes-in-a-tree-data-structure
            num_remaining_nodes = int((4 ** (remaining_depth + 1) - 1) / (4 - 1))
            return [-np.inf] * num_remaining_nodes * 6, [-np.inf] * num_remaining_nodes, [
                -np.inf] * num_remaining_nodes * 4

        data, distance, agent_data = self._split_node_into_feature_groups(node)

        if not node.childs:
            return data, distance, agent_data

        for direction in TreeObsForRailEnv.tree_explored_actions_char:
            sub_data, sub_distance, sub_agent_data = self._split_subtree_into_feature_groups(node.childs[direction],
                                                                                             current_tree_depth + 1,
                                                                                             max_tree_depth)
            data = np.concatenate((data, sub_data))
            distance = np.concatenate((distance, sub_distance))
            agent_data = np.concatenate((agent_data, sub_agent_data))

        return data, distance, agent_data

    def split_tree_into_feature_groups(self, tree: Any, max_tree_depth: int) -> (np.ndarray, np.ndarray, np.ndarray):
        """
        This function splits the tree into three difference arrays of values
        """
        data, distance, agent_data = self._split_node_into_feature_groups(tree)

        for direction in TreeObsForRailEnv.tree_explored_actions_char:
            sub_data, sub_distance, sub_agent_data = self._split_subtree_into_feature_groups(tree.childs[direction], 1,
                                                                                             max_tree_depth)
            data = np.concatenate((data, sub_data))
            distance = np.concatenate((distance, sub_distance))
            agent_data = np.concatenate((agent_data, sub_agent_data))

        return data, distance, agent_data

    def normalize_observation(self, observation: Any, tree_depth: int, observation_radius=0,
                              normalize_fixed=None):
        """
        This function normalizes the observation used by the RL algorithm
        """
        data, distance, agent_data = self.split_tree_into_feature_groups(observation, tree_depth)

        data = norm_obs_clip(data, fixed_radius=observation_radius)
        if normalize_fixed is not None:
            distance = norm_obs_clip(distance, fixed_radius=normalize_fixed)
        else:
            distance = norm_obs_clip(distance, normalize_to_range=True)
        agent_data = np.clip(agent_data, -1, 1)
        normalized_obs = np.concatenate((np.concatenate((data, distance)), agent_data))
        return normalized_obs

    def flatten(self, root: Any, handle, concat_agent_id, concat_status, **kwargs):

        obs = self.normalize_observation(observation=root,
                                         tree_depth=self.tree_depth,
                                         observation_radius=10,
                                         normalize_fixed=self.normalize_fixed)
        if concat_agent_id:
            agent_one_hot = np.zeros(self.num_agents)
            agent_one_hot[handle % self.num_agents] = 1
            obs = np.concatenate([obs, agent_one_hot])

        if concat_status:
            status_one_hot = np.zeros(4)
            status_one_hot[self.builder.env.agents[handle].status.value] = 1
            obs = np.concatenate([obs, status_one_hot])

        return obs
