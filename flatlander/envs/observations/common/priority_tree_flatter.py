from typing import Any

import numpy as np

from flatlander.envs.observations.common.grouping_tree_flatter import GroupingTreeFlattener
from flatlander.envs.observations.common.utils import norm_obs_clip


class PriorityTreeFlattener(GroupingTreeFlattener):

    def __init__(self, tree_depth=2, normalize_fixed=True, num_agents=5):
        super().__init__(tree_depth, normalize_fixed, num_agents)
        self.tree_depth = tree_depth
        self.normalize_fixed = normalize_fixed
        self.num_agents = num_agents

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

        data = norm_obs_clip(data, fixed_radius=observation_radius)
        if self.normalize_fixed is not None:
            distance = norm_obs_clip(distance, fixed_radius=self.normalize_fixed)
        else:
            distance = norm_obs_clip(distance, normalize_to_range=True)
        agent_data = np.clip(agent_data, -1, 1)
        agent_one_hot = np.zeros(self.num_agents)
        agent_one_hot[handle % self.num_agents] = 1
        normalized_obs = np.concatenate((np.concatenate((data, distance)), agent_data, agent_one_hot))
        return normalized_obs

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
                                                    observation_radius=10, handle=handle)
        else:
            norm_obs = self.normalize(data=data, distance=distance, agent_data=agent_data,
                                      observation_radius=10)

        obs = np.concatenate([agent_info, norm_obs])
        return obs
