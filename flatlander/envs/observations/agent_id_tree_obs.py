from typing import Optional, List

import gym
import numpy as np

from flatland.core.env_observation_builder import ObservationBuilder
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatlander.envs.observations import Observation, register_obs
from flatlander.envs.observations.builders.agent_id_tree import AgentIdTreeObservationBuilder, AgentIdNode
from flatlander.envs.observations.utils import norm_obs_clip


@register_obs("agent_id_tree")
class AgentIdTreeObservation(Observation):

    def __init__(self, config) -> None:
        super().__init__(config)
        self._concat_agent_id = config.get('concat_agent_id', False)
        self._max_n_agents = config.get('max_n_agents', 10)
        self._builder = AgentIdTreeObsWrapper(
            AgentIdTreeObservationBuilder(
                max_depth=config['max_depth'],
                predictor=ShortestPathPredictorForRailEnv(config['shortest_path_max_depth']),
                max_n_agents=self._max_n_agents
            ),
            config.get('normalize_fixed', None),
            self._concat_agent_id,
            self._max_n_agents
        )

    def builder(self) -> ObservationBuilder:
        return self._builder

    def observation_space(self) -> gym.Space:
        num_features_per_node = self._builder.observation_dim
        nr_nodes = 0
        for i in range(self.config['max_depth'] + 1):
            nr_nodes += np.power(4, i)
        dim = num_features_per_node * nr_nodes
        if self._concat_agent_id:
            dim += self._max_n_agents
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(dim,))


class AgentIdTreeObsWrapper(ObservationBuilder):

    def __init__(self, tree_obs_builder: AgentIdTreeObservationBuilder, normalize_fixed=None, concat_agent_id=False,
                 num_agents=5):
        super().__init__()
        self._builder = tree_obs_builder
        self._normalize_fixed = normalize_fixed
        self._concat_agent_id = concat_agent_id
        self._max_n_agents = num_agents

    @property
    def observation_dim(self):
        return self._builder.observation_dim

    def reset(self):
        self._builder.reset()

    def get(self, handle: int = 0):
        obs = self._builder.get(handle)
        norm_obs = self.normalize_observation(obs, self._builder.max_depth, observation_radius=10,
                                              normalize_fixed=self._normalize_fixed) if obs is not None else obs
        if self._concat_agent_id:
            norm_obs = self.normalize_observation(obs, self._builder.max_depth, observation_radius=10,
                                                  normalize_fixed=self._normalize_fixed,
                                                  handle=handle) if obs is not None else obs
        return norm_obs

    def get_many(self, handles: Optional[List[int]] = None):
        return {k: self.normalize_observation(o, self._builder.max_depth, observation_radius=10,
                                              normalize_fixed=self._normalize_fixed,
                                              handle=k)
                for k, o in self._builder.get_many(handles).items() if o is not None}

    def util_print_obs_subtree(self, tree):
        self._builder.util_print_obs_subtree(tree)

    def print_subtree(self, node, label, indent):
        self._builder.print_subtree(node, label, indent)

    def set_env(self, env):
        self._builder.set_env(env)

    def _split_node_into_feature_groups(self, node: AgentIdNode) -> (np.ndarray, np.ndarray, np.ndarray):
        data = np.zeros(6)
        distance = np.zeros(1)
        agent_data = np.zeros(13)

        data[0] = node.dist_own_target_encountered
        data[1] = node.dist_other_target_encountered
        data[2] = node.dist_other_agent_encountered
        data[3] = node.dist_potential_conflict
        data[3] = node.dist_potential_conflict
        data[4] = node.dist_unusable_switch
        data[5] = node.dist_to_next_branch

        distance[0] = node.dist_min_to_target

        agent_data[0] = node.num_agents_same_direction
        agent_data[1] = node.num_agents_opposite_direction
        agent_data[2] = node.num_agents_malfunctioning
        agent_data[3] = node.speed_min_fractional
        agent_data[4:4 + self._max_n_agents] = node.handle_potential_conflict
        agent_data[4 + self._max_n_agents:] = node.available_transitions

        return data, distance, agent_data

    def _split_subtree_into_feature_groups(self, node: AgentIdNode,
                                           current_tree_depth: int,
                                           max_tree_depth: int) -> (np.ndarray, np.ndarray, np.ndarray):
        if node == -np.inf:
            remaining_depth = max_tree_depth - current_tree_depth
            num_remaining_nodes = int((4 ** (remaining_depth + 1) - 1) / (4 - 1))
            return [-np.inf] * num_remaining_nodes * 6, [-np.inf] * num_remaining_nodes, [
                -np.inf] * num_remaining_nodes * 13

        data, distance, agent_data = self._split_node_into_feature_groups(node)

        if not node.childs:
            return data, distance, agent_data

        for direction in AgentIdTreeObservationBuilder.tree_explored_actions_char:
            sub_data, sub_distance, sub_agent_data = self._split_subtree_into_feature_groups(node.childs[direction],
                                                                                             current_tree_depth + 1,
                                                                                             max_tree_depth)
            data = np.concatenate((data, sub_data))
            distance = np.concatenate((distance, sub_distance))
            agent_data = np.concatenate((agent_data, sub_agent_data))

        return data, distance, agent_data

    def split_tree_into_feature_groups(self, tree: AgentIdNode, max_tree_depth: int) -> (np.ndarray, np.ndarray, np.ndarray):
        """
        This function splits the tree into three difference arrays of values
        """
        data, distance, agent_data = self._split_node_into_feature_groups(tree)

        for direction in AgentIdTreeObservationBuilder.tree_explored_actions_char:
            sub_data, sub_distance, sub_agent_data = self._split_subtree_into_feature_groups(tree.childs[direction], 1,
                                                                                             max_tree_depth)
            data = np.concatenate((data, sub_data))
            distance = np.concatenate((distance, sub_distance))
            agent_data = np.concatenate((agent_data, sub_agent_data))

        return data, distance, agent_data

    def normalize_observation(self, observation: AgentIdNode, tree_depth: int, observation_radius=0,
                              normalize_fixed=None, handle=0):
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
        agent_one_hot = np.zeros(self._max_n_agents)
        agent_one_hot[handle % self._max_n_agents] = 1
        normalized_obs = np.concatenate((np.concatenate((data, distance)), agent_data, agent_one_hot))
        return normalized_obs
