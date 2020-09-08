from copy import deepcopy

import numpy as np

from flatland.core.env_observation_builder import ObservationBuilder
from flatland.envs.observations import TreeObsForRailEnv


class DummyBuilder(ObservationBuilder):

    def __init__(self, org_builder):
        super().__init__()
        self.org_builder = org_builder

    def __getattr__(self, attr):
        return getattr(self.org_builder, attr)

    def get(self, handle: int = 0):
        node = TreeObsForRailEnv.Node(dist_own_target_encountered=0,
                                      dist_other_target_encountered=0,
                                      dist_other_agent_encountered=0,
                                      dist_potential_conflict=0,
                                      dist_unusable_switch=0,
                                      dist_to_next_branch=0,
                                      dist_min_to_target=0,
                                      num_agents_same_direction=0,
                                      num_agents_opposite_direction=0,
                                      num_agents_malfunctioning=0,
                                      speed_min_fractional=0,
                                      num_agents_ready_to_depart=0,
                                      childs={'R': -np.inf,
                                              'S': -np.inf,
                                              'L': -np.inf,
                                              'B': -np.inf,
                                              'F': -np.inf
                                              })
        middle_nodes = TreeObsForRailEnv.Node(dist_own_target_encountered=0,
                                              dist_other_target_encountered=0,
                                              dist_other_agent_encountered=0,
                                              dist_potential_conflict=0,
                                              dist_unusable_switch=0,
                                              dist_to_next_branch=0,
                                              dist_min_to_target=0,
                                              num_agents_same_direction=0,
                                              num_agents_opposite_direction=0,
                                              num_agents_malfunctioning=0,
                                              speed_min_fractional=0,
                                              num_agents_ready_to_depart=0,
                                              childs={'R': deepcopy(node),
                                                      'S': deepcopy(node),
                                                      'L': -np.inf,
                                                      'B': -np.inf,
                                                      'F': -np.inf
                                                      })
        root = TreeObsForRailEnv.Node(dist_own_target_encountered=0,
                                      dist_other_target_encountered=0,
                                      dist_other_agent_encountered=0,
                                      dist_potential_conflict=0,
                                      dist_unusable_switch=0,
                                      dist_to_next_branch=0,
                                      dist_min_to_target=0,
                                      num_agents_same_direction=0,
                                      num_agents_opposite_direction=0,
                                      num_agents_malfunctioning=0,
                                      speed_min_fractional=0,
                                      num_agents_ready_to_depart=0,
                                      childs={'L': deepcopy(middle_nodes),
                                              'R': deepcopy(middle_nodes),
                                              'S': -np.inf,
                                              'B': -np.inf,
                                              'F': -np.inf})
        return root

    def reset(self):
        pass


class DummyBuilderForward(ObservationBuilder):

    def __init__(self, org_builder):
        super().__init__()
        self.org_builder = org_builder

    def __getattr__(self, attr):
        return getattr(self.org_builder, attr)

    def get(self, handle: int = 0):
        node = TreeObsForRailEnv.Node(dist_own_target_encountered=0,
                                      dist_other_target_encountered=0,
                                      dist_other_agent_encountered=0,
                                      dist_potential_conflict=10,
                                      dist_unusable_switch=10,
                                      dist_to_next_branch=10,
                                      dist_min_to_target=10,
                                      num_agents_same_direction=0,
                                      num_agents_opposite_direction=0,
                                      num_agents_malfunctioning=0,
                                      speed_min_fractional=0,
                                      num_agents_ready_to_depart=0,
                                      childs={'L': -np.inf,
                                              'R': -np.inf,
                                              'S': -np.inf,
                                              'B': -np.inf,
                                              'F': -np.inf})
        middle_nodes = TreeObsForRailEnv.Node(dist_own_target_encountered=0,
                                              dist_other_target_encountered=0,
                                              dist_other_agent_encountered=0,
                                              dist_potential_conflict=5,
                                              dist_unusable_switch=5,
                                              dist_to_next_branch=5,
                                              dist_min_to_target=5,
                                              num_agents_same_direction=0,
                                              num_agents_opposite_direction=0,
                                              num_agents_malfunctioning=0,
                                              speed_min_fractional=0,
                                              num_agents_ready_to_depart=0,
                                              childs={'F': deepcopy(node),
                                                      'B': deepcopy(node),
                                                      'S': -np.inf,
                                                      'R': -np.inf,
                                                      'L': -np.inf})
        root = TreeObsForRailEnv.Node(dist_own_target_encountered=0,
                                      dist_other_target_encountered=0,
                                      dist_other_agent_encountered=0,
                                      dist_potential_conflict=0,
                                      dist_unusable_switch=0,
                                      dist_to_next_branch=0,
                                      dist_min_to_target=0,
                                      num_agents_same_direction=0,
                                      num_agents_opposite_direction=0,
                                      num_agents_malfunctioning=0,
                                      speed_min_fractional=0,
                                      num_agents_ready_to_depart=0,
                                      childs={'F': deepcopy(middle_nodes),
                                              'R': deepcopy(middle_nodes),
                                              'S': -np.inf,
                                              'B': -np.inf,
                                              'L': -np.inf})
        return root

    def reset(self):
        pass
