import unittest
from copy import deepcopy

import numpy as np
import numpy.testing as npt

from flatland.core.env_observation_builder import ObservationBuilder
from flatland.envs.observations import TreeObsForRailEnv
from flatlander.envs.observations.positional_tree_obs import PositionalTreeObservation


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
                                      childs={})
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
                                              childs={'R': deepcopy(node), 'S': deepcopy(node)})
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
                                      childs={'L': deepcopy(middle_nodes), 'R': deepcopy(middle_nodes)})
        return root

    def reset(self):
        pass


class PositionalTreeObservationTest(unittest.TestCase):

    """
        0: 'B',
        1: 'L',
        2: 'F',
        3: 'R',
        4: 'S',
    """

    def setUp(self) -> None:
        self.obs = PositionalTreeObservation({'max_depth': 2, 'shortest_path_max_depth': 30})
        self.obs._builder._builder = DummyBuilder(self.obs._builder._builder)

    def test_pos_encoding_root(self):
        obs_tuple = self.obs.builder().get(handle=0)
        pos_encodings = obs_tuple[1]
        print("encoded", pos_encodings[5])

        correct_encoding = np.zeros(self.obs._builder.positional_encoding_len)
        print("correct", correct_encoding)
        npt.assert_equal(pos_encodings[5], correct_encoding)

    def test_pos_encoding_leaf(self):
        obs_tuple = self.obs.builder().get(handle=0)
        pos_encodings = obs_tuple[1]
        print("encoded", pos_encodings[0])

        correct_encoding = np.zeros(self.obs._builder.positional_encoding_len)
        correct_encoding[0+1] = 1
        correct_encoding[4+3] = 1
        print("correct", correct_encoding)
        npt.assert_equal(pos_encodings[0], correct_encoding)

    def test_neg_inf_filling(self):
        obs_tuple = self.obs.builder().get(handle=0)
        pos_encodings = obs_tuple[1]
        print("encoded", pos_encodings[0])

        mask = pos_encodings[7:] == 0
        self.assertTrue(np.all(mask))
