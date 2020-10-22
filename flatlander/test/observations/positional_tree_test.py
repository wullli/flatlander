import unittest

import numpy as np
import numpy.testing as npt

from flatlander.envs.observations.positional_tree_obs import PositionalTreeObservation
from flatlander.test.observations.dummy_tree_builder import DummyBuilder


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
        correct_encoding[0 + 1] = 1
        correct_encoding[4 + 3] = 1
        print("correct", correct_encoding)
        npt.assert_equal(pos_encodings[0], correct_encoding)

    def test_neg_inf_filling(self):
        obs_tuple = self.obs.builder().get(handle=0)
        pos_encodings = obs_tuple[1]
        print("encoded", pos_encodings[0])

        mask = pos_encodings[7:] == 0
        self.assertTrue(np.all(mask))
