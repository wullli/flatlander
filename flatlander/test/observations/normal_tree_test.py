import unittest

import numpy as np
import numpy.testing as npt

from flatlander.envs.observations.positional_tree_obs import PositionalTreeObservation
from flatlander.envs.observations.tree_obs import TreeObservation
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
        self.obs = TreeObservation({'max_depth': 2, 'shortest_path_max_depth': 30})
        self.obs._builder._builder = DummyBuilder(self.obs._builder._builder)

    def test_pos_encoding_root(self):
        obs = self.obs.builder().get(handle=0)
        print("obs", obs)
        print("len", len(obs))

