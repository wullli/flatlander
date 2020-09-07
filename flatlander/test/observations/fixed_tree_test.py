import unittest

import numpy as np
import numpy.testing as npt

from flatlander.envs.observations.fixed_tree_obs import FixedTreeObservation
from flatlander.test.observations.dummy_tree_builder import DummyBuilder, DummyBuilderForward


class FixedTreeObservationTest(unittest.TestCase):
    """
        0: 'B',
        1: 'L',
        2: 'F',
        3: 'R',
        4: 'S',
    """

    def prep_obs(self):
        self.obs = FixedTreeObservation({'max_depth': 2, 'shortest_path_max_depth': 30})
        self.obs._builder._builder = DummyBuilder(self.obs._builder._builder)

    def prep_obs_forward(self):
        self.obs = FixedTreeObservation({'max_depth': 2, 'shortest_path_max_depth': 30})
        self.obs._builder._builder = DummyBuilderForward(self.obs._builder._builder)

    def test_root_position(self):
        self.prep_obs()
        obs = self.obs.builder().get(handle=0)
        print(obs)
        assert np.all(obs[-1] != -1)

    def test__leaf_position(self):
        self.prep_obs_forward()
        obs = self.obs.builder().get(handle=0)
        print(obs)
        assert np.all(obs[0] != -np.inf)
        assert np.all(obs[1] != -np.inf)
        assert np.all(obs[4] != -np.inf)
        assert np.all(obs[5] != -np.inf)

