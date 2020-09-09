import unittest

import numpy as np
import numpy.testing as npt

from flatlander.envs.observations.fixed_tree_obs import FixedTreeObservation
from flatlander.test.observations.dummy_tree_builder import DummyBuilder, DummyBuilderForward, \
    DummyBuilderForwardAlternative, DummyBuilderBackward


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

    def prep_obs_forward_alt(self):
        self.obs = FixedTreeObservation({'max_depth': 2, 'shortest_path_max_depth': 30})
        self.obs._builder._builder = DummyBuilderForwardAlternative(self.obs._builder._builder)

    def prep_obs_backward(self):
        self.obs = FixedTreeObservation({'max_depth': 2, 'shortest_path_max_depth': 30})
        self.obs._builder._builder = DummyBuilderBackward(self.obs._builder._builder)

    def test_root_position(self):
        self.prep_obs()
        obs = self.obs.builder().get(handle=0)
        print(obs)
        assert np.all(obs[-1] != -1)

    def test_leaf_position_forward_tree(self):
        self.prep_obs_forward()
        obs = self.obs.builder().get(handle=0)
        print(obs)
        assert np.all(obs[0] != -1)
        assert np.all(obs[1] != -1)
        assert np.all(obs[4] != -1)
        assert np.all(obs[15] != -1)
        assert np.all(obs[16] != -1)
        assert np.all(obs[19] != -1)
        assert np.all(obs[20] != -1)

    def test_leaf_position_fix_tree(self):
        self.prep_obs_forward_alt()
        obs = self.obs.builder().get(handle=0)
        print(obs)
        assert np.all(obs[0] != -1)

    def test_leaf_position_backward_tree(self):
        self.prep_obs_backward()
        obs = self.obs.builder().get(handle=0)
        print(obs)
        assert np.all(obs[1] != -1)


    def test_leaf_position_left_tree(self):
        self.prep_obs()
        obs = self.obs.builder().get(handle=0)
        print(obs)
        assert np.all(obs[13] != -1)
        assert np.all(obs[14] != -1)
        assert np.all(obs[18] != -1)
        assert np.all(obs[19] != -1)
        assert np.all(obs[20] != -1)

