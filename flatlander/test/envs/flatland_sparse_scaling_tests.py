import os
import unittest

import yaml

from flatlander.envs.flatland_sparse_scaling import FlatlandSparseScaling


class FlatlandSparseScalingTests(unittest.TestCase):

    def test_num_agents(self):
        with open(os.path.join(os.path.dirname(__file__), 'test_config.yaml')) as f:
            env_config = yaml.load(f)
        env = FlatlandSparseScaling(env_config)
        assert env.get_num_agents(1000) == 1
        assert env.get_num_agents(100001) == 2
        assert env.get_num_agents(474000) == 3
        assert env.get_num_agents(475001) == 4

    def test_max_agents(self):
        with open(os.path.join(os.path.dirname(__file__), 'test_config.yaml')) as f:
            env_config = yaml.load(f)
        env = FlatlandSparseScaling(env_config)
        assert env.get_num_agents(100000000000) == 10
