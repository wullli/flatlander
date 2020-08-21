import unittest

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import rail_from_manual_specifications_generator, rail_from_grid_transition_map, \
    sparse_rail_generator
from numpy.random.mtrand import RandomState


class RailEnvTest(unittest.TestCase):

    def test_static_env(self):
        rail_generator = sparse_rail_generator(
            seed=42,
            max_num_cities=3,
            grid_mode=True,
            max_rails_between_cities=4,
            max_rails_in_city=4
        )
        rail = rail_generator(25,
                              25,
                              5, 0, RandomState(None))[0]
        rail_from_grid_transition_map(rail)


if __name__ == '__main__':
    unittest.main()
