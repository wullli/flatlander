from flatland.envs.rail_generators import RailGenerator, rail_from_grid_transition_map
from numpy.random.mtrand import RandomState

from flatlander.envs.flatland_sparse import FlatlandSparse


class FlatlandSparseStatic(FlatlandSparse):

    def get_rail_generator(self):
        rail_generator = super().get_rail_generator()
        rail = rail_generator(25,
                              25,
                              5, 0,
                              RandomState(self._config['seed']))[0]
        return rail_from_grid_transition_map(rail)
