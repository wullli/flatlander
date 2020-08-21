from flatland.envs.rail_generators import RailGenerator, rail_from_grid_transition_map
from numpy.random.mtrand import RandomState

from flatlander.envs.flatland_sparse import FlatlandSparse


def rail_from_grid_transition_map_with_hints(rail_map, hints):
    """
    Utility to convert a rail given by a GridTransitionMap map with the correct
    16-bit transitions specifications.

    Parameters
    ----------
    rail_map : GridTransitionMap object
        GridTransitionMap object to return when the generator is called.

    Returns
    -------
    function
        Generator function that always returns the given `rail_map` object.
    """

    def generator(width: int, height: int, num_agents: int, num_resets: int = 0,
                  np_random: RandomState = None):
        return rail_map, hints

    return generator


class FlatlandSparseStatic(FlatlandSparse):

    def get_rail_generator(self):
        rail_generator = super().get_rail_generator()
        rail, hints = rail_generator(25,
                                     25,
                                     5, 0,
                                     RandomState(self._config['seed']))
        return rail_from_grid_transition_map_with_hints(rail, hints)
