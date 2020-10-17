from typing import Any

from flatland.core.transition_map import GridTransitionMap
from flatland.envs.schedule_generators import SparseSchedGen
from flatland.envs.schedule_utils import Schedule
from numpy.random.mtrand import RandomState


class SequentialSparseSchedGen(SparseSchedGen):
    """

    This is the schedule generator which is used for Round 2 of the Flatland challenge. It produces schedules
    to railway networks provided by sparse_rail_generator.
    :param speed_ratio_map: Speed ratios of all agents. They are probabilities of all different speeds and have to
            add up to 1.
    :param seed: Initiate random seed generator
    """

    def generate(self, rail: GridTransitionMap,
                 num_agents: int,
                 hints: Any = None,
                 num_resets: int = 0,
                 np_random: RandomState = None) -> Schedule:
        """

        The generator that assigns tasks to all the agents
        :param rail: Rail infrastructure given by the rail_generator
        :param num_agents: Number of agents to include in the schedule
        :param hints: Hints provided by the rail_generator These include positions of start/target positions
        :param num_resets: How often the generator has been reset.
        :return: Returns the generator to the rail constructor
        """
        schedule = super().generate(rail, num_agents, hints, num_resets, np_random)
        schedule._replace(max_episode_steps=schedule.max_episode_steps * num_agents)
        return schedule
