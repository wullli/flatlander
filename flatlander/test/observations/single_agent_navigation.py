import time

import numpy as np
from flatland.core.grid.grid4_utils import get_new_position

from flatland.envs.agent_utils import RailAgentStatus

from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import random_rail_generator, complex_rail_generator, sparse_rail_generator
from flatland.utils.rendertools import RenderTool


class SingleAgentNavigationObs(TreeObsForRailEnv):
    """
    We derive our observation builder from TreeObsForRailEnv, to exploit the existing implementation to compute
    the minimum distances from each grid node to each agent's target.

    We then build a representation vector with 3 binary components, indicating which of the 3 available directions
    for each agent (Left, Forward, Right) lead to the shortest path to its target.
    E.g., if taking the Left branch (if available) is the shortest route to the agent's target, the observation vector
    will be [1, 0, 0].
    """

    def __init__(self):
        super().__init__(max_depth=0)
        # We set max_depth=0 in because we only need to look at the current
        # position of the agent to decide what direction is shortest.

    def reset(self):
        # Recompute the distance map, if the environment has changed.
        super().reset()

    def get(self, handle):
        # Here we access agent information from the environment.
        # Information from the environment can be accessed but not changed!

        self.env: RailEnv = self.env
        agent = self.env.agents[handle]
        if agent.status == RailAgentStatus.READY_TO_DEPART:
            agent_virtual_position = agent.initial_position
        elif agent.status == RailAgentStatus.ACTIVE:
            agent_virtual_position = agent.position
        elif agent.status == RailAgentStatus.DONE:
            agent_virtual_position = agent.target
        else:
            return None

        agent = self.env.agents[handle]

        possible_transitions = self.env.rail.get_transitions(*agent_virtual_position, agent.direction)
        num_transitions = np.count_nonzero(possible_transitions)

        # Start from the current orientation, and see which transitions are available;
        # organize them as [left, forward, right], relative to the current orientation
        # If only one transition is possible, the forward branch is aligned with it.
        if num_transitions == 1:
            observation = [0, 1, 0]
        else:
            min_distances = []
            for direction in [(agent.direction + i) % 4 for i in range(-1, 2)]:
                if possible_transitions[direction]:
                    new_position = get_new_position(agent.position, direction)
                    min_distances.append(
                        self.env.distance_map.get()[handle, new_position[0], new_position[1], direction])
                else:
                    min_distances.append(np.inf)

            observation = [0, 0, 0]
            observation[np.argmin(min_distances)] = 1

        return observation


n_agents = 5
env = RailEnv(width=25, height=25,
              rail_generator=sparse_rail_generator(),
              number_of_agents=n_agents,
              obs_builder_object=SingleAgentNavigationObs())

env_renderer = None

for _ in range(100):
    if env_renderer is not None:
        env_renderer.close_window()
    obs, _ = env.reset()
    env_renderer = RenderTool(env)
    env_renderer.render_env(show=True, frames=True, show_observations=False)
    done = {"__all__": False}
    while not done["__all__"]:
        action = {i: np.argmax(o) for i, o in obs.items()}
        obs, all_rewards, done, _ = env.step(action)
        print("Rewards: ", all_rewards, "  [done=", done, "]")
        print("Observations: ", obs)
        assert len(obs.keys()) == n_agents

        env_renderer.render_env(show=True, frames=True, show_observations=False)
        time.sleep(0.1)
