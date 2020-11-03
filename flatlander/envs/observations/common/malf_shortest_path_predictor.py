from copy import deepcopy

from flatland.core.env_prediction_builder import PredictionBuilder
from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.distance_map import DistanceMap
from flatland.envs.rail_env import RailEnvActions
import numpy as np
from flatland.utils.ordered_set import OrderedSet

from flatlander.envs.utils.shortest_path import get_shortest_paths


class MalfShortestPathPredictorForRailEnv(PredictionBuilder):
    """
    ShortestPathPredictorForRailEnv object.

    This object returns shortest-path predictions for agents in the RailEnv environment.
    The prediction acts as if no other agent is in the environment and always takes the forward action.
    """

    def __init__(self, max_depth: int = 20, branch_only=False):
        super().__init__(max_depth)
        self.branch_only = branch_only

    def get(self, handle: int = None, handles=None, positions=None, directions=None):
        """
        Called whenever get_many in the observation build is called.
        Requires distance_map to extract the shortest path.
        Does not take into account future positions of other agents!

        If there is no shortest path, the agent just stands still and stops moving.

        Parameters
        ----------
        handle : int, optional
            Handle of the agent for which to compute the observation vector.

        Returns
        -------
        np.array
            Returns a dictionary indexed by the agent handle and for each agent a vector of (max_depth + 1)x5 elements:
            - time_offset
            - position axis 0
            - position axis 1
            - direction
            - action taken to come here (not implemented yet)
            The prediction at 0 is the current position, direction etc.
        """
        agents = self.env.agents
        if handle:
            agents = [self.env.agents[handle]]

        if handles is not None:
            agents = [self.env.agents[h] for h in handles]

        distance_map: DistanceMap = self.env.distance_map

        shortest_paths = get_shortest_paths(distance_map, handles=handles, max_depth=self.max_depth,
                                            branch_only=self.branch_only)

        prediction_dict = {}

        agents = deepcopy(agents)
        for agent in agents:

            if not agent.status == RailAgentStatus.ACTIVE and not agent.status == RailAgentStatus.READY_TO_DEPART:
                prediction = np.zeros(shape=(self.max_depth + 1, 5))
                for i in range(self.max_depth):
                    prediction[i] = [i, None, None, None, None]
                prediction_dict[agent.handle] = prediction
                continue

            agent_virtual_direction = agent.direction
            agent_virtual_position = agent.position
            if agent.status == RailAgentStatus.READY_TO_DEPART:
                agent_virtual_position = agent.initial_position

            agent_speed = agent.speed_data["speed"]
            times_per_cell = int(np.reciprocal(agent_speed))
            prediction = np.zeros(shape=(self.max_depth + 1, 5))

            prediction[0] = [0, *agent_virtual_position, agent_virtual_direction, 0]

            shortest_path = shortest_paths[agent.handle]

            # if there is a shortest path, remove the initial position
            if shortest_path:
                shortest_path = shortest_path[1:]

            if positions is not None and positions.get(agent.handle, None) is not None:
                new_direction = directions[agent.handle]
                new_position = positions[agent.handle]
            else:
                new_direction = agent_virtual_direction
                new_position = agent_virtual_position
            visited = OrderedSet()
            for index in range(1, self.max_depth + 1):

                if self.branch_only:
                    cell_transitions = self.env.rail.get_transitions(*new_position, new_direction)
                    if np.count_nonzero(cell_transitions) > 1:
                        break

                if not shortest_path:
                    prediction[index] = [index, *new_position, new_direction, RailEnvActions.STOP_MOVING]
                    visited.add((*new_position, agent.direction))
                    continue

                if agent.malfunction_data["malfunction"] > 0:
                    agent.malfunction_data["malfunction"] -= 1
                    prediction[index] = [index, *new_position, None, RailEnvActions.STOP_MOVING]
                    visited.add((*new_position, agent.direction))
                    continue

                if new_position == agent.target:
                    prediction[index] = [index, *new_position, new_direction, RailEnvActions.STOP_MOVING]
                    visited.add((*new_position, agent.direction))
                    break

                if index % times_per_cell == 0:
                    new_position = shortest_path[0].position
                    new_direction = shortest_path[0].direction

                    shortest_path = shortest_path[1:]

                # prediction is ready
                prediction[index] = [index, *new_position, new_direction, 0]
                visited.add((*new_position, new_direction))

            self.env.dev_pred_dict[agent.handle] = visited
            prediction_dict[agent.handle] = prediction

        return prediction_dict
