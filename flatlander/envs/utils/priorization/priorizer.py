from abc import ABC, abstractmethod
from typing import List
import numpy as np
from flatland.envs.rail_env import RailEnv

from flatlander.envs.utils.priorization.helper import get_virtual_position


class Priorizer(ABC):

    @abstractmethod
    def priorize(self, handles: List[int], rail_env: RailEnv):
        raise NotImplementedError()


class DistToTargetPriorizer(Priorizer):
    def priorize(self, handles: List[int], rail_env: RailEnv):
        agent_distances = {}
        distance_map = rail_env.distance_map.get()
        nan_inf_mask = ((distance_map != np.inf) * (np.abs(np.isnan(distance_map) - 1))).astype(np.bool)
        max_distance = np.max(distance_map[nan_inf_mask])

        for h in handles:
            agent_virtual_position = get_virtual_position(rail_env.agents[h])
            if agent_virtual_position is not None:
                distance = distance_map[h][agent_virtual_position + (rail_env.agents[h].direction,)]
                distance = max_distance if (distance == np.inf or np.isnan(distance)) else distance
                agent_distances[h] = distance

        sorted_dists = {k: v for k, v in sorted(agent_distances.items(), key=lambda item: item[1])}
        return list(sorted_dists.keys())


class NrAgentsWaitingPriorizer(Priorizer):
    def priorize(self, handles: List[int], rail_env: RailEnv):
        agents_waiting = {}
        for h in handles:
            agent = rail_env.agents[h]

            agents_same_start = [a for a in rail_env.agents
                                 if a.initial_position == agent.initial_position]
            nr_agents_same_start = len(agents_same_start)
            agents_waiting[h] = nr_agents_same_start

        sorted_dists = {k: v for k, v in sorted(agents_waiting.items(), key=lambda item: item[1], reverse=True)}
        return list(sorted_dists.keys())


class NrAgentsSameStart(Priorizer):
    def priorize(self, handles: List[int], rail_env: RailEnv):
        agents_waiting = {}
        for h in handles:
            agent = rail_env.agents[h]

            nr_agents_same_start_and_dir = len([a.handle for a in rail_env.agents
                                                if a.initial_direction == agent.initial_direction
                                                and a.initial_position == agent.initial_position])
            agents_waiting[h] = nr_agents_same_start_and_dir

        sorted_dists = {k: v for k, v in sorted(agents_waiting.items(), key=lambda item: item[1], reverse=True)}
        return list(sorted_dists.keys())
