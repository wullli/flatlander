from collections import defaultdict

import numpy as np
from flatland.core.grid.grid_utils import coordinate_to_position
from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.rail_env import RailEnv

from flatlander.envs.observations.common.conflict_detector import ConflictDetector
from flatlander.envs.observations.common.utils import reverse_dir
from flatlander.envs.utils.shortest_path import get_shortest_paths


class Reservation:
    def __init__(self, handle, direction):
        self.handle = handle
        self.direction = direction


class TimelessConflictDetector(ConflictDetector):

    def detect_conflicts(self, handles=None, positions=None, directions=None) -> (defaultdict, defaultdict):
        pass

    def __init__(self, multi_shortest_path=False):
        super().__init__()
        self.reservations = {}
        self.multi_shortest_path = multi_shortest_path
        self.max_depth = 100

    def set_env(self, rail_env: RailEnv):
        self.rail_env = rail_env
        distance_map = self.rail_env.distance_map.get()
        max_agent_dist = np.max([distance_map[a.handle][a.initial_position + (a.initial_direction,)]
                                 for a in self.rail_env.agents])
        self.max_depth = int(max_agent_dist)

    def update(self):
        distance_map = self.rail_env.distance_map.get()
        max_agent_dist = np.max([distance_map[a.handle][a.initial_position + (a.initial_direction,)]
                                 for a in self.rail_env.agents])
        self.max_depth = int(max_agent_dist)

    def allowed_handles(self, handles=None, positions=None, directions=None):
        shortest_paths = get_shortest_paths(self.rail_env.distance_map, handles=handles, max_depth=self.max_depth)
        self.reservations = defaultdict(lambda: [])
        allowed_handles = []

        for h in handles:
            position = positions[h]
            direction = directions[h]
            shortest_path = shortest_paths[h]
            agent = self.rail_env.agents[h]
            times_per_cell = int(np.reciprocal(agent.speed_data["speed"]))
            if position is not None:
                allowed = True
                for index in range(1, self.max_depth + 1):

                    int_pos = coordinate_to_position(depth=self.rail_env.width, coords=[position])[0]

                    is_reserved = False
                    replaced_handles = []
                    for r in self.reservations[int_pos]:
                        cell_transitions = self.rail_env.rail.get_transitions(*position, direction)
                        if direction != r.direction and cell_transitions[reverse_dir(r.direction)] == 1:
                            if self.rail_env.agents[r.handle].status == RailAgentStatus.ACTIVE or \
                               self.rail_env.agents[r.handle].status == self.rail_env.agents[h].status:
                                is_reserved = True
                            else:
                                replaced_handles.append(r.handle)

                    if not is_reserved:
                        self.reservations[int_pos].append(Reservation(handle=h, direction=direction))
                        for r in replaced_handles:
                            if r in allowed_handles:
                                allowed_handles.remove(r)
                    else:
                        allowed = False

                    if position == agent.target:
                        break

                    if index % times_per_cell == 0:
                        position = shortest_path[0].position
                        direction = shortest_path[0].direction

                        shortest_path = shortest_path[1:]
                    
                if allowed:
                    allowed_handles.append(h)

        return allowed_handles
