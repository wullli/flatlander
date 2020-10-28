import numpy as np
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.observations import TreeObsForRailEnv, Node
from flatland.utils.ordered_set import OrderedSet


class DoneRemovedTreeObsForRailEnv(TreeObsForRailEnv):

    def get(self, handle: int = 0) -> Node:

        if handle > len(self.env.agents):
            print("ERROR: obs _get - handle ", handle, " len(agents)", len(self.env.agents))
        agent = self.env.agents[handle]

        if agent.status == RailAgentStatus.READY_TO_DEPART:
            agent_virtual_position = agent.initial_position
        elif agent.status == RailAgentStatus.ACTIVE:
            agent_virtual_position = agent.position
        elif agent.status == RailAgentStatus.DONE or agent.status == RailAgentStatus.DONE_REMOVED:
            agent_virtual_position = agent.target
        else:
            return None

        possible_transitions = self.env.rail.get_transitions(*agent_virtual_position, agent.direction)
        num_transitions = np.count_nonzero(possible_transitions)

        # Here information about the agent itself is stored
        distance_map = self.env.distance_map.get()

        # was referring to TreeObsForRailEnv.Node
        root_node_observation = Node(dist_own_target_encountered=0, dist_other_target_encountered=0,
                                     dist_other_agent_encountered=0, dist_potential_conflict=0,
                                     dist_unusable_switch=0, dist_to_next_branch=0,
                                     dist_min_to_target=distance_map[
                                         (handle, *agent_virtual_position,
                                          agent.direction)],
                                     num_agents_same_direction=0, num_agents_opposite_direction=0,
                                     num_agents_malfunctioning=agent.malfunction_data['malfunction'],
                                     speed_min_fractional=agent.speed_data['speed'],
                                     num_agents_ready_to_depart=0,
                                     childs={})
        # print("root node type:", type(root_node_observation))

        visited = OrderedSet()

        # Start from the current orientation, and see which transitions are available;
        # organize them as [left, forward, right, back], relative to the current orientation
        # If only one transition is possible, the tree is oriented with this transition as the forward branch.
        orientation = agent.direction

        if num_transitions == 1:
            orientation = np.argmax(possible_transitions)

        for i, branch_direction in enumerate([(orientation + i) % 4 for i in range(-1, 3)]):

            if possible_transitions[branch_direction]:
                new_cell = get_new_position(agent_virtual_position, branch_direction)

                branch_observation, branch_visited = \
                    self._explore_branch(handle, new_cell, branch_direction, 1, 1)
                root_node_observation.childs[self.tree_explored_actions_char[i]] = branch_observation

                visited |= branch_visited
            else:
                # add cells filled with infinity if no transition is possible
                root_node_observation.childs[self.tree_explored_actions_char[i]] = -np.inf
        self.env.dev_obs_dict[handle] = visited

        return root_node_observation

    def _reverse_dir(self, direction):
        return int((direction + 2) % 4) if direction is not None else None
