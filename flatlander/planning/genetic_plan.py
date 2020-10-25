from collections import defaultdict
from copy import deepcopy
from time import time
from typing import List, Any, Optional

import numpy as np
from flatland.envs.rail_env import RailEnv, RailEnvActions

from flatlander.agents.heuristic_agent import HeuristicPriorityAgent
from flatlander.utils.helper import get_agent_pos, is_done


class Phenotype:
    def __init__(self, fitness: float,
                 genotype: List[defaultdict[int, Optional[int]]],
                 possible_mutations=List[np.ndarray],
                 epsilon=0.1):
        self.fitness = fitness
        self.genotype = genotype
        self.possible_mutations = possible_mutations
        self.epsilon = epsilon

    def mutate(self):
        raise NotImplementedError()


def genetic_plan(env: RailEnv, obs_dict, budget_seconds=60, epsilon=0.1,
                 policy_agent=HeuristicPriorityAgent()):
    start_t = time()
    best_actions = []
    best_return = -np.inf
    best_pc = -np.inf
    all_returns = []
    all_pcs = []
    plan_step = 0
    budget_used = False

    phenotypes = []

    while not budget_used:
        local_env = deepcopy(env)
        episode_return = 0
        action_memory = []
        transitions_memory = []
        dones = defaultdict(lambda: False)
        print(f'\nPlanning step {plan_step + 1}')

        while not dones['__all__'] and not budget_used:
            actions: defaultdict[int, Optional[int]] = defaultdict(lambda: None, policy_agent.compute_actions(obs_dict,
                                                                                                              env=local_env))
            transitions = defaultdict[int, Optional[np.ndarray]]
            for agent in env.agents:
                pos = get_agent_pos(agent)
                next_possible_moves = local_env.rail.get_transitions(*pos, agent.direction)
                transitions[agent.handle] = next_possible_moves

                if np.random.random() < epsilon and plan_step != 0:
                    possible_actions = set(np.flatnonzero(next_possible_moves))
                    possible_actions = possible_actions.union({RailEnvActions.STOP_MOVING.value,
                                                               RailEnvActions.MOVE_FORWARD.value})
                    non_default_actions = possible_actions.difference({actions[agent.handle]})
                    actions[agent.handle] = np.random.choice(list(non_default_actions))

            action_memory.append(actions)
            transitions_memory.append(transitions)
            obs_dict, all_rewards, dones, info = local_env.step(actions)
            episode_return += np.sum(list(all_rewards))
            budget_used = (time() - start_t) > budget_seconds

        if not budget_used:
            all_returns.append(episode_return)
            pc = np.sum(np.array([1 for a in local_env.agents if is_done(a)])) / local_env.get_num_agents()
            all_pcs.append(pc)

            phenotypes.append(Phenotype(fitness=pc,
                                        genotype=action_memory,
                                        possible_mutations=transitions_memory))

            if pc > best_pc:
                best_return = episode_return
                best_pc = pc
                best_actions = action_memory

            if pc == 1.0:
                print(f'MAX PC: {best_pc}, MIN PC: {np.min(all_pcs)}, MAX RETURN: {best_return}\n')
                return best_actions

            plan_step += 1

    if len(all_pcs) > 0:
        print(f'MAX PC: {best_pc}, MIN PC: {np.min(all_pcs)}, MAX RETURN: {best_return}\n')
    else:
        print(f'Budget reached before any planning step could finish!')
    return best_actions if len(best_actions) > 0 else None
