from collections import defaultdict
from copy import deepcopy
from typing import List, Callable, Optional, Dict

from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.agent_utils import RailAgentStatus
from flatlander.agents.agent import Agent
import numpy as np

from flatlander.submission.helper import get_agent_pos, is_done


class Phenotype:
    def __init__(self,
                 genotype: List[Dict[int, Optional[int]]] = None,
                 possible_mutations: List[Dict[int, np.ndarray]] = None,
                 departed: List[Dict[int, bool]] = None,
                 env: RailEnv = None,
                 initial_obs=None,
                 budget_function: Callable = None,
                 epsilon: float = 0.1,
                 default_behaviour: Agent = None):
        self.fitness = 0
        self.epsilon = epsilon
        self.initial_obs = initial_obs
        self.default_behaviour = default_behaviour

        if genotype is None:
            self.genotype = []
        else:
            self.genotype = genotype

        if possible_mutations is None:
            self.possible_mutations = []
        else:
            self.possible_mutations = possible_mutations

        if departed is None:
            self.departed = []
        else:
            self.departed = departed

        self.budget_function = budget_function
        self.env = env

    def mutate(self):

        partial_genotype = deepcopy(self.genotype)
        partial_possible_mutations = deepcopy(self.possible_mutations)
        partial_departed = deepcopy(self.departed)

        mutation_idx = np.random.choice(np.arange(0, len(self.genotype)))
        promising_mutations = self.promising_mutations(mutation_idx)

        while not len(promising_mutations) > 0:
            mutation_idx = np.random.choice(np.arange(0, len(self.genotype)))
            promising_mutations = self.promising_mutations(mutation_idx)

        promising_mutations_keys = np.array(list(promising_mutations.keys()))
        nr_mutations = np.random.choice(np.arange(1, len(promising_mutations_keys) + 1))
        submutation_indexes = np.random.choice(np.arange(0, len(promising_mutations_keys)),
                                               nr_mutations, replace=False)
        submutation_handles = promising_mutations_keys[submutation_indexes]

        for h in submutation_handles:
            next_possible_moves = self.possible_mutations[mutation_idx][h]
            possible_actions = set(np.flatnonzero(next_possible_moves))
            possible_actions = possible_actions.union((RailEnvActions.STOP_MOVING.value,
                                                       RailEnvActions.MOVE_FORWARD.value))
            non_default_actions = possible_actions.difference({self.genotype[mutation_idx][h]})
            partial_genotype[mutation_idx][h] = np.random.choice(list(non_default_actions))

        partial_genotype = partial_genotype[:mutation_idx + 1]
        partial_possible_mutations = partial_possible_mutations[:mutation_idx + 1]
        partial_departed = partial_departed[:mutation_idx + 1]

        return Phenotype(genotype=partial_genotype,
                         possible_mutations=partial_possible_mutations,
                         departed=partial_departed,
                         env=self.env,
                         default_behaviour=self.default_behaviour,
                         budget_function=self.budget_function,
                         initial_obs=self.initial_obs)

    def promising_mutations(self, mutation_idx) -> Dict[int, int]:
        """
        Return states that have more than one transition or are the beginning state!
        :param mutation_idx:
        :return:
        """
        return {h: a for h, a in self.genotype[mutation_idx].items()
                if np.count_nonzero(self.possible_mutations[mutation_idx][h]) > 1
                or not self.departed[mutation_idx][h]}

    def simulate(self):
        episode_return = 0
        dones = defaultdict(lambda: False)
        local_env = deepcopy(self.env)
        obs_dict = self.initial_obs

        for actions in self.genotype:
            if not self.budget_function():
                obs_dict, all_rewards, dones, info = local_env.step(actions)
                episode_return += np.sum(list(all_rewards))

        while not dones['__all__'] and not self.budget_function():
            actions: defaultdict[int, Optional[int]] = defaultdict(lambda: None,
                                                                   self.default_behaviour.compute_actions(
                                                                       obs_dict,
                                                                       env=local_env))
            transitions = defaultdict(lambda: None)
            agents_departed = defaultdict(lambda: True)

            for agent in local_env.agents:
                pos = get_agent_pos(agent)
                next_possible_moves = local_env.rail.get_transitions(*pos, agent.direction)

                actions = defaultdict(lambda: None, self.default_behaviour.compute_actions(obs_dict,
                                                                                           env=local_env))
                for agent in local_env.agents:
                    pos = get_agent_pos(agent)
                    next_possible_moves = local_env.rail.get_transitions(*pos, agent.direction)

                    if np.random.random() < self.epsilon:
                        possible_actions = set(np.flatnonzero(next_possible_moves))
                        possible_actions = possible_actions.union({RailEnvActions.STOP_MOVING.value,
                                                                   RailEnvActions.MOVE_FORWARD.value})
                        non_default_actions = possible_actions.difference({actions[agent.handle]})
                        actions[agent.handle] = np.random.choice(list(non_default_actions))

                transitions[agent.handle] = next_possible_moves
                agents_departed[agent.handle] = agent.status.value != RailAgentStatus.READY_TO_DEPART.value

            self.genotype.append(actions)
            self.possible_mutations.append(transitions)
            self.departed.append(agents_departed)
            obs_dict, all_rewards, dones, info = local_env.step(actions)
            episode_return += np.sum(list(all_rewards))

        if not self.budget_function():
            percentage_complete = np.sum(
                np.array([1 for a in local_env.agents if is_done(a)])) / local_env.get_num_agents()
            self.fitness = percentage_complete

            print(f"Simulation of candidate finished with fitness (PC): {percentage_complete}")
