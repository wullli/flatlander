from time import time

from flatland.envs.rail_env import RailEnv

from flatlander.agents.agent import Agent
from flatlander.planning.genetic.phenotype import Phenotype
import numpy as np


class Population:
    def __init__(self, nr_evolutions_per_step=2,
                 default_behaviour: Agent = None,
                 env: RailEnv = None,
                 initial_obs=None,
                 budget_seconds=60,
                 desired_fitness=1.0):
        self.start_t = time()
        self.default_behaviour = default_behaviour
        self.env = env
        self.desired_fitness = desired_fitness
        self.initial_obs = initial_obs
        self.budget_seconds = budget_seconds
        self.ancestor = Phenotype(env=env,
                                  initial_obs=self.initial_obs,
                                  budget_function=self.budget_used,
                                  default_behaviour=default_behaviour)
        self.ancestor.simulate()
        self.phenotypes = [self.ancestor]
        self.nr_evolutions_per_step = nr_evolutions_per_step

    def evolve(self):
        self.phenotypes.sort(key=lambda pt: pt.fitness)

        evolutions = 0
        while not self.budget_used() and evolutions < self.nr_evolutions_per_step:
            idx = min(evolutions + 1, len(self.phenotypes))
            candidate = self.phenotypes[-idx].mutate()
            candidate.simulate()
            self.phenotypes.append(candidate)
            if self.evolution_complete():
                print("Desired fitness level reached!")
                return

    def evolution_complete(self):
        return self.best_phenotype().fitness >= self.desired_fitness

    def budget_used(self):
        return (time() - self.start_t) > self.budget_seconds

    def best_phenotype(self) -> Phenotype:
        return max(self.phenotypes, key=lambda pt: pt.fitness)

    def worst_phenotype(self) -> Phenotype:
        return min(self.phenotypes, key=lambda pt: pt.fitness)

    def mean_fitness(self) -> float:
        return float(np.mean([pt.fitness for pt in self.phenotypes]))
