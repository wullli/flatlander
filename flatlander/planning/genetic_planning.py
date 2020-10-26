from flatland.envs.rail_env import RailEnv

from flatlander.agents.heuristic_agent import HeuristicPriorityAgent
from flatlander.planning.genetic.population import Population


def genetic_plan(env: RailEnv, obs_dict, budget_seconds=60, policy_agent=HeuristicPriorityAgent()):
    pop = Population(nr_evolutions_per_step=2,
                     default_behaviour=policy_agent,
                     env=env,
                     budget_seconds=budget_seconds,
                     initial_obs=obs_dict)

    while not pop.budget_used() and not pop.evolution_complete():
        pop.evolve()

    best_phenotype = pop.best_phenotype()
    print(f'MAX PC: {best_phenotype.fitness}, MIN PC: {pop.worst_phenotype().fitness}, MEAN PC: {pop.mean_fitness()}')

    return best_phenotype.genotype
