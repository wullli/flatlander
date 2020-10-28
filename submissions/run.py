import os
from collections import defaultdict

from tqdm import tqdm

from flatlander.agents.rllib_agent import RllibAgent
from flatlander.agents.shortest_path_agent import ShortestPathAgent
from flatlander.envs.utils.robust_gym_env import RobustFlatlandGymEnv
from flatlander.planning.epsilon_greedy_planning import epsilon_greedy_plan
from flatlander.planning.genetic_planning import genetic_plan

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

from flatland.evaluators.client import FlatlandRemoteClient, TimeoutException
from flatlander.envs.observations import make_obs
from flatlander.utils.helper import episode_start_info, episode_end_info, init_run, get_agent
from timeit import default_timer as timer

tf.compat.v1.disable_eager_execution()
remote_client = FlatlandRemoteClient()

TUNE = False
PLAN = False
PLANNING_METHODS = {"epsilon_greedy": epsilon_greedy_plan, "genetic": genetic_plan}
planning_function = PLANNING_METHODS["epsilon_greedy"]
TIME_LIMIT = 60 * 60 * 7.75
ROBUST = True

def evaluate(config, run):
    start_time = timer()
    #obs_builder = make_obs(config["env_config"]['observation'],
    #                       config["env_config"].get('observation_config')).builder()

    obs_builder = make_obs("agent_one_hot", {"max_n_agents": 50}).builder()

    evaluation_number = 0
    total_reward = 0
    all_rewards = []
    n_agents = 0
    done = defaultdict(lambda: False)
    agent = None

    while True:
        try:
            if (timer() - start_time) > TIME_LIMIT:
                remote_client.submit()
                break

            observation, info = remote_client.env_create(obs_builder_object=obs_builder)

            if not observation:
                break

            if n_agents != remote_client.env.get_num_agents():
                n_agents = remote_client.env.get_num_agents()
                trainer = get_agent(config, run, n_agents)
                agent = RllibAgent(trainer, explore=False)

            steps = 0
            done = defaultdict(lambda: False)
            memorized_actions = None

            if PLAN:
                memorized_actions = planning_function(env=remote_client.env,
                                                      obs_dict=observation,
                                                      budget_seconds=60 * 4,
                                                      policy_agent=agent)

            evaluation_number += 1
            episode_start_info(evaluation_number, remote_client=remote_client)
            robust_env = RobustFlatlandGymEnv(rail_env=remote_client.env, observation_space=None)
            sorted_handles = robust_env.prioritized_agents(handles=observation.keys())

            while True:
                if PLAN and memorized_actions is not None:
                    for a in tqdm(memorized_actions):
                        observation, all_rewards, done, info = remote_client.env_step(a)
                        steps += 1

                        if done['__all__']:
                            break

                while not done['__all__']:
                    actions = ShortestPathAgent().compute_actions(observation, remote_client.env)
                    robust_actions = robust_env.get_robust_actions(actions, sorted_handles=sorted_handles)

                    observation, all_rewards, done, info = remote_client.env_step(robust_actions)
                    steps += 1
                    print('.', end='', flush=True)

                if done['__all__']:
                    total_reward = episode_end_info(all_rewards,
                                                    total_reward,
                                                    evaluation_number,
                                                    steps, remote_client=remote_client)
                    break

        except TimeoutException as te:
            while not done['__all__']:
                observation, all_rewards, done, info = remote_client.env_step({})
                print('!', end='', flush=True)
            print("TimeoutExeption occured!", te)

    print("Evaluation of all environments complete...")
    print(remote_client.submit())


if __name__ == "__main__":
    config, run = init_run()
    evaluate(config, run)
