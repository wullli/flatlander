import sys
sys.path.append(".")

import time

import numpy as np

from flatland.evaluators.client import FlatlandRemoteClient
from flatlander.env.observations.default_observation_builder import CustomObservationBuilder

remote_client = FlatlandRemoteClient()


def random_agent(_, n_agents: int):
    _action = {}
    for _idx in range(n_agents):
        _action[_idx] = np.random.randint(0, 5)
    return _action


obs_builder = CustomObservationBuilder()

evaluation_number = 0
while True:

    evaluation_number += 1
    time_start = time.time()
    observation, info = remote_client.env_create(
        obs_builder_object=obs_builder
    )
    env_creation_time = time.time() - time_start
    if not observation:
        break

    print("Evaluation Number : {}".format(evaluation_number))

    local_env = remote_client.env
    number_of_agents = len(local_env.agents)

    time_taken_by_controller = []
    time_taken_per_step = []
    steps = 0
    while True:
        time_start = time.time()
        action_list = []

        action = random_agent(observation, n_agents=number_of_agents)

        time_taken = time.time() - time_start
        time_taken_by_controller.append(time_taken)

        time_start = time.time()
        observation, all_rewards, done, info = remote_client.env_step(action)
        steps += 1
        time_taken = time.time() - time_start
        time_taken_per_step.append(time_taken)

        if done['__all__']:
            print("Reward : ", sum(list(all_rewards.values())))
            break

    np_time_taken_by_controller = np.array(time_taken_by_controller)
    np_time_taken_per_step = np.array(time_taken_per_step)
    print("=" * 100)
    print("=" * 100)
    print("Evaluation Number : ", evaluation_number)
    print("Current Env Path : ", remote_client.current_env_path)
    print("Env Creation Time : ", env_creation_time)
    print("Number of Steps : ", steps)
    print("Mean/Std of Time taken by Controller : ", np_time_taken_by_controller.mean(),
          np_time_taken_by_controller.std())
    print("Mean/Std of Time per Step : ", np_time_taken_per_step.mean(), np_time_taken_per_step.std())
    print("=" * 100)

print("Evaluation of all environments complete...")
print(remote_client.submit())
