import os

from flatlander.agents.heuristic_agent import HeuristicPriorityAgent
from flatlander.envs.observations.conflict_piority_shortest_path_obs import ConflictPriorityShortestPathObservation

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

from flatland.evaluators.client import FlatlandRemoteClient
from flatlander.submission.helper import episode_start_info, episode_end_info

tf.compat.v1.disable_eager_execution()
remote_client = FlatlandRemoteClient()

TUNE = False
TIME_LIMIT = 60 * 60 * 7.75


def evaluate():
    obs_builder = ConflictPriorityShortestPathObservation(config={'predictor': 'custom', 'asserts': True,
                                                                  'shortest_path_max_depth': 100}).builder()
    evaluation_number = 0
    total_reward = 0
    agent = HeuristicPriorityAgent()

    while True:

        observation, info = remote_client.env_create(obs_builder_object=obs_builder)

        if not observation:
            break

        evaluation_number += 1
        episode_start_info(evaluation_number, remote_client=remote_client)

        steps = 0

        while True:

            actions = agent.compute_actions(observation, remote_client.env)
            observation, all_rewards, done, info = remote_client.env_step(actions)
            steps += 1

            print('.', end='', flush=True)

            if done['__all__']:
                total_reward = episode_end_info(all_rewards,
                                                total_reward,
                                                evaluation_number,
                                                steps, remote_client=remote_client)
                break

    print("Evaluation of all environments complete...")
    print(remote_client.submit())


if __name__ == "__main__":
    evaluate()
