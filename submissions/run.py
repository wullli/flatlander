from collections import defaultdict
from flatlander.agents.shortest_path_rllib_agent import ShortestPathRllibAgent
from flatlander.envs.utils.robust_gym_env import RobustFlatlandGymEnv
from flatland.evaluators.client import FlatlandRemoteClient, TimeoutException
from flatlander.envs.observations import make_obs
from flatlander.submission.helper import episode_start_info, episode_end_info, init_run, get_agent
from time import time, sleep
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
remote_client = FlatlandRemoteClient()

TIME_LIMIT = 60 * 60 * 8


def skip(done):
    print("Skipping episode")
    while not done['__all__']:
        observation, all_rewards, done, info = remote_client.env_step({})
        print('!', end='', flush=True)


def evaluate(config, run):
    start_time = time()
    obs_builder = make_obs(config["env_config"]['observation'],
                           config["env_config"].get('observation_config')).builder()
    evaluation_number = 0
    total_reward = 0
    all_rewards = []
    trainer = get_agent(config, run)
    agent = ShortestPathRllibAgent(trainer, explore=False)

    while True:
        try:
            observation, info = remote_client.env_create(obs_builder_object=obs_builder)

            if not observation:
                break

            steps = 0
            done = defaultdict(lambda: False)

            evaluation_number += 1
            episode_start_info(evaluation_number, remote_client=remote_client)
            robust_env = RobustFlatlandGymEnv(rail_env=remote_client.env, observation_space=None, allow_noop=True)
            sorted_handles = robust_env.prioritized_agents(handles=observation.keys())

            while True:
                try:
                    while not done['__all__']:
                        actions = agent.compute_actions(observation, remote_client.env)
                        robust_actions = robust_env.get_robust_actions(actions, sorted_handles=sorted_handles)

                        observation, all_rewards, done, info = remote_client.env_step(robust_actions)
                        steps += 1
                        print('.', end='', flush=True)

                        if (time() - start_time) > TIME_LIMIT:
                            skip(done)
                            break

                    if done['__all__']:
                        total_reward = episode_end_info(all_rewards,
                                                        total_reward,
                                                        evaluation_number,
                                                        steps, remote_client=remote_client)
                        break

                except TimeoutException as err:
                    print("Timeout! Will skip this episode and go to the next.", err)
                    break
        except TimeoutException as err:
            print("Timeout during planning time. Will skip to next evaluation!", err)

    print("Evaluation of all environments complete...")
    print(remote_client.submit())


if __name__ == "__main__":
    config, run = init_run()
    evaluate(config, run)
