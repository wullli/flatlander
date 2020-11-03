import os

from flatlander.agents.shortest_path_agent import ShortestPathAgent
from flatlander.envs.observations.dummy_obs import DummyObs
from flatlander.envs.utils.cpr_gym_env import CprFlatlandGymEnv
from flatlander.envs.utils.priorization.priorizer import NrAgentsSameStart

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from collections import defaultdict
from flatland.evaluators.client import FlatlandRemoteClient, TimeoutException
from flatlander.submission.helper import episode_start_info, episode_end_info
from time import time
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
remote_client = FlatlandRemoteClient()

TIME_LIMIT = 60 * 60 * 8


def skip(done):
    print("Skipping episode")
    while not done['__all__']:
        observation, all_rewards, done, info = remote_client.env_step({})
        print('!', end='', flush=True)


def evaluate():
    start_time = time()
    #obs_builder = make_obs(config["env_config"]['observation'],
    #                       config["env_config"].get('observation_config')).builder()
    evaluation_number = 0
    total_reward = 0
    all_rewards = []
    # trainer = get_agent(config, run)
    # agent = ShortestPathRllibAgent(trainer, explore=False)

    while True:
        try:
            observation, info = remote_client.env_create(obs_builder_object=DummyObs())

            if not observation:
                break

            steps = 0
            done = defaultdict(lambda: False)

            evaluation_number += 1
            episode_start_info(evaluation_number, remote_client=remote_client)
            robust_env = CprFlatlandGymEnv(rail_env=remote_client.env,
                                           max_nr_active_agents=100,
                                           observation_space=None,
                                           priorizer=NrAgentsSameStart(),
                                           allow_noop=True)
            sorted_handles = robust_env.priorizer.priorize(handles=observation.keys(), rail_env=remote_client.env)

            while True:
                try:
                    while not done['__all__']:
                        actions = ShortestPathAgent().compute_actions(observation, remote_client.env)
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
    #config, run = init_run()
    evaluate()
