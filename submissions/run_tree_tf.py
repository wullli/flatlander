import os
import time

import numpy as np
import ray
import yaml
from ray.rllib.agents.ppo import PPOTrainer

from flatlander.models.common.transformer_ppo_policy import TTFPPOPolicyInfer

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.evaluators.client import FlatlandRemoteClient
from flatlander.envs.observations.fixed_tree_obs import FixedTreeObsWrapper
from flatlander.utils.deadlock_check import check_if_all_blocked
from flatlander.utils.loader import load_envs, load_models

remote_client = FlatlandRemoteClient()


def init():
    with open(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                           "model_checkpoints/tree_tf_1/checkpoint_5860_scaling/config.yaml"))) as f:
        config = yaml.safe_load(f)
    load_envs(os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../flatlander/runner")))
    load_models(os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../flatlander/runner")))

    obs_builder = FixedTreeObsWrapper(
        TreeObsForRailEnv(
            max_depth=config['env_config']['observation_config']['max_depth'],
            predictor=ShortestPathPredictorForRailEnv(
                config['env_config']['observation_config']['shortest_path_max_depth'])
        ),
        small_tree=False
    )

    ray.init(local_mode=False, num_gpus=0)
    agent = PPOTrainer.with_updates(name="TTFPPOPolicyInfer",
                                    get_policy_class=lambda c: TTFPPOPolicyInfer)(config=config)
    agent.restore(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                               "model_checkpoints/tree_tf_1/checkpoint_5860_scaling/checkpoint-5860")))
    policy = agent.get_policy()
    return policy, obs_builder


def evaluate(policy, obs_builder):
    evaluation_number = 0
    total_reward = 0
    while True:

        evaluation_number += 1
        time_start = time.time()
        observation, info = remote_client.env_create(
            obs_builder_object=obs_builder

        )
        env_creation_time = time.time() - time_start
        if not observation:
            break

        print("Starting evaluation #{}".format(evaluation_number))
        print("Number of agents:", len(remote_client.env.agents))
        print("Environment size:", remote_client.env.width, "x", remote_client.env.height)

        time_taken_by_controller = []
        time_taken_per_step = []
        steps = 0

        while True:
            if not check_if_all_blocked(env=remote_client.env):
                time_start = time.time()

                obs_batch = np.array(list(observation.values()))
                action_batch = policy.compute_actions(obs_batch, explore=False)
                actions = dict(zip(observation.keys(), action_batch[0]))

                time_taken = time.time() - time_start
                time_taken_by_controller.append(time_taken)

                time_start = time.time()
                observation, all_rewards, done, info = remote_client.env_step(actions)

                steps += 1

                while len(observation) == 0:
                    if done['__all__']:
                        break
                    observation, all_rewards, done, info = remote_client.env_step({})
                    print('.', end='', flush=True)
                    steps += 1

                time_taken = time.time() - time_start
                time_taken_per_step.append(time_taken)
                print('.', end='', flush=True)

            else:
                if done['__all__']:
                    break
                time_start = time.time()
                _, all_rewards, done, info = remote_client.env_step({})
                step_time = time.time() - time_start
                time_taken_per_step.append(step_time)
                print('!', end='', flush=True)

            if done['__all__']:
                reward_values = np.array(list(all_rewards.values()))
                gained_reward = np.mean(1 + reward_values)
                total_reward += gained_reward
                print("\n\nGained reward: ", gained_reward, "/ Max possible:",1)
                print("Total reward: ", total_reward, "\n")
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


if __name__ == "__main__":
    p, o = init()
    evaluate(p, o)
