import os
from argparse import ArgumentParser
from pathlib import Path

import gym
import ray
import ray.tune.result as ray_results
import yaml
from gym.spaces import Tuple
from ray.cluster_utils import Cluster
from ray.rllib.utils import try_import_tf, try_import_torch
from ray.tune import run_experiments, register_env
from ray.tune.logger import TBXLogger
from ray.tune.resources import resources_to_json
from ray.tune.tune import _make_scheduler
from ray.tune.utils import merge_dicts

from flatlander.envs import get_eval_config
from flatlander.envs.flatland_sparse import FlatlandSparse
from flatlander.envs.observations import make_obs
from flatlander.envs.utils.global_gym_env import GlobalFlatlandGymEnv
from flatlander.envs.utils.gym_env_fill_missing import FillingFlatlandGymEnv
from flatlander.logging.custom_metrics import on_episode_end
from flatlander.logging.wandb_logger import WandbLogger
from flatlander.utils.loader import load_envs, load_models

ray_results.DEFAULT_RESULTS_DIR = os.path.join(os.getcwd(), "..", "..", "..", "flatland-challenge-data/results")


class ExperimentRunner:
    group_algorithms = ["QMIX", "QMIXApex"]

    def __init__(self):
        self.tf = try_import_tf()
        self.torch, _ = try_import_torch()
        load_envs(os.path.dirname(__file__))
        load_models(os.path.dirname(__file__))

    @staticmethod
    def get_experiments(run_args, arg_parser: ArgumentParser = None):
        if run_args.config_file:
            with open(run_args.config_file) as f:
                experiments = yaml.safe_load(f)
        else:
            experiments = {
                run_args.experiment_name: {  # i.e. log to ~/ray_results/default
                    "run": run_args.run,
                    "checkpoint_freq": run_args.checkpoint_freq,
                    "keep_checkpoints_num": run_args.keep_checkpoints_num,
                    "checkpoint_score_attr": run_args.checkpoint_score_attr,
                    "local_dir": run_args.local_dir,
                    "resources_per_trial": (
                            run_args.resources_per_trial and
                            resources_to_json(run_args.resources_per_trial)),
                    "stop": run_args.stop,
                    "config": dict(run_args.config, env=run_args.env),
                    "restore": run_args.restore,
                    "num_samples": run_args.num_samples,
                    "upload_dir": run_args.upload_dir,
                }
            }

            if arg_parser is not None:
                for exp in experiments.values():
                    if not exp.get("run"):
                        arg_parser.error("the following arguments are required: --run")
                    if not exp.get("envs") and not exp.get("config", {}).get("envs"):
                        arg_parser.error("the following arguments are required: --envs")

        return experiments

    @staticmethod
    def setup_grouping(config: dict):
        grouping = {
            "group_1": list(range(config["env_config"]["max_n_agents"])),
        }

        obs_space = Tuple([make_obs(config["env_config"]["observation"],
                                    config["env_config"]["observation_config"]).observation_space()
                           for _ in range(config["env_config"]["max_n_agents"])])

        act_space = Tuple([GlobalFlatlandGymEnv.action_space for _ in range(config["env_config"]["max_n_agents"])])

        register_env(
            "flatland_sparse_grouped",
            lambda config: FlatlandSparse(config).with_agent_groups(
                grouping, obs_space=obs_space, act_space=act_space))

    def setup_policy_map(self, config: dict):
        obs_space = make_obs(config["env_config"]["observation"],
                             config["env_config"]["observation_config"]).observation_space()
        config["multiagent"] = {
            "policies": {"pol_" + str(i): (None, obs_space, FillingFlatlandGymEnv.action_space, {"agent_id": i})
                         for i in range(config["env_config"]["observation_config"]["max_n_agents"])},
            "policy_mapping_fn": lambda agent_id: "pol_" + str(agent_id)}

    def setup_hierarchical_policies(self, config: dict):
        obs_space: gym.spaces.Tuple = make_obs(config["env_config"]["observation"],
                             config["env_config"]["observation_config"]).observation_space()
        config["multiagent"] = {
            "policies": {"meta": (None, obs_space.spaces[0], gym.spaces.Box(high=1, low=0, shape=(1,)), {}),
                         "agent": (None, obs_space.spaces[1], FillingFlatlandGymEnv.action_space, {})
                         },
            "policy_mapping_fn": lambda agent_id: "meta" if 'meta' in str(agent_id) else "agent"
        }

    def apply_args(self, run_args, experiments: dict):
        verbose = 1
        webui_host = '127.0.0.1'
        for exp in experiments.values():
            if run_args.eager:
                exp["config"]["eager"] = True
            if run_args.torch:
                exp["config"]["use_pytorch"] = True
            if run_args.v:
                exp["config"]["log_level"] = "INFO"
                verbose = 2
            if run_args.vv:
                exp["config"]["log_level"] = "DEBUG"
                verbose = 3
            if run_args.trace:
                if not exp["config"].get("eager"):
                    raise ValueError("Must enable --eager to enable tracing.")
                exp["config"]["eager_tracing"] = True
            if run_args.bind_all:
                webui_host = "0.0.0.0"
            if run_args.log_flatland_stats:
                exp['config']['callbacks'] = {
                    'on_episode_end': on_episode_end,
                }
            return experiments, verbose

    @staticmethod
    def evaluate(exp):
        eval_configs = get_eval_config(exp['config'].get('env_config',
                                                         {}).get('eval_generator', "default"))
        eval_seed = eval_configs.get('evaluation_config', {}).get('env_config', {}).get('seed')

        # add evaluation config to the current config
        exp['config'] = merge_dicts(exp['config'], eval_configs)
        if exp['config'].get('evaluation_config'):
            exp['config']['evaluation_config']['env_config'] = exp['config'].get('env_config')
            eval_env_config = exp['config']['evaluation_config'].get('env_config')
            if eval_seed and eval_env_config:
                # We override the envs seed from the evaluation config
                eval_env_config['seed'] = eval_seed

            # Remove any wandb related configs
            if eval_env_config:
                if eval_env_config.get('wandb'):
                    del eval_env_config['wandb']

        # Remove any wandb related configs
        if exp['config']['evaluation_config'].get('wandb'):
            del exp['config']['evaluation_config']['wandb']

    def run(self, experiments: dict, args=None):
        verbose = 1
        webui_host = "localhost"
        for exp in experiments.values():
            if exp.get("config", {}).get("input"):
                if not isinstance(exp.get("config", {}).get("input"), dict):
                    if not os.path.exists(exp["config"]["input"]):
                        rllib_dir = Path(__file__).parent
                        input_file = rllib_dir.absolute().joinpath(exp["config"]["input"])
                        exp["config"]["input"] = str(input_file)

            if exp["run"] in self.group_algorithms:
                self.setup_grouping(exp.get("config"))

            if exp["run"] == "contrib/MADDPG" or exp["config"].get("individual_policies", False):
                self.setup_policy_map(exp.get("config"))
                if exp["config"].get("individual_policies", False):
                    del exp["config"]["individual_policies"]
            if exp["run"] == "contrib/MADDPG":
                exp.get("config")["env_config"]["learning_starts"] = 100
                exp.get("config")["env_config"]["actions_are_logits"] = True

            if exp["env"] == "flatland_sparse_hierarchical":
                self.setup_hierarchical_policies(exp.get("config"))

            if args is not None:
                experiments, verbose = self.apply_args(run_args=args, experiments=experiments)

                if args.eval:
                    self.evaluate(exp)

                if args.config_file:
                    # TODO should be in exp['config'] directly
                    exp['config']['env_config']['yaml_config'] = args.config_file
                exp['loggers'] = [WandbLogger, TBXLogger]

        if args.ray_num_nodes:
            cluster = Cluster()
            for _ in range(args.ray_num_nodes):
                cluster.add_node(
                    num_cpus=args.ray_num_cpus or 1,
                    num_gpus=args.ray_num_gpus or 0,
                    object_store_memory=args.ray_object_store_memory,
                    memory=args.ray_memory,
                    redis_max_memory=args.ray_redis_max_memory)
            ray.init(address=cluster.address)
        else:
            import multiprocessing
            n_cpu = multiprocessing.cpu_count()
            import tensorflow as tf
            n_gpu = len(tf.config.experimental.list_physical_devices('GPU'))
            print("NUM_CPUS AVAILABLE: ", n_cpu)
            print("NUM_GPUS AVAILABLE: ", n_gpu)
            print("NUM_CPUS ARGS: ", args.ray_num_cpus)
            print("NUM_GPUS ARGS: ", args.ray_num_gpus)
            ray.init(
                local_mode=True if args.local else False,
                address=args.ray_address,
                object_store_memory=args.ray_object_store_memory,
                num_cpus=args.ray_num_cpus if args.ray_num_cpus is not None else n_cpu,
                num_gpus=args.ray_num_gpus if args.ray_num_gpus is not None else n_gpu)

        run_experiments(
            experiments,
            scheduler=_make_scheduler(args),
            queue_trials=args.queue_trials,
            resume=args.resume,
            verbose=verbose,
            concurrent=True)

        ray.shutdown()
