import os
from argparse import ArgumentParser
from pathlib import Path

import ray
import yaml
from ray.cluster_utils import Cluster
from ray.rllib.evaluation import MultiAgentEpisode
from ray.rllib.utils import try_import_tf, try_import_torch
from ray.tune import run_experiments
from ray.tune.logger import TBXLogger
from ray.tune.resources import resources_to_json
import ray.tune.result as ray_results
from ray.tune.tune import _make_scheduler
from ray.tune.utils import merge_dicts

from flatlander.envs import get_eval_config
from flatlander.utils.loader import load_envs, load_models
from flatlander.logging.wandb_logger import WandbLogger

ray_results.DEFAULT_RESULTS_DIR = os.path.join(os.getcwd(), "..", "..", "..", "flatland-challenge-data/results")


class ExperimentRunner:

    def __init__(self):
        self.tf = try_import_tf()
        self.torch, _ = try_import_torch()
        load_envs(os.path.dirname(__file__))
        load_models(os.path.dirname(__file__))

    @staticmethod
    def on_episode_end(info):
        episode: MultiAgentEpisode = info["episode"]

        episode_steps = 0
        episode_max_steps = 0
        episode_num_agents = 0
        episode_score = 0
        episode_done_agents = 0
        episode_num_swaps = 0

        for agent, agent_info in episode._agent_to_last_info.items():
            if episode_max_steps == 0:
                episode_max_steps = agent_info["max_episode_steps"]
                episode_num_agents = agent_info["num_agents"]
            episode_steps = max(episode_steps, agent_info["agent_step"])
            episode_score += agent_info["agent_score"]
            if "num_swaps" in agent_info:
                episode_num_swaps += agent_info["num_swaps"]
            if agent_info["agent_done"]:
                episode_done_agents += 1

        # Not a valid check when considering a single policy for multiple agents
        # assert len(episode._agent_to_last_info) == episode_num_agents

        norm_factor = 1.0 / (episode_max_steps * episode_num_agents)
        percentage_complete = float(episode_done_agents) / episode_num_agents

        episode.custom_metrics["episode_steps"] = episode_steps
        episode.custom_metrics["episode_max_steps"] = episode_max_steps
        episode.custom_metrics["episode_num_agents"] = episode_num_agents
        episode.custom_metrics["episode_return"] = episode.total_reward
        episode.custom_metrics["episode_score"] = episode_score
        episode.custom_metrics["episode_score_normalized"] = episode_score * norm_factor
        episode.custom_metrics["episode_num_swaps"] = episode_num_swaps / 2
        episode.custom_metrics["percentage_complete"] = percentage_complete

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
                    'on_episode_end': self.on_episode_end,
                }
            return experiments, verbose, webui_host

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

            if args is not None:
                experiments, verbose, webui_host = self.apply_args(run_args=args, experiments=experiments)

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
                local_mode=False,
                address=args.ray_address,
                object_store_memory=args.ray_object_store_memory,
                memory=args.ray_memory,
                redis_max_memory=args.ray_redis_max_memory,
                num_cpus=args.ray_num_cpus if args.ray_num_cpus is not None else n_cpu,
                num_gpus=args.ray_num_gpus if args.ray_num_gpus is not None else n_gpu,
                webui_host=webui_host)

        run_experiments(
            experiments,
            scheduler=_make_scheduler(args),
            queue_trials=args.queue_trials,
            resume=args.resume,
            verbose=verbose,
            concurrent=True)

        ray.shutdown()
