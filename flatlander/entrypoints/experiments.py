#!/usr/bin/env python
import glob

from flatlander.runner.experiment_runner import ExperimentRunner
from flatlander.utils.argparser import create_parser
from argparse import ArgumentParser
import os
import logging
logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    parser: ArgumentParser = create_parser()
    args = parser.parse_args()
    runner = ExperimentRunner()
    experiments_files = glob.glob(os.path.join(args.experiments_dir, "*.yaml"))
    experiments = {}
    print("RUNNING EXPERIMENTS: ", "\n".join(experiments_files))
    for baseline_yaml in experiments_files:
        args.config_file = baseline_yaml
        exp = runner.get_experiments(args)
        experiments = dict(list(experiments.items()) + list(exp.items()))
    runner.run(experiments=experiments, args=args)
