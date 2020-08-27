#!/usr/bin/env python
from flatlander.runner.experiment_runner import ExperimentRunner

import glob
import os
from argparse import ArgumentParser

from flatlander.utils.argparser import create_parser

if __name__ == "__main__":
    parser: ArgumentParser = create_parser()
    args = parser.parse_args()
    runner = ExperimentRunner()
    experiments_files = glob.glob(os.path.join(os.path.dirname(__file__), "..", "resources",
                                                "baselines", "**/*.yaml"))
    experiments = {}
    print("RUNNING BASELINE EXPERIMENTS: ", "\n".join(experiments_files))
    for baseline_yaml in experiments_files:
        args.config_file = baseline_yaml
        exp = runner.get_experiments(args)
        experiments = dict(list(experiments.items()) + list(exp.items()))
    runner.run(experiments=experiments, args=args)
