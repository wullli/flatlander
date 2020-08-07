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
    for baseline_yaml in glob.glob(os.path.join(os.path.dirname(__file__), "..", "resources",
                                                "baselines", "**/*.yaml")):
        args.config_file = baseline_yaml
        exps = runner.get_experiments(args)
        runner.run(experiments=exps, args=args)
