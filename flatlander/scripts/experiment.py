#!/usr/bin/envs python

from flatlander.runner.experiment_runner import ExperimentRunner
from flatlander.utils.argparser import create_parser
from argparse import ArgumentParser
import logging
logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    runner = ExperimentRunner()
    parser: ArgumentParser = create_parser()
    args = parser.parse_args()
    exps = runner.get_experiments(args, parser)
    runner.run(experiments=exps, args=args)
