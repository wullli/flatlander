#!/usr/bin/env python

from argparse import ArgumentParser

from flatlander.runner.experiment_runner import ExperimentRunner
from flatlander.utils.argparser import create_parser


if __name__ == "__main__":
    runner = ExperimentRunner()
    parser: ArgumentParser = create_parser()
    args = parser.parse_args()
    exps = runner.get_experiments(args, parser)
    runner.run(experiments=exps, args=args)
