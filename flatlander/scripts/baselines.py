#!/usr/bin/env python
import glob
import os

from flatlander.runner.experiment_runner import ExperimentRunner

if __name__ == "__main__":
    runner = ExperimentRunner()
    for baseline_yaml in glob.glob(os.path.join(os.getcwd(), "..", "baselines", "*.yaml")):
        args = object()
        args.config_file = baseline_yaml
        exps = runner.get_experiments(args)
        runner.run(experiments=exps, args=args)
