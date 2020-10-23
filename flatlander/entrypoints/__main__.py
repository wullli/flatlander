import argparse
import os
import sys
from pathlib import Path


class FlatlanderCLI(object):

    def __init__(self):
        parser = argparse.ArgumentParser(
            description='Flatlander CLI interface',
            usage='''flatlander <command> [<args>]
            
                     Valid commands:
                     
                     unused_baselines:     execute baseline algorithms
                     experiment:    run a yaml experiment
                     rebuild:       build the docker image from scratch
                     build:         build the docker image using cache''')

        parser.add_argument('command', help='Subcommand to run')

        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            parser.print_help()
            exit(1)

        getattr(self, args.command)()

    @staticmethod
    def baselines():
        # pass -d if daemon mode desired
        name = "fl_baselines"
        cmd_prefix = 'docker run --log-opt max-size=10m --log-opt max-file=5 --shm-size ' \
                     '200000000000 ' \
                     '--name ' + str(name)

        if "-d" in sys.argv[2:]:
            cmd_prefix += ' -d'

        if "-g" in sys.argv[2:]:
            cmd_prefix += ' --gpus all'

        repo_dir = Path(os.path.dirname(__file__)).parent.parent
        data_dir = repo_dir.parent / "flatland-challenge-data/expert_data"
        out_dir = repo_dir.parent / "flatland-challenge-data/out"
        cmd = cmd_prefix \
              + ' -v ' + str(data_dir) + ':/tmp/flatland-out ' \
              + ' -v ' + str(out_dir) + ':/home/$USER/ray_results ' \
              + ' -v ' + str(repo_dir) + ':/src -it fl:latest bash -c "pip install -e /src && wandb login ' \
                                         '319a2411b4ecd4527410bb49e84d0b8398bed6bc && ' \
                                         'python3 /src/flatlander/entrypoints/baselines.py ' \
              + " ".join(filter(lambda arg: arg != "-d" and arg != "-g", sys.argv[2:])) + ' "'
        os.system(cmd)

    @staticmethod
    def experiment():
        parser = argparse.ArgumentParser()
        parser.add_argument('-f', help='Experiment config file')
        args, _ = parser.parse_known_args()
        repo_dir = Path(os.path.dirname(__file__)).parent.parent
        out_dir = repo_dir.parent / "flatland-challenge-data/out"
        base = os.path.basename(args.f)
        exp_name = os.path.splitext(base)[0]
        name = "fl_experiment_" + str(exp_name)

        cmd_prefix = 'docker run --log-opt max-size=1m --log-opt max-file=5 --shm-size ' \
                     '200000000000 ' \
                     '--name ' + str(name)

        if "-d" in sys.argv[2:]:
            cmd_prefix += ' -d'

        if "-g" in sys.argv[2:]:
            cmd_prefix += ' --gpus all'

        if not "-noresults" in sys.argv[2:]:
            cmd_prefix += ' -v ' + str(out_dir) + ':/home/$USER/ray_results '
        cmd = cmd_prefix \
              + ' -v ' + str(repo_dir) + ':/src -it fl:latest bash -c \'pip install -e /src && wandb login ' \
                                         '319a2411b4ecd4527410bb49e84d0b8398bed6bc && ' \
                                         'python3 /src/flatlander/entrypoints/experiment.py ' \
              + " ".join(filter(lambda arg: arg != "-d"
                                            and arg != "-noresults"
                                            and arg != "-g",
                                sys.argv[2:])) + ' \''
        os.system(cmd)

    @staticmethod
    def experiments():
        parser = argparse.ArgumentParser()
        parser.add_argument('--experiments-dir', help='Experiment config file')
        args, _ = parser.parse_known_args()
        repo_dir = Path(os.path.dirname(__file__)).parent.parent
        out_dir = repo_dir.parent / "flatland-challenge-data/out"
        base = os.path.basename(args.experiments_dir)
        exp_name = os.path.splitext(base)[0]
        name = "fl_experiment_" + str(exp_name)

        cmd_prefix = 'docker run --log-opt max-size=1m --log-opt max-file=5 --shm-size ' \
                     '200000000000 ' \
                     '--name ' + str(name)

        if "-d" in sys.argv[2:]:
            cmd_prefix += ' -d'

        if "-g" in sys.argv[2:]:
            cmd_prefix += ' --gpus all'

        if not "-noresults" in sys.argv[2:]:
            cmd_prefix += ' -v ' + str(out_dir) + ':/home/$USER/ray_results '

        cmd = cmd_prefix \
              + ' -v ' + str(repo_dir) + ':/src -it fl:latest bash -c \'pip install -e /src && wandb login ' \
                                         '319a2411b4ecd4527410bb49e84d0b8398bed6bc && ' \
                                         'python3 /src/flatlander/entrypoints/experiments.py ' \
              + " ".join(filter(lambda arg: arg != "-d"
                                            and arg != "-noresults"
                                            and arg != "-g",
                                sys.argv[2:])) + ' \''
        os.system(cmd)

    @staticmethod
    def rebuild():
        os.system('docker image rm fl:latest')
        cmd = os.path.join(os.path.dirname(__file__), "build.sh")
        os.system(cmd)

    @staticmethod
    def build():
        cmd = os.path.join(os.path.dirname(__file__), "build.sh")
        os.system(cmd)

    @staticmethod
    def evaluate():
        parser = argparse.ArgumentParser()
        parser.add_argument('-f', help='Experiment config file')
        args, _ = parser.parse_known_args()
        repo_dir = Path(os.path.dirname(__file__)).parent.parent
        out_dir = repo_dir.parent / "flatland-challenge-data/out"
        base = os.path.basename(args.f)
        exp_name = os.path.splitext(base)[0]
        name = "fl_evaluate_" + str(exp_name)

        cmd_prefix = 'docker run --log-opt max-size=1m --log-opt max-file=5 --shm-size ' \
                     '200000000000 ' \
                     '--name ' + str(name)

        if "-d" in sys.argv[2:]:
            cmd_prefix += ' -d'

        if "-g" in sys.argv[2:]:
            cmd_prefix += ' --gpus all'

        cmd = cmd_prefix \
              + ' -v ' + str(out_dir) + ':/home/$USER/ray_results ' \
              + ' -v ' + str(repo_dir) + ':/src -it fl:latest bash -c \'pip install -e /src &&' \
                                         'python3 /src/flatlander/entrypoints/rollout.py ' \
              + " ".join(filter(lambda arg: arg != "-d" and arg != "-g", sys.argv[2:])) + ' \''
        os.system(cmd)

    @staticmethod
    def load_image():
        parser = argparse.ArgumentParser(description='Flatlander CLI interface')
        parser.add_argument('-i', '--input', help='input tar file', required=True)
        args = parser.parse_args(sys.argv[2:])
        os.system('docker load -i ' + args.input)


def main():
    FlatlanderCLI()


if __name__ == "__main__":
    main()
