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
                     
                     baselines:     execute baseline algorithms
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
        cmd_prefix = 'docker run --name ' + str(name)
        if "-d" in sys.argv[2:]:
            cmd_prefix += ' -d'

        repo_dir = Path(os.path.dirname(__file__)).parent.parent
        data_dir = repo_dir.parent / "flatland-challenge-data/expert_data"
        cmd = cmd_prefix + ' -v ' + str(data_dir) + ':/tmp/flatland-out -v ' + str(
            repo_dir) + ':/src -it fl:latest bash -c "pip install -e /src && wandb login ' \
                        '319a2411b4ecd4527410bb49e84d0b8398bed6bc && ' \
                        'python3 /src/flatlander/scripts/baselines.py ' \
              + " ".join(filter(lambda arg: arg != "-d", sys.argv[2:])) + ' "'
        os.system(cmd)

    @staticmethod
    def experiment():
        # pass -d if daemon mode desired
        repo_dir = Path(os.path.dirname(__file__)).parent.parent
        name = "fl_experiment"
        cmd_prefix = 'docker run --name ' + str(name)
        if "-d" in sys.argv[2:]:
            cmd_prefix += ' -d'

        cmd = cmd_prefix + ' -v ' + str(
            repo_dir) + ':/src -it fl:latest bash -c "pip install -e /src && wandb login ' \
                        '319a2411b4ecd4527410bb49e84d0b8398bed6bc && ' \
                        'python3 /src/flatlander/scripts/experiment.py ' \
              + " ".join(filter(lambda arg: arg != "-d", sys.argv[2:])) + ' "'
        os.system(cmd)

    @staticmethod
    def rebuild():
        os.system('docker image rm flatland-docker')
        cmd = os.path.join(os.path.dirname(__file__), "build.sh")
        os.system(cmd)

    @staticmethod
    def build():
        cmd = os.path.join(os.path.dirname(__file__), "build.sh")
        os.system(cmd)

    @staticmethod
    def load_image():
        parser = argparse.ArgumentParser(description='Flatlander CLI interface')
        parser.add_argument('-i', '--input', help='input tar file', required=True)
        args = parser.parse_args(sys.argv[2:])
        os.system('docker load -i ' + args.input)

    @staticmethod
    def logs():
        parser = argparse.ArgumentParser(description='Flatlander CLI interface')
        parser.add_argument('-t', '--type', help='if experiment or baselines run',
                            default="experiment",
                            choices=["experiment", "baselines"])
        args, unknown = parser.parse_known_args(sys.argv[2:])
        docker_args = " ".join(unknown)
        os.system('docker logs fl_' + args.type + " " + docker_args)


def main():
    FlatlanderCLI()


if __name__ == "__main__":
    main()
