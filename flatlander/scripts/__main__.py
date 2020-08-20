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
        repo_dir = Path(os.path.dirname(__file__)).parent.parent
        cmd = 'docker run -v ' + str(repo_dir) + ':/src -it fl:latest bash -c "pip install -e /src && wandb login 319a2411b4ecd4527410bb49e84d0b8398bed6bc && ' \
              'python3 /src/flatlander/scripts/baselines.py ' \
              + " ".join(sys.argv[2:]) + ' "'
        os.system(cmd)

    @staticmethod
    def experiment():
        repo_dir = Path(os.path.dirname(__file__)).parent.parent
        cmd = 'docker run -v ' + str(repo_dir) + ':/src -it fl:latest bash -c "pip install -e /src && wandb login 319a2411b4ecd4527410bb49e84d0b8398bed6bc && ' \
              'python3 /src/flatlander/scripts/experiment.py ' \
              + " ".join(sys.argv[2:]) + ' "'
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


def main():
    FlatlanderCLI()


if __name__ == "__main__":
    main()
