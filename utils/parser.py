import os
import argparse
from pathlib import Path

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='yaml config file')
    parser.add_argument('--exp_name', type=str, default='default', help='experiment name')

    args = parser.parse_args()

    args.experiment_path = os.path.join('./experiments', Path(args.config).stem, Path(args.config).parent.stem, args.exp_name)
    create_experiment_dir(args)

    return args

def create_experiment_dir(args):
    if not os.path.exists(args.experiment_path):
        os.makedirs(args.experiment_path)
        print('Create experiment path succesfully at %s' % args.experiment_path)
    