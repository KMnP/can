#!/usr/bin/env python3
import argparse
from src.experiments.simulate import main as simulate


def setup():
    parser = argparse.ArgumentParser(
        description='Experiments for visual datasets')

    parser.add_argument(
        '--data-root', type=str)
    parser.add_argument('--step', type=int, default=1)
    args = parser.parse_args()
    return args


def main():
    args = setup()
    simulate(args.data_root, args.step)


if __name__ == '__main__':
    main()
