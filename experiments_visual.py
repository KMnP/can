#!/usr/bin/env python3
import argparse
from src.experiments.imagenet import main as imagenet_main


def setup():
    parser = argparse.ArgumentParser(
        description='Experiments for visual datasets')

    parser.add_argument('--dataset', type=str)
    parser.add_argument(
        '--data-root', type=str)
    args = parser.parse_args()
    return args


def main():
    args = setup()

    if args.dataset == "imagenet":
        print("=" * 80)
        print("ImageNet experiments")
        print("=" * 80)
        imagenet_main(args.data_root)

    else:
        print(f"{args.dataset} is not supported.")


if __name__ == '__main__':
    main()
