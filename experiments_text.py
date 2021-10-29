#!/usr/bin/env python3
import argparse
from src.experiments.ultrafine_entity_typing.main import run as uf_entity_main
from src.experiments.dialogue_re.main import run as dialre_main


def setup():
    parser = argparse.ArgumentParser(
        description='Experiments for nlp datasets')

    parser.add_argument('--dataset', type=str)
    parser.add_argument(
        '--data-root', type=str)
    parser.add_argument(
        '--model-type', type=str,
        default="baseline")
    args = parser.parse_args()
    return args


def main():
    args = setup()

    if args.dataset == "ultrafine_entity_typing":
        print("=" * 80)
        print(f"Ultra-Fine Entity Typing experiments with {args.model_type} model")
        print("=" * 80)
        uf_entity_main(args.data_root, args.model_type)

    elif args.dataset == "dialogue_re":
        print("=" * 80)
        print("Dialogue-based Relation Extraction experiments")
        print("=" * 80)
        dialre_main(args.data_root)

    else:
        print(f"{args.dataset} is not supported.")


if __name__ == '__main__':
    main()
