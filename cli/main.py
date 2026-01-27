import argparse
from runners.experiment import run_experiment


def main():
    parser = argparse.ArgumentParser(prog="rl")

    subparsers = parser.add_subparsers(dest="command", required=True)

    exp = subparsers.add_parser("experiment")
    exp.add_argument("--env", required=True)
    exp.add_argument("--alg", required=True)
    exp.add_argument("--render", action="store_true")

    args = parser.parse_args()

    if args.command == "experiment":
        run_experiment(
            env_name=args.env,
            alg_name=args.alg,
            render=args.render,
        )
