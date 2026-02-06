import argparse
from runners.experiment import AGENT_REGISTRY, ENV_REGISTRY, run_experiment


def main():
    parser = argparse.ArgumentParser("rl")

    parser.add_argument("--env", required=True, choices=ENV_REGISTRY.keys())
    parser.add_argument("--agent", required=True, choices=AGENT_REGISTRY.keys())
    parser.add_argument("--alg", required=True)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--render", action="store_true")

    parser.add_argument(
        "--subdir",
        type=str,
        default="default",
        help="Subdirectory under experiments/latest/ to save results",
    )

    args = parser.parse_args()

    run_experiment(
        env_name=args.env,
        agent_type=args.agent,
        algorithm=args.alg,
        render=args.render,
        num_episodes=args.episodes,
        subdir=args.subdir,
    )


if __name__ == "__main__":
    main()
