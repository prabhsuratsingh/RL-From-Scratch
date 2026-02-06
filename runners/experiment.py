from agents.td_agent import TDAgent
from envs.gridworld import GridWorldEnv
from envs.cliff_walking import CliffWalkingEnv

from algorithms.sarsa import run_sarsa
from algorithms.qlearning import run_qlearning
from algorithms.expected_sarsa import run_expected_sarsa
from utils.plots import plot_learning_history


ENV_REGISTRY = {
    "grid_world": GridWorldEnv,
    "cliff_walking": CliffWalkingEnv,
}

ALG_REGISTRY = {
    "sarsa": run_sarsa,
    "q_learning": run_qlearning,
    "expected_sarsa": run_expected_sarsa
}


def run_experiment(env_name, alg_name, render=False):
    if env_name not in ENV_REGISTRY:
        raise ValueError(f"Unknown env: {env_name}")

    if alg_name not in ALG_REGISTRY:
        raise ValueError(f"Unknown alg: {alg_name}")

    env_class = ENV_REGISTRY[env_name]
    env = env_class(render_mode="human")

    agent = TDAgent(env, algorithm=alg_name)

    history = ALG_REGISTRY[alg_name](agent, env)

    env.close()
    plot_learning_history(
        history,
        env_name,
        alg_name,
        save_dir="experiments", 
    )
