from agents.double_td_agent import DoubleTDAgent
from agents.dyna.dyna_q_agent import DynaQAgent
from agents.n_step_td_agent import NStepTDAgent
from agents.td_agent import TDAgent
from envs.dyna_maze import DynaMazeEnv
from envs.gridworld import GridWorldEnv
from envs.cliff_walking import CliffWalkingEnv
from training.run_td_control import run_td_control
from utils.plots import plot_learning_history


ENV_REGISTRY = {
    "grid_world": GridWorldEnv,
    "cliff_walking": CliffWalkingEnv,
    "dyna_maze": DynaMazeEnv,
}

AGENT_REGISTRY = {
    "td": TDAgent,
    "double_td": DoubleTDAgent,
    "n_step_td": NStepTDAgent,
    "dynaq": DynaQAgent,
}

TD_ALGOS = {
    "q_learning",
    "sarsa",
    "expected_sarsa",
}

N_STEP_TD_ALGOS = {
    "q_learning",
    "sarsa",
    "expected_sarsa",
}

DOUBLE_TD_ALGOS = {
    "double_q",
    "double_sarsa",
    "double_expected_sarsa",
}

def run_experiment(
    env_name,
    agent_type,
    algorithm,
    render,
    num_episodes,
    subdir,
):
    if env_name not in ENV_REGISTRY:
        raise ValueError(f"Unknown env: {env_name}")

    if agent_type not in AGENT_REGISTRY:
        raise ValueError(f"Unknown agent type: {agent_type}")

    if agent_type == "td" and algorithm not in TD_ALGOS:
        raise ValueError(f"Invalid TD algorithm: {algorithm}")

    if agent_type == "double_td" and algorithm not in DOUBLE_TD_ALGOS:
        raise ValueError(f"Invalid Double TD algorithm: {algorithm}")
    
    if agent_type == "n_step_td" and algorithm not in N_STEP_TD_ALGOS:
        raise ValueError(f"Invalid n-Step TD algorithm: {algorithm}")


    env_class = ENV_REGISTRY[env_name]
    env = env_class(render_mode="human" if render else None)

    agent_class = AGENT_REGISTRY[agent_type]
    agent = agent_class(env, algorithm=algorithm)

    history = run_td_control(
        agent,
        env,
        num_episodes=num_episodes,
    )

    env.close()

    save_dir = f"experiments/latest/{subdir}/{env_name}"

    plot_learning_history(
        history,
        env_name=env_name,
        alg_name=algorithm,
        save_dir=save_dir,
    )
