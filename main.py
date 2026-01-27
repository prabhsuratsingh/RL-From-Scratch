from agents.agent import Agent
from algorithms.qlearning import run_qlearning
from algorithms.sarsa import run_sarsa
from envs.cliff_walking import CliffWalkingEnv
from envs.gridworld import GridWorldEnv
from utils.plots import plot_learning_history


# env = CliffWalkingEnv(render_mode="human")
# agent = Agent(env, algorithm="q_learning")

# history = run_qlearning(agent, env)

# env.close()
# plot_learning_history(history, "cliffwalking", "q-learning")

env = GridWorldEnv(num_rows=5, num_cols=6, render_mode="human")
agent = Agent(env, algorithm="sarsa")

history = run_sarsa(agent, env,)

env.close()
plot_learning_history(history, "gridworld", "sarsa")