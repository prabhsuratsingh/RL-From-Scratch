from agents.agent import Agent
from algorithms.qlearning import plot_learning_history, run_qlearning
from envs.cliff_walking import CliffWalkingEnv
from envs.gridworld import GridWorldEnv


env = CliffWalkingEnv(render_mode="human")
agent = Agent(env)

history = run_qlearning(agent, env)

env.close()
plot_learning_history(history, "cliffwalking", "q-learning")

# env = GridWorldEnv(num_rows=5, num_cols=6, render_mode="human")
# agent = Agent(env)

# history = run_qlearning(agent, env,)

# env.close()
# plot_learning_history(history, "gridworld", "q-learning")