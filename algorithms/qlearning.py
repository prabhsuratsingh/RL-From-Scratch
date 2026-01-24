from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt

from envs.gridworld import GridWorldEnv
from agents.agent import Agent

Transition = namedtuple(
    'Transition',
    ('state', 'action', 'reward', 'next_state', 'terminated', 'truncated')
)

def run_qlearning(agent, env, num_episodes=50):
    history = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        env.render()

        final_reward, n_moves = 0.0, 0

        while True:
            action = agent.choose_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)

            agent.learn(
                Transition(
                    state,
                    action,
                    reward,
                    next_state,
                    terminated,
                    truncated
                )
            )

            env.render()

            state = next_state
            n_moves += 1
            final_reward = reward

            if terminated or truncated:
                break

        history.append((n_moves, final_reward))
        print(
            f"Episode {episode}: "
            f"Reward {final_reward:.2f} "
            f"#Moves {n_moves}"
        )

    return history


def plot_learning_history(history):
    fig = plt.figure(1, figsize=(14, 10))
    ax = fig.add_subplot(2, 1, 1)
    episodes = np.arange(len(history))
    moves = np.array([h[0] for h in history])
    plt.plot(episodes, moves, lw=4,
             marker="o", markersize=10)
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.xlabel('Episodes', size=20)
    plt.ylabel('# moves', size=20)

    ax = fig.add_subplot(2, 1, 2)
    rewards = np.array([h[1] for h in history])
    plt.step(episodes, rewards, lw=4)
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.xlabel('Episodes', size=20)
    plt.ylabel('Final rewards', size=20)
    plt.savefig('q-learning-history.png', dpi=300)
    plt.show()



if __name__ == "__main__":
    env = GridWorldEnv(num_rows=5, num_cols=6, render_mode="human")
    agent = Agent(env)

    history = run_qlearning(agent, env)

    env.close()
    plot_learning_history(history)
