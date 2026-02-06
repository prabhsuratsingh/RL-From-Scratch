from collections import namedtuple

from envs.gridworld import GridWorldEnv
from agents.td_agent import Agent
from utils.plots import plot_learning_history

Transition = namedtuple(
    'Transition',
    ('state', 'action', 'reward', 'next_state', 'terminated', 'truncated')
)

def run_expected_sarsa(agent, env, num_episodes=50, max_steps=500):
    history = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        env.render()

        final_reward, n_moves = 0.0, 0

        for step in range(max_steps):
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

            if terminated:
                break

        else:
            truncated = True

        history.append((n_moves, final_reward))
        print(
            f"Episode {episode}: "
            f"Reward {final_reward:.2f} "
            f"#Moves {n_moves}"
        )

    return history


if __name__ == "__main__":
    env = GridWorldEnv(num_rows=5, num_cols=6, render_mode="human")
    agent = Agent(env, algorithm="expected_sarsa")

    history = run_expected_sarsa(agent, env,)

    env.close()
    plot_learning_history(history, "gridworld", "expected_sarsa")
