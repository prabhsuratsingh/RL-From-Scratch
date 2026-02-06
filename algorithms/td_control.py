from agents.double_td_agent import DoubleTDAgent
from envs.cliff_walking import CliffWalkingEnv
from envs.gridworld import GridWorldEnv
from agents.agent import Agent
from utils.plots import plot_learning_history


def run_td_control(agent, env, num_episodes=50, max_steps=500):
    history = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        env.render()

        final_reward, n_moves = 0.0, 0

        for step in range(max_steps):
            action = agent.choose_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            next_action = agent.choose_action(next_state)

            transition = (
                state,
                action,
                reward,
                next_state,
                next_action,
                terminated,
                truncated
            )

            agent.learn(transition)

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
    # env = GridWorldEnv(num_rows=5, num_cols=6, render_mode="human")
    env = CliffWalkingEnv(render_mode="human")
    agent = DoubleTDAgent(env, algorithm="double_sarsa")

    history = run_td_control(agent, env,)

    env.close()
    plot_learning_history(history, "cliff_walking", "double_sarsa", save_dir="experiments/td")
