from agents.deep_rl.dqn import DQNAgent
from envs.cart_pole import CartPoleEnv
from utils.plots import plot_learning_history


def run_dqn(agent, env, num_episodes=200, max_steps=500, batch_size=64, render=False):
    rewards, losses = [], []
    history = []

    for ep in range(num_episodes):
        state, _ = env.reset()
        ep_reward = 0

        for step in range(max_steps):
            if render:
                env.render()

            action = agent.choose_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.remember((state, action, reward, next_state, done))

            loss = agent.replay(batch_size)
            losses.append(loss)

            state = next_state
            ep_reward += reward

            if done:
                break

        history.append((step + 1, ep_reward))
        rewards.append(ep_reward)
        print(f"Episode {ep}: Reward {ep_reward}")

    return rewards, losses, history

if __name__ == "__main__":
    env = CartPoleEnv(render_mode="human")
    agent = DQNAgent(env)

    rewards, losses, history = run_dqn(agent, env, render=True)

    env.close()
    plot_learning_history(history, "cart_pole", "dqn", save_dir="experiments/latest/dqn")
