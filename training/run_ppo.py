import torch

from agents.actor_critic.ppo.ppo_agent import PPOAgent
from envs.cart_pole import CartPoleEnv
from utils.plots import plot_learning_history


def run_ppo(
        agent,
        env,
        num_episodes=500,
        max_steps=500,
        render=False
) :
    history = []

    for ep in range(num_episodes):
        states = []
        actions = []
        rewards = []
        log_probs = []
        values = []
        dones = []

        state, _ = env.reset()

        for step in range(max_steps):
            if render:
                env.render()
            
            action, log_prob, value, _ = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)
            dones.append(done)

            state = next_state
            if done:
                break

        returns = agent.compute_returns(rewards, dones, values)
        values = torch.tensor(values)
        advantages = returns - values

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions)
        old_log_probs = torch.stack(log_probs)

        loss = agent.update(
            states,
            actions,
            old_log_probs,
            returns,
            advantages
        )

        ep_reward = sum(rewards)
        history.append((step + 1, ep_reward))

        print(f"Episode {ep} : Reward {ep_reward}")

    return history


if __name__ == "__main__":
    env = CartPoleEnv(render_mode="human")
    agent = PPOAgent(env)

    history = run_ppo(agent, env)

    plot_learning_history(history, "cart_pole", "ppo", "experiments/latest/ppo")