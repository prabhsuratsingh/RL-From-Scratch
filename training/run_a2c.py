from agents.actor_critic.a2c_agent import A2CAgent
from envs.cart_pole import CartPoleEnv
from utils.plots import plot_learning_history


def run_a2c(
        agent,
        env,
        num_episodes=500,
        max_steps=500,
        render=False
) :
    history = []

    for ep in range(num_episodes):
        state, _ = env.reset()
        ep_reward = 0

        for step in range(max_steps):
            if render:
                env.render()
            
            action, log_prob, entropy = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.learn(
                state,
                log_prob,
                entropy,
                reward,
                next_state,
                done
            )

            state = next_state
            ep_reward += reward

            if done:
                break
        
        history.append((step + 1, ep_reward))

        print(f"Episode {ep}: Reward {ep_reward}")
    
    return history

if __name__ == "__main__":
    env = CartPoleEnv(render_mode="human")
    agent = A2CAgent(env)

    history = run_a2c(agent, env, render=True)

    env.close()
    plot_learning_history(history, "cart_pole", "a2c", save_dir="experiments/latest/actor_critic")
