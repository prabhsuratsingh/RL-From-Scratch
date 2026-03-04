from agents.policy_gradient.reinforce_agent import ReinforceAgent
from envs.cart_pole import CartPoleEnv
from utils.plots import plot_learning_history, plot_multiple_learning_history


def run_reinforce(
        agent,
        env,
        num_episodes=500,
        max_steps=500,
        render=False
) :
    history, losses = [], []

    for ep in range(num_episodes):
        state, _ = env.reset()
        ep_reward = 0


        for step in range(max_steps):
            if render:
                env.render()
            
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            agent.store_reward(reward)

            state = next_state
            ep_reward += reward

            if terminated or truncated:
                break
        
        loss = agent.update()
        losses.append(loss)
        history.append((step + 1, ep_reward))

        print(f"Episode {ep}: Reward {ep_reward}")
    
    return history, losses

# if __name__ == "__main__":
#     env = CartPoleEnv(render_mode="human")
#     #vanilla reinforce
#     agent = ReinforceAgent(env)

#     #with baseline
#     agent = ReinforceAgent(env, use_baseline=True)

#     #with entropy
#     agent = ReinforceAgent(env, use_entropy=True)

#     #emtropy + baseline
#     agent = ReinforceAgent(env, use_baseline=True, use_entropy=True, entropy_coeff=0.01)

#     history, losses = run_reinforce(agent, env, render=True)

#     env.close()
#     plot_learning_history(history, "cart_pole", "reinforce", save_dir="experiments/latest/reinforce")

if __name__ == "__main__":

    variants = {
        "reinforce": {},
        "reinforce_baseline": {
            "use_baseline": True
        },
        "reinforce_entropy": {
            "use_entropy": True,
            "entropy_coeff": 0.01
        },
        "reinforce_baseline_entropy": {
            "use_baseline": True,
            "use_entropy": True,
            "entropy_coeff": 0.01
        }
    }

    results = {}

    for name, params in variants.items():

        print(f"\nRunning {name}\n")

        env = CartPoleEnv()

        agent = ReinforceAgent(env, **params)

        history, losses = run_reinforce(agent, env)

        env.close()

        results[name] = history

    plot_multiple_learning_history(
        results,
        "cart_pole",
        "reinforce_variants",
        save_dir="experiments/latest/reinforce"
    )