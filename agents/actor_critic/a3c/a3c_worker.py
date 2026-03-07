import torch
import torch.multiprocessing as mp

class A3CWorker(mp.Process):
    def __init__(
            self,
            global_net,
            optimizer,
            env_fn,
            gamma,
            worker_id,
            history,
            episode_counter,
            max_episodes,
            max_steps=500
    ):
        super().__init__()

        self.global_net = global_net
        self.optimizer = optimizer
        self.env = env_fn()

        self.gamma = gamma
        self.worker_id = worker_id
        self.history = history
        self.episode_counter = episode_counter
        self.max_episodes = max_episodes
        self.max_steps = max_steps

        self.local_net = type(global_net)(
            self.env.observation_space.shape[0],
            self.env.action_space.n
        )

        self.local_net.load_state_dict(self.global_net.state_dict())

    def choose_action(self, state):
        state_t = torch.tensor(state, dtype=torch.float32)

        probs, value = self.local_net(state_t)
        dist = torch.distributions.Categorical(probs)

        action = dist.sample()

        return action.item(), dist.log_prob(action), dist.entropy(), value
    
    def run(self):

        state, _ = self.env.reset()

        while True:

            if self.episode_counter.value >= self.max_episodes:
                break

            log_probs = []
            values = []
            rewards = []
            entropies = []

            for step in range(self.max_steps):

                action, log_prob, entropy, value = self.choose_action(state)

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                log_probs.append(log_prob)
                values.append(value)
                rewards.append(reward)
                entropies.append(entropy)

                state = next_state

                if done:

                    episode_length = step + 1
                    episode_reward = sum(rewards)

                    with self.episode_counter.get_lock():
                        self.episode_counter.value += 1
                        ep = self.episode_counter.value

                    self.history.append((episode_length, episode_reward))

                    print(
                        f"Worker {self.worker_id} "
                        f"Episode {ep} "
                        f"Reward {episode_reward} "
                        f"Length {episode_length}"
                    )

                    state, _ = self.env.reset()
                    break

            self.update(log_probs, values, rewards, entropies, done, state)
    
    def update(
            self,
            log_probs,
            values,
            rewards,
            entropies,
            done,
            next_state
    ):
        if done:
            R = 0
        else:
            _, value = self.local_net(torch.tensor(next_state, dtype=torch.float32))
            R = value.detach()
        
        returns = []

        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns)

        values = torch.stack(values)
        log_probs = torch.stack(log_probs)
        entropies = torch.stack(entropies)

        advantage = returns - values

        policy_loss = -(log_probs * advantage.detach()).mean()
        value_loss = advantage.pow(2).mean()
        entropy_loss = -0.01 * entropies.mean()

        loss = policy_loss + 0.5 * value_loss + entropy_loss

        self.optimizer.zero_grad()
        loss.backward()

        for local_param, global_param in zip(
            self.local_net.parameters(),
            self.global_net.parameters()
        ):
            global_param._grad = local_param.grad
        
        self.optimizer.step()

        self.local_net.load_state_dict(self.global_net.state_dict())

