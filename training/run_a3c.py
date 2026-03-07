import torch
import torch.multiprocessing as mp

from agents.actor_critic.a3c.a3c_network import ActorCritic
from agents.actor_critic.a3c.a3c_worker import A3CWorker
from utils.plots import plot_learning_history

def run_a3c(env_fn, num_workers=4):
    env = env_fn()

    manager = mp.Manager()
    shared_history = manager.list()
    episode_counter = mp.Value("i", 0)

    state_dim = env.observation_space.shape[0]
    nA = env.action_space.n

    global_net = ActorCritic(state_dim, nA)
    global_net.share_memory()

    optimizer = torch.optim.Adam(global_net.parameters(), lr=1e-4)

    workers = []
    for i in range(num_workers):

        worker = A3CWorker(
            global_net,
            optimizer,
            env_fn,
            gamma=0.99,
            worker_id=i,
            history=shared_history,
            episode_counter=episode_counter,
            max_episodes=500
        )

        worker.start()
        workers.append(worker)
    
    for worker in workers:
        worker.join()
    
    return shared_history

if __name__ == "__main__":
    from envs.cart_pole import CartPoleEnv
    mp.set_start_method("spawn")

    shared_history = run_a3c(lambda: CartPoleEnv())
    history = list(shared_history)

    plot_learning_history(
        history,
        "cart_pole",
        "a3c",
        save_dir="experiments/latest/actor_critic"
    )