from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def plot_learning_history(history, env_name, alg_name, save_dir="experiments"):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(1, figsize=(14, 10))

    ax = fig.add_subplot(2, 1, 1)
    episodes = np.arange(len(history))
    moves = np.array([h[0] for h in history])
    plt.plot(episodes, moves, lw=4, marker="o", markersize=10)
    ax.tick_params(axis="both", which="major", labelsize=15)
    plt.xlabel("Episodes", size=20)
    plt.ylabel("# moves", size=20)

    ax = fig.add_subplot(2, 1, 2)
    rewards = np.array([h[1] for h in history])
    plt.step(episodes, rewards, lw=4)
    ax.tick_params(axis="both", which="major", labelsize=15)
    plt.xlabel("Episodes", size=20)
    plt.ylabel("Final rewards", size=20)

    filename = f"{env_name}-{alg_name}-history.png"
    plt.savefig(save_dir / filename, dpi=300)
    plt.show()


def plot_multiple_learning_history(histories, env_name, alg_name, save_dir="experiments"):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(14, 8))

    for name, history in histories.items():

        episodes = np.arange(len(history))
        rewards = np.array([h[1] for h in history])

        plt.plot(episodes, rewards, label=name)

    plt.xlabel("Episodes", fontsize=16)
    plt.ylabel("Episode Reward", fontsize=16)
    plt.title(f"{env_name} - {alg_name}", fontsize=18)

    plt.legend()
    plt.grid(True)

    filename = f"{env_name}-{alg_name}-comparison.png"
    plt.savefig(save_dir / filename, dpi=300)
    plt.show()
