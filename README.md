# Reinforcement Learning Playground

This project is a lightweight reinforcement learning framework built from scratch, featuring custom environments, tabular RL algorithms, and a simple command-line interface for running experiments. It is designed for learning, experimentation, and rapid prototyping rather than performance or large-scale training.

## Features

* Custom RL environments (no external RL libraries required)
* Tabular RL algorithms (SARSA, Q-learning)
* Simple, extensible CLI interface
* Training visualization via learning curves
* Clear separation between environments, agents, algorithms, and runners

## Project Structure

```
.
├── agents/
│   └── agent.py
├── algorithms/
│   ├── sarsa.py
│   └── qlearning.py
│   └── expected_sarsa.py
├── envs/
│   ├── gridworld.py
│   └── cliff_walking.py
├── runners/
│   └── experiment.py
├── utils/
│   └── plots.py
├── experiments/
│   └── (saved plots)
├── rl.py
└── README.md
```

## Environments

The following environments are currently supported:

* `grid_world` – A classic GridWorld environment
* `cliff_walking` – Cliff Walking environment similar to the Sutton & Barto example

Environments are registered in `ENV_REGISTRY` inside `runners/experiment.py`.

## Algorithms

The following algorithms are implemented:

* `sarsa` – On-policy temporal-difference control
* `q_learning` – Off-policy temporal-difference control

Algorithms are registered in `ALG_REGISTRY` inside `runners/experiment.py`.

## Command-Line Interface

The main entry point is the CLI defined in `rl.py`.

### Usage

```bash
python rl.py experiment --env <env_name> --alg <alg_name> [--render]
```

### Arguments

* `--env` (required): Name of the environment

  * Options: `grid_world`, `cliff_walking`
* `--alg` (required): Name of the algorithm

  * Options: `sarsa`, `q_learning`, `expected_sarsa`
* `--render` (optional): Render the environment using human mode

### Examples

Run Q-learning on GridWorld:

```bash
python rl.py experiment --env grid_world --alg q_learning
```

Run SARSA on Cliff Walking with rendering enabled:

```bash
python rl.py experiment --env cliff_walking --alg sarsa --render
```

## How It Works

1. The CLI parses command-line arguments.
2. `run_experiment` validates the environment and algorithm.
3. The environment is instantiated.
4. An `Agent` is created and bound to the environment.
5. The selected algorithm runs training.
6. Learning history is collected.
7. A learning curve is plotted and saved to the `experiments/` directory.

## Output

After training, a plot of the learning history (e.g., episode rewards) is saved under:

```
experiments/<env_name>_<alg_name>.png
```

## Extending the Project

### Adding a New Environment

1. Create a new environment class in `envs/`
2. Add it to `ENV_REGISTRY`:

   ```python
   ENV_REGISTRY["my_env"] = MyEnvClass
   ```

### Adding a New Algorithm

1. Implement the algorithm in `algorithms/`
2. Register it in `ALG_REGISTRY`:

   ```python
   ALG_REGISTRY["my_alg"] = run_my_algorithm
   ```

## Dependencies

* Python 3.8+
* matplotlib (for plotting)

No external reinforcement learning libraries are required.

## Purpose

This project is intended for:

* Learning reinforcement learning fundamentals
* Understanding tabular RL algorithms end-to-end
* Experimenting with custom environments
* Educational and research prototyping

## License

MIT License
