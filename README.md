# On the Effective Horizon of Inverse Reinforcement Learning

This repository contains the source code for experiments investigating the **Effective Horizon** in Inverse Reinforcement Learning (IRL). In our [accompanying paper](https://arxiv.org/abs/2307.06541), we show that the effective horizon governs policy complexity in IRL. In practice, choosing a smaller horizon (or equivalently, a larger discount factor) than the ground truth can mitigate overfitting—particularly when expert demonstration data is scarce.

## Overview

To test this claim, we:

1. **Implemented four task environments:**
   - Two variants of *Gridworld* (simple and hard).
   - Two variants of *Objectworld* (linear and non-linear).
2. **Adapted two classical IRL algorithms**—Linear-Programming IRL and Maximum Entropy IRL—to handle:
   - Partial expert demonstrations.
   - Variable horizon (or discount) settings.
3. **Optimized the computation** by implementing these methods in JAX, allowing us to quickly iterate over discount/horizon values to find an optimal choice.

---

## Installation

All dependencies are listed in **`requirements.txt`**. To install them, run:

```bash
pip install -r requirements.txt
```

---

## Usage

We provide four environment families:

- `gridworld-simple`
- `gridworld-hard`
- `objectworld-linear`
- `objectworld-non_linear`

For each environment, we sweep over a range of discount factors (or horizons) and evaluate the resulting policy to determine the effective horizon. We support three run modes:

1. **`single`**: Evaluate a single MDP.
2. **`batch`**: Evaluate a batch of N MDPs with **oracle** error counts.
3. **`cross`**: Evaluate a batch of N MDPs with **cross-validation** error counts.

### Running Linear-Programming IRL

To run Linear-Programming IRL for these modes, use:

```bash
# Single MDP over 60 gammas
python effective_horizon_lp.py -t gridworld-simple -m single -n 60 -N 1

# Batch of 10 MDPs with oracle error over 60 gammas
python effective_horizon_lp.py -t gridworld-simple -m batch -n 60 -N 10

# Batch of 10 MDPs with cross-validation over 60 gammas
python effective_horizon_lp.py -t gridworld-simple -m cross -n 60 -N 10
```

### Running Maximum-Entropy IRL

Similarly, for Maximum-Entropy IRL:

```bash
# Single MDP over 17 different horizons
python effective_horizon_maxent.py -t gridworld-simple -m single -n 17 -N 1

# Batch of 10 MDPs with oracle error over 17 different horizons
python effective_horizon_maxent.py -t gridworld-simple -m batch -n 17 -N 10

# Batch of 10 MDPs with cross-validation over 17 different horizons
python effective_horizon_maxent.py -t gridworld-simple -m cross -n 17 -N 10
```

### Demo Notebook
For a quick start and example usage, see the 
[demo notebook](./demo_notebook.ipynb).

It demonstrates:

- Creating a simple Gridworld environment.
- Generating partial expert demonstrations.
- Running both Linear Programming IRL and Maximum Entropy IRL.
- Visualizing the learned reward functions.

### Output and Plotting

- All results (reward functions, value functions, policies, error curves, etc.) are saved under:
  ```
  ./output/{env}/{method}/{mode}
  ```
- Plots of the error or performance vs. gamma/horizon values are also generated and stored in the same directory.
- Additional plotting and analysis utilities are available in `irl/utils.py`.

You can generate summary plots across tasks using:
- `run_detailed_plots()` to produce per-task plots.
- `run_submission_summary_plots()` for the style used in our AAMAS submission.

Plots are saved in the `./plots` directory.

---



## Environments

### Gridworld

A Gridworld environment is defined on an N × N grid with randomly sampled goal states. By default, rewards are sparse (goal states have positive reward; others have zero).

**Key Attributes**  
- **`Gridworld(grid_size, reward_model, seed)`**  
  - `grid_size`: Integer size of the grid.  
  - `reward_model`: Either `"simple"` or `"hard"`, indicating the difficulty or complexity of the reward layout.  
  - `seed`: Random seed for reproducibility.

**Methods**  
- **`get_optimal_policy()`**  
  Returns the optimal action for each state.
- **`get_optimal_value()`**  
  Returns the optimal value for each state.
- **`generate_expert_demonstrations(m_expert, cross_validate_ratio)`**  
  Samples `m_expert` optimal state-action pairs and splits them into training/validation sets by `cross_validate_ratio`.
- **`generate_all_trajectories(n_states, trajectory_length)`**  
  Generates `n_states` trajectories of length `trajectory_length` from randomly chosen initial states under the optimal policy.
- **`evaluate_learnt_reward(reward, discount)`**  
  Given a reward function and discount factor, computes the policy and value function and returns performance metrics.

**Example Usage**:

```python
from irl.mdp.gridworld import Gridworld

world = Gridworld(grid_size=10, reward_model="simple", seed=0)

n_trajectories, trajectory_length = 10, 20
trajectories = world.generate_all_trajectories(
    n_trajectories,
    trajectory_length
)
```

---

### Objectworld

Objectworld is a variant of Gridworld with 10 randomly placed objects, each having an *inner color* and an *outer color*. The reward function can be linear or non-linear combinations of these object attributes.

**Key Attributes**  
- **`Objectworld(grid_size, reward_model, seed)`**  
  - `grid_size`: Grid size.
  - `reward_model`: `"linear"` or `"non_linear"`.
  - `seed`: Random seed.

**Example Usage**:

```python
from irl.mdp.objectworld import Objectworld

world = Objectworld(grid_size=10, reward_model="non_linear", seed=0)
trajectories = world.generate_all_trajectories(
    n_trajectories=10,
    trajectory_length=20,
)
```

---

## IRL Methods

### Linear-Programming IRL

We adapt the Linear-Programming IRL approach from **Ng & Russell (2000)** to:

1. Handle partial expert demonstrations (sample-based transition estimation).  
2. Treat discount factor as a variable.  
3. Remove L1 regularization to avoid reward confounding.  
4. Use stricter constraints to ensure a unique optimal policy.

**Example**:

```python
from effective_horizon_lp import lp_env_init, lp_irl

# Initialize environment and load partial demonstrations
world_env, ground_r, expert_policy, expert_demonstrated_states = lp_env_init(
    task="gridworld",
    expert_fraction=0.5,  # 50% of expert data
    seed=42,
    reward_model="simple"
)

training_discount = 0.5  # Example discount factor

estimate_r, learned_v, learned_policy = lp_irl(
    world_env,
    expert_demonstrated_states,
    training_discount,
    slack_variable=0.001
)
```

---

### Maximum Entropy IRL

We implement **Maximum Entropy IRL (Ziebart et al., 2008)** with modifications for partial demonstrations and variable horizons (by slicing or truncating trajectories).

**Example**:

```python
from effective_horizon_maxent import load_maxent_expert, maxent_irl
import numpy.random as rn
import numpy as np

# Load environment, expert transitions, etc.
transition_fn, _, _, trajs, feature_matrix, n_actions, n_states, _ = load_maxent_expert(
    task_env="gridworld", 
    mode="single", 
    reward_model="simple"
)

# Train with a finite horizon
training_horizon = 10

# extract training traj with N randomly sample initial states and truncate the trajectories at the given training horizon
init_states = rn.choice(range(100), 10, replace=False)
training_trajs = np.array([trajs[i] for i in init_states])[:, :training_horizon]

estimate_r, learned_v, policy = maxent_irl(
    feature_matrix, 
    n_states, 
    n_actions, 
    transition_fn,
    training_trajs,
    epochs=200, 
    learning_rate=0.01, 
    finite_horizon=training_horizon
)
```

This code is adapted from [MatthewJA/Inverse-Reinforcement-Learning](https://github.com/MatthewJA/Inverse-Reinforcement-Learning):

```bibtex
@misc{alger16,
  author       = {Matthew Alger},
  title        = {Inverse Reinforcement Learning},
  year         = 2016,
  doi          = {10.5281/zenodo.555999},
  url          = {https://doi.org/10.5281/zenodo.555999}
}
```

---

## Citation

If you find this repository useful in your research, please cite our work:

```bibtex
@inproceedings{xu2025effective,
  title={On the Effective Horizon of Inverse Reinforcement Learning},
  author={Xu, Yiqing and Doshi-Velez, Finale and Hsu, David},
  booktitle={International Conference on Autonomous Agents and Multiagent Systems},
  year={2025}
}
```

---

Feel free to open issues or pull requests if you encounter any problems or have suggestions!