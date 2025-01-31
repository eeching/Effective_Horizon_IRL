import sys
import math
import pickle
import getopt
import pdb
import tqdm

# Third-party
import numpy as np
import numpy.random as rn
import jax.numpy as jnp
import matplotlib.pyplot as plt

from irl.utils import (
    plot_error_curve,
    plot_batch_error_curve,
    plot_cross_validation_curve,
    make_dir
)
import irl.value_iteration_jax as value_iteration
import irl.maxent_jax as maxent
from irl.mdp.objectworld import Objectworld
from irl.mdp.gridworld import Gridworld

gt_T = 20
shortest_horizon = 4

def maxent_parse(argv):
    """
    Parse command-line arguments for the Maximum Entropy IRL experiment.

    Args:
        argv (list[str]): Command line arguments, typically sys.argv.

    Returns:
        tuple: (task_env, reward_model, mode, n_horizons, n_mdps)
          - task_env (str): The environment name ("gridworld" or "objectworld", etc.)
          - reward_model (str): The reward model type (e.g., "linear", "nonlinear")
          - mode (str): The IRL experiment method ("single", "batch", "cross")
          - n_horizons (int): Number of horizons to test (for batch or cross-validation).
          - n_mdps (int): Number of distinct MDPs to process (for batch or cross-validation).
    """
    arg_task = ""
    arg_mode = ""
    arg_n_horizons = ""
    arg_n_mdps = ""

    arg_help = f"{argv[0]} -t <task> -m <mode> -n <nhorizons> -N <nmdps>"
    try:
        opts, _ = getopt.getopt(
            argv[1:], "h:t:m:n:N:", ["help", "task=", "mode=", "nhorizons=", "nmdps="]
        )
    except getopt.GetoptError:
        print(arg_help)
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(arg_help)
            sys.exit(0)
        elif opt in ("-t", "--task"):
            arg_task = arg
        elif opt in ("-m", "--mode"):
            arg_mode = arg
        elif opt in ("-n", "--nhorizons"):
            arg_n_horizons = arg
        elif opt in ("-N", "--nmdps"):
            arg_n_mdps = arg
    # Expecting something like "gridworld_linear" => split into (gridworld, linear)
    try:
        task_env, reward_model = arg_task.split("-")
    except ValueError:
        print("ERROR: Task argument must be in the form 'task-rewardmodel'.")
        sys.exit(2)

    return task_env, reward_model, arg_mode, int(arg_n_horizons), int(arg_n_mdps)

def load_maxent_expert(task_env, mode, reward_model):
    """
    Load stored expert demonstration data for MaxEnt IRL from a pickle file

    The pickle file is expected to contain:
      - A dictionary with:
          "T": the transition function (will be popped out and returned separately)
          {seed_value}: a sub-dictionary with fields like:
              "gt_r", "expert_policy", "trajectories", 
              "feature_matrix", "n_actions", "n_states", "opt_v", etc.

    Depending on the 'mode' argument, this function returns different structures:
    Args:
        task_env (str): Environment name, e.g. "objectworld" or "gridworld".
        mode (str): "single" or any other string (commonly "batch").
        reward_model (str): The reward model label (e.g. "hard", "soft").
    """
    # Load the pickle file
    load_path = f"./expert_demo/{task_env}/reward_model_{reward_model}.pkl"

    with open(load_path, 'rb') as fp:
        data = pickle.load(fp)

    # Extract and remove the transition function from the dictionary
    transition_function = np.array(data.pop("T"))

    # Collect all seed keys
    seeds = list(data.keys())
    if not seeds:
        raise ValueError(f"No seeds found in {load_path} for environment '{task_env}'.")

    # Decide on return structure based on 'mode'
    if mode == "single":
        if task_env == "objectworld":
            seed = seeds[0]
        else:  
            seed = seeds[2]

        env_data = data[seed]

        ground_r = np.array(env_data["gt_r"])
        expert_policy = np.array(env_data["expert_policy"])
        trajectories = env_data["trajectories"]
        feature_matrix = np.array(env_data["feature_matrix"])
        n_actions = env_data["n_actions"]
        n_states = env_data["n_states"]
        opt_v = env_data["opt_v"]

        return (transition_function, ground_r, expert_policy,
                trajectories, feature_matrix, n_actions,
                n_states, opt_v)

    else:
        seed_itr = iter(seeds)
        return seed_itr, transition_function, data

def get_expert_trajectories(data, seed):
    """
    Retrieve environment data for a specific seed from the provided dictionary.

    Args:
        data (dict): A mapping of seed -> environment data dictionary.
        seed (int): The specific seed key to extract from 'data'.

    Returns:
        tuple: (expert_policy, trajectories, feature_matrix, n_actions, 
                n_states, ground_r, opt_v)
            - expert_policy (np.ndarray): 1D or 2D array of actions for each state.
            - trajectories (list or np.ndarray): Expert demonstration trajectories.
            - feature_matrix (np.ndarray): The feature matrix for each state.
            - n_actions (int): Number of possible actions in the environment.
            - n_states (int): Number of states in the environment.
            - ground_r (np.ndarray): Ground-truth reward array for each state.
            - opt_v (np.ndarray): The optimal state value function under ground_r.

    Raises:
        KeyError: If the specified 'seed' is not found in 'data'.
    """
    if seed not in data:
        raise KeyError(f"Seed {seed} not found in the provided 'data' dictionary.")

    demo = data[seed]

    expert_policy = np.array(demo["expert_policy"])
    trajectories = demo["trajectories"]
    feature_matrix = np.array(demo["feature_matrix"])
    n_actions = demo["n_actions"]
    n_states = demo["n_states"]
    ground_r = np.array(demo["gt_r"])
    opt_v = np.array(demo["opt_v"])

    return (expert_policy, trajectories, feature_matrix, 
            n_actions, n_states, ground_r, opt_v)

def maxent_irl(
    feature_matrix,
    n_states,
    n_actions,
    transition_function,
    trajectories,
    epochs,
    learning_rate,
    finite_horizon=None,
    gt_alpha=None
):
    """
    Run Maximum Entropy Inverse Reinforcement Learning on the provided environment data,
    then compute the optimal value function and policy from the learned reward.

    Args:
        feature_matrix (np.ndarray): Matrix of shape (n_states, n_features), representing
            the feature vector for each state.
        n_states (int): Number of states in the environment.
        n_actions (int): Number of possible actions.
        transition_function (np.ndarray): Transition probabilities of shape
            (n_states, n_actions, n_states).
        trajectories (list or np.ndarray): Expert demonstration trajectories, 
            each trajectory is a list/array of (state, action) or (state, action, next_state).
        epochs (int): Number of gradient iterations for the MaxEnt IRL algorithm.
        learning_rate (float): Learning rate for the MaxEnt IRL optimization.
        finite_horizon (int, optional): If provided, use this horizon for value iteration; 
            otherwise, treat it as infinite horizon or a default.
        gt_alpha (float, optional): If specified, may be used internally by the MaxEnt IRL 
            solver (e.g., for regularization or known mixture weight). 

    Returns:
        tuple: (r, v_opt, policy) where
            r (np.ndarray): The learned reward array of shape (n_states,).
            v_opt (np.ndarray): The optimal value function of shape (n_states,).
            policy (np.ndarray): The deterministic policy, shape (n_states,).

    Note:
        - This function calls `maxent.irl(...)` to learn the reward.
        - It then uses `value_iteration.optimal_value(...)` and
          `value_iteration.find_policy(...)` to compute the value function and policy
          from the learned reward.
        - The discount factor is set to 1.0 (training_discount = 1).
    """
    # IRL to obtain the learned reward
    discount_factor = 1.0
    r_learned = maxent.irl(
        feature_matrix,
        n_actions,
        discount_factor,
        transition_function,
        trajectories,
        epochs,
        learning_rate,
        finite_horizon,
        gt_alpha=gt_alpha
    )

    # Compute the optimal value function for the learned reward
    v_opt = value_iteration.optimal_value(
        n_states,
        n_actions,
        transition_function,
        r_learned,
        discount_factor,
        T=finite_horizon
    )

    # Determine the (deterministic) policy based on that value function
    policy, _ = value_iteration.find_policy(
        n_states,
        n_actions,
        transition_function,
        r_learned,
        discount_factor,
        v_opt,
        stochastic=False,
        T=finite_horizon
    )

    return np.asarray(r_learned), np.asarray(v_opt), np.asarray(policy)

def single_mdp(
    task_env: str,
    expert_frac: float,
    reward_model: str,
    n_horizons: int,
    log_dir: str,
    epochs: int = 200,
    learning_rate: float = 0.01,
    finite_horizon: int = gt_T
) -> None:
    """
    Run maxent inverse reinforcement learning on a gridworld MDP for a single environment.

    This function loads expert demonstrations for the specified environment, trains a maximum 
    entropy inverse reinforcement learning (MaxEnt IRL) model to recover the reward function from 
    a fraction of expert trajectories, then evaluates the learned policy. Various plots and 
    metrics are saved to the given log directory.

    Args:
        task_env (str): Identifier of the environment or task to be loaded.
        expert_frac (float): Fraction of expert states (grid cells) to use for IRL training.
        reward_model (str): Specifies the type of reward model.
                           "hard" => uses grid_size = 15.
                           otherwise => uses grid_size = 10.
        n_horizons (int): Number of horizons to test (for batch or cross-validation).
        log_dir (str): Directory path for saving outputs, including figures and pickled results.
        epochs (int, optional): Number of epochs to run the maxent IRL training. Defaults to 200.
        learning_rate (float, optional): Learning rate for the IRL training. Defaults to 0.01.
        finite_horizon (int, optional): Planning horizon for the environment. Defaults to gt_T.

    Returns:
        None. This function saves plots and pickled data structures as side effects.

    Notes:
        - The function plots the ground truth reward, learned reward, ground truth value function,
          and evaluated value function for different horizons.
        - It also computes and saves the discrepancy between learned and expert policies.
    """

    # Use the specified horizon for trajectory length.
    traj_len = finite_horizon

    # Determine grid size based on reward model.
    if reward_model == "hard":
        grid_size = 15
    else:
        grid_size = 10

    # Compute the total number of expert states.
    # m_expert is the total number of expert "cells" (states) used across the trajectories.
    m_expert = int(expert_frac * grid_size**2)

    # Load environment details, transition function, ground truth reward, expert policy,
    # expert trajectories, and feature matrix.
    transition_function, ground_r, expert_policy, trajectories, feature_matrix, \
        n_actions, n_states, opt_v = load_maxent_expert(task_env, "single", reward_model)
    print("loading expert demo ================")

    # Initialize random seed for reproducibility and select initial states for the expert demos.
    rn.seed(0)
    init_states = rn.choice(range(n_states), n_states, replace=False)

    # Prepare containers to store intermediate results, policies, coverage, etc.
    result = []
    training_result = []
    learned_r = []
    learned_v = []
    eval_v = []
    policies = []
    state_coverage = []
    state_set = []

    # Create figure and subplots to visualize the ground truth and learned quantities.
    fig, axs = plt.subplots(4, 6, layout="constrained", figsize=(25, 10), sharex=True, sharey=True)
    fig.suptitle(f"Performance for each gamma for {expert_frac*100}% expert")

    # Range of different trajectory lengths to test (from 4 to 20, inclusive).
    traj_len_list = np.arange(shortest_horizon, shortest_horizon + n_horizons).tolist()
    assert n_horizons == len(traj_len_list)

    # Plot the ground truth value and reward as the first column in the subplot grid.
    ax = axs[0][0]
    im = ax.pcolor(opt_v.reshape((grid_size, grid_size)))
    plt.colorbar(im, ax=ax)

    ax1 = axs[1][0]
    im1 = ax1.pcolor(ground_r.reshape((grid_size, grid_size)))
    plt.colorbar(im1, ax=ax1)

    # Define arrow directions for policy visualization.
    # The key is the action index, and the value is a (dx, dy) movement.
    arrows = {
        0: (1, 0),
        1: (1, 1),
        2: (0, 1),
        3: (-1, 1),
        4: (-1, 0),
        5: (-1, -1),
        6: (0, -1),
        7: (1, -1),
        8: (0, 0)
    }
    scale = 0.25

    # Visualize the expert policy.
    p = np.array(expert_policy).reshape(grid_size, grid_size)
    ax2 = axs[2][0]
    color_code = {0: "black", 1: "seagreen"}
    for r in range(grid_size):
        for c in range(grid_size):
            action = p[r, c]
            ax2.arrow(c, r, scale * arrows[action][0], scale * arrows[action][1], 
                      head_width=0.1, color=color_code[0])

    # Compute the value of the expert policy under the ground truth reward for plotting.
    gt_v, _ = np.array(value_iteration.value(
        expert_policy, n_states, transition_function, ground_r, 1, T=finite_horizon
    ))
    ax3 = axs[3][0]
    im3 = ax3.pcolor(gt_v.reshape((grid_size, grid_size)))
    plt.colorbar(im3, ax=ax3)
    print("finish computing value")

    # Set titles for the first column subplots.
    ax.set_title("GT V", fontsize='small')
    ax1.set_title("GT reward", fontsize='small')
    ax2.set_title("V induced policy", fontsize='small')
    ax3.set_title("Policy Evaluated Under R and gamma", fontsize='small')

    print(f"expert coverage {expert_frac}")

    # Main loop over possible horizon (trajectory) lengths.
    for i, horizon in enumerate(traj_len_list):
        traj_len = traj_len_list[i]
        # Number of expert trajectories is total expert states divided by length of each trajectory.
        n_traj = int(m_expert / traj_len)

        # Build the training trajectories by slicing the expert trajectories up to traj_len.
        training_trajs = np.array([trajectories[i] for i in init_states[:n_traj]])[:, :traj_len]

        # Compute "expert_cover" as the set of states visited by these demonstration trajectories.
        # A state can appear multiple times if visited repeatedly, which helps account for density.
        expert_cover = training_trajs[:, :, 0].flatten().tolist()

        # Train maxent IRL to learn reward function using the partial expert trajectories.
        r, learned_V, policy = maxent_irl(
            feature_matrix, n_states, n_actions, transition_function, 
            training_trajs, epochs, learning_rate, horizon
        )

        # Measure discrepancy between the learned policy and the expert policy.
        diff = n_states - np.sum(np.equal(policy, expert_policy))
        expert_diff = np.sum([policy[i] != expert_policy[i] for i in expert_cover])
        result.append(diff)
        training_result.append(expert_diff)
        learned_r.append(r)
        learned_v.append(learned_V)
        policies.append(policy)
        state_coverage.append(expert_cover)
        state_set.append(len(set(expert_cover)))

        # Evaluate the learned policy under the learned reward for a fixed horizon (finite_horizon).
        v, _ = np.array(value_iteration.value(
            policy, n_states, transition_function, r, 1, T=finite_horizon
        ))
        eval_v.append(v)

        print("Diff", diff)

        # Periodically plot the learned reward, value, and policy arrows (every 4 steps in horizon).
        if horizon % 4 == 0:
            idx = int(horizon / 4)

            # Plot the learned reward in row 2 of the subplot grid.
            ax2 = axs[1][idx]
            im2 = ax2.pcolor(r.reshape((grid_size, grid_size)))
            plt.colorbar(im2, ax=ax2)

            # Plot the learned value in row 1 of the subplot grid.
            ax3 = axs[0][idx]
            im3 = ax3.pcolor(learned_V.reshape((grid_size, grid_size)))
            plt.colorbar(im3, ax=ax3)

            # Set the title based on horizon.
            if finite_horizon is None:
                ax3.set_title(f"Gamma = {gamma}", fontsize='small')
            else:
                ax3.set_title(f"Horizon = {horizon}", fontsize='small')

            # Plot the learned policy with arrows.
            p = np.array(policy).reshape(grid_size, grid_size)
            discrepancy = np.equal(expert_policy, policy).reshape(grid_size, grid_size)
            ax4 = axs[2][idx]
            color_code = {True: "black", False: "silver"}
            color_code_expert = {True: "seagreen", False: "springgreen"}

            # Identify which states are in the expert coverage.
            expert_mask = np.array([1 if i in expert_cover else 0 for i in range(n_states)]).reshape(grid_size, grid_size)
            for rr in range(grid_size):
                for cc in range(grid_size):
                    action = p[rr, cc]
                    if expert_mask[rr, cc] == 1:
                        ax4.arrow(cc, rr, scale * arrows[action][0], scale * arrows[action][1], 
                                  head_width=0.1, color=color_code_expert[discrepancy[rr, cc]])
                    else:
                        ax4.arrow(cc, rr, scale * arrows[action][0], scale * arrows[action][1], 
                                  head_width=0.1, color=color_code[discrepancy[rr, cc]])
            ax4.set_title(f"effective state coverage: {len(set(expert_cover))}", fontsize='small')

            # Plot the evaluated value (under the learned reward) in the bottom row.
            ax5 = axs[3][idx]
            im5 = ax5.pcolor(v.reshape((grid_size, grid_size)))
            plt.colorbar(im5, ax=ax5)

    # Save the figure showing reward/value landscapes for various horizons.
    plt.savefig(f"{log_dir}/expert_{expert_frac}_V_R_.jpg")
    print(result)

    # Pickle dumping all intermediate results into a file.
    with open(f'{log_dir}/expert_{expert_frac}.p', 'wb') as fp:
        pickle.dump(
            {
                "horizon": traj_len_list,
                "error": result,
                "training_error": training_result,
                "learned_r": learned_r,
                "learned_v": learned_v,
                "gt_r": ground_r,
                "gt_v": gt_v,
                "eval_v": eval_v,
                "init_states": init_states,
                "policies": policies,
                "state_coverage": state_coverage,
                "state_set": state_set
            },
            fp
        )

    # Plot and save a curve for the error metric over different horizons.
    output_dir = f"{log_dir}/expert_{expert_frac}_error_curve.jpg"
    plot_error_curve(
        output_dir,
        horizon_list=traj_len_list,
        error=result,
        training_error=training_result,
        state_set=state_set
    )

def batch_mdps(
    task_env: str,
    expert_frac: float,
    n_mdp: int,
    reward_model: str,
    n_horizons: int,
    log_dir: str,
    epochs: int = 200,
    learning_rate: float = 0.01,
    finite_horizon: int = gt_T
) -> None:
    """
    Evaluate maxent IRL across multiple MDPs using a batch of expert demonstrations.

    This function loads expert demonstrations from multiple seeds (each corresponding 
    to a distinct MDP), then for each MDP:
      • It trains a maxent IRL model with a subset of states determined by 'expert_frac'.
      • Varies the horizon length from 4 to 20, evaluates the resulting policy, and 
        measures policy discrepancy relative to the expert policy.
      • Stores results (errors, learned rewards, values) in a pickle file after each MDP 
        and also plots a batch error curve at the end.

    Args:
        task_env (str): Identifier or name of the environment.
        expert_frac (float): Fraction of the total grid cells used as "expert" demonstrations.
        n_mdp (int): Number of distinct MDPs (with different seeds) to process.
        reward_model (str): Either "hard" or another string specifying the difficulty setting. 
                            "hard" uses a 15×15 grid, otherwise 10×10.
        n_horizons (int): Number of horizons to test.
        log_dir (str): Directory path for saving outputs (pickle files, plots).
        epochs (int, optional): Number of training epochs for maxent IRL. Defaults to 200.
        learning_rate (float, optional): Learning rate used in the IRL gradient updates. Defaults to 0.01.
        finite_horizon (int, optional): Planning horizon for value iteration. Defaults to gt_T.

    Returns:
        None. This function saves outputs as side-effects (figures and pickle files).
    """

    # Determine the grid_size based on the reward model.
    if reward_model == "hard":
        grid_size = 15
    else:
        grid_size = 10

    # Number of expert states to consider based on the fraction of total grid cells.
    m_expert = int(expert_frac * grid_size**2)

    # Load the batch expert data and a seed iterator for multiple MDPs.
    # transition_function, data, etc. are used for training IRL on each MDP.
    seed_itr, transition_function, data = load_maxent_expert(task_env, "batch", reward_model)

    # Seed the random number generator for reproducibility of initial state selection.
    rn.seed(42)
    init_states = rn.choice(range(grid_size**2), grid_size**2, replace=True)

    # Prepare a range of horizon lengths (from 4 to 20).
    traj_len_list = np.arange(shortest_horizon, shortest_horizon + n_horizons).tolist()
    assert n_horizons == len(traj_len_list)


    # Allocate containers to store results across MDPs and horizon lengths.
    result = np.zeros((n_mdp, n_horizons))          # Policy discrepancy over the full state space.
    training_result = np.zeros((n_mdp, n_horizons)) # Policy discrepancy over states seen in training.
    learned_r = []                                   # List of learned reward arrays, for each MDP.
    learned_v = []                                   # List of learned value arrays, for each MDP.
    eval_v = []                                      # List of evaluated value arrays, for each MDP.

    # Loop through each MDP (defined by a different random seed).
    for i in range(n_mdp):

        # Retrieve a new seed and the corresponding expert data for this MDP.
        seed = next(seed_itr)
        expert_policy, trajectories, feature_matrix, n_actions, n_states, ground_r, opt_v = get_expert_trajectories(
            data, seed
        )

        # Prepare containers for storing horizon-specific results.
        r_list = []
        v_list = []
        eval_v_list = []

        # Loop over all horizons (4 to 20).
        for j, horizon in enumerate(traj_len_list):

            traj_len = horizon
            # Calculate how many demonstrations to use based on m_expert and the trajectory length.
            n_traj = int(m_expert / traj_len)

            # Slice the expert trajectories to build training data of length 'traj_len'.
            training_trajs = np.array([trajectories[k] for k in init_states[:n_traj]])[:, :traj_len]

            # Track the states covered in these training demos (may contain duplicates).
            expert_cover = training_trajs[:, :, 0].flatten().tolist()

            # Learn reward function and policy with maxent IRL.
            r, v, policy = maxent_irl(
                feature_matrix, n_states, n_actions, transition_function, 
                training_trajs, epochs, learning_rate, horizon
            )

            # Compute policy discrepancy: 
            #   "diff" is the total mismatch across all states,
            #   "expert_diff" is mismatch restricted to visited states in 'expert_cover'.
            diff = n_states - np.sum(np.equal(policy, expert_policy))
            expert_diff = np.sum([policy[i] != expert_policy[i] for i in expert_cover])

            # Store results.
            result[i][j] = diff
            training_result[i][j] = expert_diff
            r_list.append(r)
            v_list.append(v)

            # Evaluate this learned policy with the ground truth reward for the same horizon.
            e_v = np.array(value_iteration.value(policy, n_states, transition_function, ground_r, 1, T=finite_horizon))
            eval_v_list.append(e_v)

            print(f"MDP {i}, horizon {horizon}, error {diff}")

        # Append ground truth reward and value for reference in the last position of the lists.
        r_list.append(ground_r)
        v_list.append(opt_v)

        learned_r.append(r_list)
        learned_v.append(v_list)
        eval_v.append(eval_v_list)

        # Save intermediate results for each MDP to a pickle file.
        with open(f'{log_dir}/{n_mdp}_expert_{expert_frac}.p', 'wb') as fp:
            pickle.dump(
                {
                    "horizon": traj_len_list,
                    "error": result,
                    "learned_r": learned_r,
                    "learned_v": learned_v,
                    "training_error": training_result,
                    "eval_v": eval_v,
                    "n_mdp": i + 1
                },
                fp
            )

    # Print final result arrays for inspection.
    print(result)

    # Plot the aggregated error curve for all MDPs over the horizon range.
    output_dir = f"{log_dir}/MDPs_{n_mdp}_expert_{expert_frac}_error_curve.jpg"
    plot_batch_error_curve(output_dir, horizon_list=traj_len_list, error=result)

def cross_validate_mdps(
    task_env: str,
    expert_frac: float,
    n_mdp: int,
    reward_model: str,
    n_horizons: int,
    log_dir: str,
    epochs: int = 200,
    learning_rate: float = 0.01,
    finite_horizon: int = gt_T
) -> None:
    """
    Perform cross-validation on multiple MDPs using a fraction of expert demonstrations.

    For each MDP (identified by a unique random seed), a portion of expert demonstrations 
    is split into training and validation sets based on the ratio 80%:20%. MaxEnt IRL 
    is trained on the training portion, validated on the held-out portion, and the learned 
    policy is also compared against the ground truth expert policy across all states.

    Results:
      • Ground-truth error  (gt_error): Discrepancy over the entire state space.
      • Training error      (train_error): Discrepancy on the training states.
      • Validation error    (validate_error): Discrepancy on the validation states.
      • Expert error        (expert_error): Combined training + validation discrepancy.

    Args:
        task_env (str): Identifier/name of the environment to load.
        expert_frac (float): Fraction of the total number of states to use as expert data.
        n_mdp (int): Number of MDPs (with distinct random seeds) to process.
        reward_model (str): Either "hard" or another string specifying difficulty. 
                            "hard" => grid_size = 15, else => grid_size = 10.
        n_horizons (int): Number of horizons to test (for batch or cross-validation).
        epochs (int, optional): Number of optimization epochs in the MaxEnt IRL. Defaults to 200.
        learning_rate (float, optional): Learning rate for the IRL gradient updates. Defaults to 0.01.
        finite_horizon (int, optional): Horizon length for value iteration. Defaults to gt_T.
        log_dir (str, optional): Directory to save logs and outputs. Defaults to "./logs".
    
    Returns:
        None. The function logs training progress to the console and saves 
        intermediate results and plots as side effects.
    """

    # Select grid size based on the reward model.
    if reward_model == "hard":
        grid_size = 15
    else:
        grid_size = 10

    # Load expert data for a batch run, including a seed iterator for each MDP.
    seed_itr, transition_function, data = load_maxent_expert(
        task_env, 
        "batch", 
        reward_model
    )

    # Seed for reproducible initialization of states used in building trajectories.
    rn.seed(42)
    init_states = rn.choice(range(grid_size**2), grid_size**2, replace=True)
    
    # Number of expert states to use in total.
    m_expert = int(expert_frac * grid_size**2)

    # Prepare lists of horizon lengths to be tested.
    traj_len_list = np.arange(shortest_horizon, shortest_horizon + n_horizons).tolist()
    assert n_horizons == len(traj_len_list)


    # Arrays for storing errors across all MDPs and horizons.
    gt_error = np.zeros((n_mdp, n_horizons))       # Error over entire state space.
    train_error = np.zeros((n_mdp, n_horizons))    # Error on just train states.
    validate_error = np.zeros((n_mdp, n_horizons)) # Error on validation states.
    expert_error = np.zeros((n_mdp, n_horizons))   # Combined training + validation error.

    # Lists to collect learned rewards, values, and evaluated values.
    learned_r = []
    learned_v = []
    eval_v = []

    # Loop over each MDP to perform cross-validation.
    for i in range(n_mdp):
        # Retrieve a new seed, then load expert demonstrations/features for that seed.
        seed = next(seed_itr)
        expert_policy, trajectories, feature_matrix, n_actions, n_states, ground_r, opt_v = get_expert_trajectories(
            data, 
            seed
        )

        # Temporary lists for rewards, values, and evaluation results across horizons.
        r_list = []
        v_list = []
        eval_v_list = []

        # Sweep through the range of horizon lengths.
        for j, horizon in enumerate(traj_len_list):

            # Determine how many trajectories go into training vs validation.
            traj_len = horizon
            n_train_traj = int(m_expert * 0.8 / traj_len)
            n_val_traj = int(m_expert * 0.2) // traj_len
            r_val_traj = int(m_expert * 0.2) % traj_len
            
            # Build the training trajectories; each is sliced to the current horizon.
           
            training_trajs = np.array([trajectories[k] for k in init_states[:n_train_traj]])[:, :traj_len]
            train_states = training_trajs[:, :, 0].flatten().tolist()

            # Build the validation states from the remaining portion of expert demonstrations.
            if n_val_traj == 0:
                # If we have no full trajectory for validation, use the leftover part.
                val_states = np.array(
                    trajectories[init_states[n_train_traj]]
                )[:r_val_traj, 0].flatten().tolist()
            else:
                # Otherwise, we can use n_val_traj full trajectories.
                val_states = np.array([
                    trajectories[k] for k in init_states[n_train_traj : n_train_traj + n_val_traj]
                ])[:, :traj_len, 0].flatten().tolist()
                
                # If there's a leftover portion, we also add it to val_states.
                if r_val_traj > 0:
                    val_states += np.array(
                        trajectories[init_states[n_train_traj + n_val_traj]]
                    )[:r_val_traj, 0].flatten().tolist()

            # Learn the reward function and policy via MaxEnt IRL on the training subset.
            r, v, policy = maxent_irl(
                feature_matrix, 
                n_states, 
                n_actions, 
                transition_function,
                training_trajs, 
                epochs, 
                learning_rate, 
                horizon
            )

            # Compute training, validation, and ground-truth errors. 
            train_diff = np.sum([policy[state_idx] != expert_policy[state_idx] for state_idx in train_states])
            val_diff   = np.sum([policy[state_idx] != expert_policy[state_idx] for state_idx in val_states])
            gt_diff    = (n_states - np.sum(np.equal(policy, expert_policy)))

            # Record the error metrics.
            gt_error[i][j]       = gt_diff
            train_error[i][j]    = train_diff
            validate_error[i][j] = val_diff
            expert_error[i][j]   = train_diff + val_diff

            # Store learned reward and value.
            r_list.append(r)
            v_list.append(v)

            # Evaluate the learned policy using the ground truth reward.
            e_v = np.array(value_iteration.value(policy, n_states, transition_function, ground_r, 1, T=finite_horizon))
            eval_v_list.append(e_v)

            print(f"MDP {i}, horizon {horizon}, val error {val_diff}, expert_error {train_diff}, gt_error {gt_diff}")

        # Append the ground truth reward and optimal value at the end for reference.
        r_list.append(ground_r)
        v_list.append(opt_v)

        # Collect all learned rewards, values, and evaluations for this MDP.
        learned_r.append(r_list)
        learned_v.append(v_list)
        eval_v.append(eval_v_list)
        
        # Save intermediate results for this MDP to a pickle file.
        with open(f'{log_dir}/{n_mdp}_expert_{expert_frac}.p', 'wb') as fp:
            pickle.dump(
                {
                    "horizon": traj_len_list,
                    "gt_error": gt_error,
                    "val_error": validate_error,
                    "expert_error": expert_error,
                    "train_error": train_error,
                    "learned_r": learned_r,
                    "learned_v": learned_v,
                    "eval_v": eval_v
                },
                fp
            )

    # Generate and save a plot that visualizes the cross-validation errors across horizons.
    output_dir = f"{log_dir}/MDPs_{n_mdp}_expert_{expert_frac}_error_curve.jpg"
    plot_cross_validation_curve(
        output_dir, 
        horizon_list=traj_len_list, 
        gt_error=gt_error, 
        val_error=validate_error, 
        expert_error=expert_error, 
        train_error=train_error
    )

def cache_expert_demo(
    task: str,
    n_mdp: int,
    reward_model: str,
    log_dir: str
) -> None:
    """
    Cache expert demonstrations for a specified environment and reward model.

    This function:
      1) Creates several MDP instances (identified by random seeds). 
      2) For each MDP, initializes the environment (Objectworld or Gridworld),
         retrieves the ground truth reward, value, and policy. 
      3) Generates fixed-length trajectories assuming all states as possible initial states. 
      4) Visualizes ground truth value, reward, and policy arrows, optionally also 
         objectworld-specific attributes if the task is "objectworld".
      5) Aggregates and pickles all environment data (trajectories, feature matrix, etc.) 
         in a dictionary for future use.

    Args:
        task (str): The environment type, e.g. "objectworld" or "gridworld".
        n_mdp (int): Number of distinct MDP instances to create (using different seeds).
        reward_model (str): Specifies the environment's reward function (e.g., "hard" or another).
        log_dir (str): Directory path where figures and pickled data are saved.

    Returns:
        None. The function saves both a figure (aggregating MDP visualizations) and a 
        pickle file containing expert demonstration information to 'log_dir'.
    """

    # Depending on the task type, prepare subplots with a different number of rows.
    if task == "objectworld":
        fig, axs = plt.subplots(
            nrows=5, ncols=n_mdp, layout="constrained", figsize=(35, 7),
            sharex=True, sharey=True
        )
    elif task == "gridworld":
        fig, axs = plt.subplots(
            nrows=3, ncols=n_mdp, layout="constrained", figsize=(35, 10),
            sharex=True, sharey=True
        )
    else:
        raise ValueError(f"Unknown task environment: {task}")

    # Determine grid size based on reward model.
    if reward_model == "hard":
        grid_size = 15
    else:
        grid_size = 10

    # Fix a random seed and pick n_mdp distinct seeds for MDP initialization.
    rn.seed(0)
    seeds = rn.choice(range(grid_size**2), n_mdp, replace=False)

    # Dictionary to collect environment data for all MDPs.
    demo = {}

    # Fixed horizon for trajectory generation.
    traj_len = 25

    # Create each MDP instance.
    for i in range(n_mdp):
        print("===============", i)
        seed = seeds[i]

        # Initialize the environment according to the task.
        if task == "objectworld":
            world_env = Objectworld(
                seed=seed, reward_model=reward_model, finite_horizon=traj_len
            )
        elif task == "gridworld":
            world_env = Gridworld(
                seed=seed, demo=(i == 0), reward_model=reward_model, finite_horizon=traj_len
            )

        # Extract the transition probability for reference.
        # Stored under key 'T' in the dictionary. Note that with each loop, this overwrites demo['T'].
        # If you want a distinct T for each MDP, store it under a seed-specific subkey.
        T = world_env.transition_probability
        demo['T'] = T

        # Ground truth reward array.
        ground_r = world_env.reward_array

        # Generate trajectories of length 'traj_len' using all states as initial states.
        trajectories = world_env.generate_all_trajectories(world_env.n_states, traj_len)

        # Build up the feature matrix. For gridworld, we use "ident"; for objectworld, discrete features.
        if task == "gridworld":
            feature_matrix = world_env.feature_matrix(feature_map="ident")
            demo[seed] = {
                "gt_r": ground_r,
                "expert_policy": world_env.policy,
                "trajectories": trajectories,
                "n_actions": world_env.n_actions,
                "n_states": world_env.n_states,
                "opt_v": world_env.opt_v,
                "feature_matrix": feature_matrix
            }
        else:  # "objectworld"
            feature_matrix = world_env.feature_matrix(discrete=True)
            demo[seed] = {
                "gt_r": ground_r,
                "expert_policy": world_env.policy,
                "trajectories": trajectories,
                "n_actions": world_env.n_actions,
                "n_states": world_env.n_states,
                "opt_v": world_env.opt_v,
                "objects": world_env.objects,
                "feature_matrix": feature_matrix,
                "gt_T": traj_len
            }

        # --------------------------------------
        # Plot ground truth value (row 0).
        ax = axs[0][i]
        im = ax.pcolor(world_env.opt_v.reshape((grid_size, grid_size)))
        plt.colorbar(im, ax=ax)
        ax.set_title("GT V", fontsize='small')

        # Plot ground truth reward (row 1).
        ax1 = axs[1][i]
        im1 = ax1.pcolor(ground_r.reshape((grid_size, grid_size)))
        plt.colorbar(im1, ax=ax1)
        ax1.set_title("GT reward", fontsize='small')

        # Plot the (expert) policy arrows (row 2).
        # Define direction offsets for each action index.
        arrows = {
            0: (1, 0),  1: (1, 1),  2: (0, 1),  3: (-1, 1),
            4: (-1, 0), 5: (-1, -1), 6: (0, -1), 7: (1, -1),
            8: (0, 0)
        }
        scale = 0.25
        policy_reshaped = np.array(world_env.policy).reshape(grid_size, grid_size)

        ax2 = axs[2][i]
        for rr in range(grid_size):
            for cc in range(grid_size):
                action = policy_reshaped[rr, cc]
                ax2.arrow(
                    cc, rr,
                    scale * arrows[action][0],
                    scale * arrows[action][1],
                    head_width=0.1
                )

        # If using objectworld, plot the inner and outer object colors (rows 3 and 4).
        if task == "objectworld":
            inner = np.zeros((grid_size, grid_size))
            outer = np.zeros((grid_size, grid_size))

            # 'world_env.objects' is presumably a dict mapping coordinates -> object attributes.
            for (x, y), obj in world_env.objects.items():
                # Convert (inner_colour, outer_colour) to integer codes.
                inner[x][y] = obj.inner_colour + 1
                outer[x][y] = obj.outer_colour + 1

            ax3 = axs[3][i]
            im3 = ax3.pcolor(inner)
            plt.colorbar(im3, ax=ax3)
            ax3.set_title("Inner color", fontsize='small')

            ax4 = axs[4][i]
            im4 = ax4.pcolor(outer)
            plt.colorbar(im4, ax=ax4)
            ax4.set_title("Outer color", fontsize='small')

    # Save a single figure containing the ground truth plots for all MDPs.
    plt.savefig(f"{log_dir}/reward_model_{reward_model}_expert_plots.jpg")

    # Save the 'demo' dictionary (trajectory data, feature matrices, etc.) to a pickle file.
    with open(f"{log_dir}/reward_model_{reward_model}.pkl", 'wb') as fp:
        pickle.dump(demo, fp)

        
def run_experiments(task_env, reward_model, method, n_horizons, n_mdps, log_base_dir):
    """
    Dispatch experiments for a list of expert fractions, either as a single MDP,
    batch of MDPs, or cross-validation runs.

    Args:
        task_env (str): Name of the environment (e.g., "gridworld", "objectworld").
        reward_model (str): The reward model (e.g., "linear", "nonlinear").
        method (str): Which IRL method to run ("single", "batch", or "cross").
        n_horizons (int): Number of horizon values to try.
        n_mdps (int): Number of MDPs to run for batch/cross-validation.
        log_base_dir (str): Directory path for saving logs/results.

    Returns:
        None
    """
    expert_list = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    if n_mdps is None:
        n_mdps = 10  # Number of MDPs for batch/cross runs

    if method == "expert":
        cache_expert_demo(task_env, n_mdps, reward_model, log_base_dir)
    else:

        for expert_frac in expert_list:
            if method == "single":
                single_mdp(
                    task_env, float(expert_frac),
                    reward_model,
                    n_horizons,
                    log_base_dir
                )
            elif method == "batch":
                batch_mdps(
                    task_env, float(expert_frac),
                    n_mdps, reward_model,
                    n_horizons,
                    log_base_dir
                )
            elif method == "cross":
                cross_validate_mdps(
                    task_env, float(expert_frac),
                    n_mdps, reward_model,
                    n_horizons,
                    log_base_dir
                )
            else:
                print(f"Unknown method: {method}")
                break

def main(argv):
   
    task_env, reward_model, method, n_horizons, n_mdps = maxent_parse(argv)

    # Build output directory based on arguments
    if method == "expert":
        log_base_dir = f"./expert_demo/{task_env}"
    else:
        log_base_dir = f"./output/{task_env}/maxent/{method}/reward_model_{reward_model}"
    make_dir(log_base_dir)

    # Dispatch experiments based on 'method'
    run_experiments(task_env, reward_model, method, n_horizons, n_mdps, log_base_dir)

if __name__ == "__main__":
    main(sys.argv)