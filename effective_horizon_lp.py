import sys
import math
import pickle
import getopt

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
import irl.lp_jax as linear_irl
from irl.mdp.objectworld import Objectworld
from irl.mdp.gridworld import Gridworld

gt_gamma = 0.99

def lp_parse(argv):
    """
    Parse command-line arguments for the LP IRL experiment.

    Args:
        argv (list[str]): Command line arguments, typically sys.argv.

    Returns:
        tuple: (task_env, reward_model, mode, n_gammas, n_mdps)
          - task_env (str): The environment name ("gridworld" or "objectworld", etc.)
          - reward_model (str): The reward model type (e.g., "linear", "nonlinear")
          - mode (str): The IRL experiment method ("single", "batch", "cross")
          - n_gammas (int): The number of gamma values to iterate over
          - n_mdp (int): The number of MDPs to generate (for batch or cross-validation)
    """
    arg_task = ""
    arg_mode = ""
    arg_n_gammas = ""
    arg_n_mdp = ""

    arg_help = f"{argv[0]} -t <task> -m <mode> -n <ngammas> -N <nmdp>"
    try:
        opts, _ = getopt.getopt(
            argv[1:], "h:t:m:n:N:", ["help", "task=", "mode=", "ngammas=", "nmdp="]
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
        elif opt in ("-n", "--ngammas"):
            arg_n_gammas = arg
        elif opt in ("-N", "--nmdp"):
            arg_n_mdp = arg

    # Expecting something like "gridworld_linear" => split into (gridworld, linear)
    try:
        task_env, reward_model = arg_task.split("-")
    except ValueError:
        print("ERROR: Task argument must be in the form 'task-rewardmodel'.")
        sys.exit(2)

    # Convert n_gammas to int
    if not arg_n_gammas.isdigit():
        print("ERROR: Number of gammas must be an integer.")
        sys.exit(2)

    return task_env, reward_model, arg_mode, int(arg_n_gammas), int(arg_n_mdp)

def lp_env_init(task, expert_fraction, reward_model, seed=None, demo=False, cross_validate=False):
    """
    Initialize the specified environment (Gridworld or Objectworld), load or generate
    the expert policy and reward, and optionally split expert demonstrations into training/validation.

    Args:
        task (str): "objectworld" or "gridworld".
        expert_fraction (float): Fraction of states for which expert demonstration data is used.
        reward_model (str): The reward model to be used (if environment-specific).
        seed (int, optional): Random seed.
        demo (bool, optional): Whether to generate demonstration data.
        cross_validate (float, optional): Ratio for splitting data into training/validation sets.

    Returns:
        tuple: If cross_validate is None:
                 (env, ground_r, expert_policy, m_expert)
               else:
                 (env, ground_r, expert_policy, m_expert, training, validation)
        Where:
            env: The MDP environment instance (Gridworld or Objectworld).
            ground_r (np.ndarray): Ground truth reward array.
            expert_policy (list[int]): Expert policy actions for each state.
            m_expert (list[int]): Indices of states for which we have expert demonstrations.
            training (list[int]): Training set states (only if cross_validate is not None).
            validation (list[int]): Validation set states (only if cross_validate is not None).
    """
    if task == "objectworld":
        world_env = Objectworld(seed=seed, demo=demo, reward_model=reward_model)
    elif task == "gridworld":
        world_env = Gridworld(seed=seed, demo=demo, reward_model=reward_model)
    else:
        raise ValueError(f"Unknown task environment: {task}")

    ground_r = world_env.reward_array
    expert_policy = world_env.policy
    expert_idx = int(world_env.n_states * expert_fraction)

    # Generate or split expert demonstrations
    if cross_validate:
        m_expert, training, validation = world_env.generate_expert_demonstrations(
            expert_idx, cross_validate_ratio=0.8
        )
        return world_env, ground_r, expert_policy, m_expert, training, validation
    else:
        m_expert = world_env.generate_expert_demonstrations(expert_idx)
        return world_env, ground_r, expert_policy, m_expert
        
def lp_irl(world_env, m_expert, training_discount, slack_variable=0.001, ground_r=None, inv_v=None, approx=False):
    """
    Solve the linear programming IRL problem for a given environment and expert data.

    Args:
        world_env: The environment (Gridworld or Objectworld).
        m_expert (list[int]): Indices of expert-demonstrated states.
        training_discount (float): Discount factor for training the IRL solver.
        ground_r (np.ndarray, optional): Ground truth reward (used if `approx=False`).
        inv_v (np.ndarray, optional): Inverse of the value function or an approximation (used if `approx=False`).
        approx (bool, optional): If True, uses approximate approach with `large_irl`, 
                                 else uses exact approach with `irl(...)`.

    Returns:
        tuple: (r, V, policy)
            r (np.ndarray): Learned reward of shape (n_states,).
            V (np.ndarray): Value function induced by the learned reward.
            policy (list[int]): The derived policy from the environment's evaluate_learnt_reward.
    """
    if approx:
        # Example approximate method
        S_prime = list(set(range(world_env.n_states)) - set(m_expert))
        transition_expert = world_env.transition_probability[np.arange(world_env.n_states), world_env.policy, :]
        transition_expert[S_prime, :] = 0

        non_expert_policy = np.repeat(
            set(np.arange(world_env.n_actions)), world_env.n_states
        ) - np.array([{i} for i in np.array(world_env.policy)])
        non_expert_policy = np.array([list(i) for i in non_expert_policy])
        transition_non_expert = np.zeros((world_env.n_states, world_env.n_actions - 1, world_env.n_states))

        for i in range(world_env.n_actions - 1):
            transition_non_expert[:, i, :] = world_env.transition_probability[
                np.arange(world_env.n_states), non_expert_policy[:, i], :
            ]
        transition_non_expert[S_prime, :, :] = 0

        feature_matrix = world_env.feature_matrix()  # shape: (n_states, D)
        values = value_iteration.value_matrix(
            world_env.n_states, transition_expert, feature_matrix, training_discount
        )
        r = linear_irl.large_irl(
            values, feature_matrix, world_env.n_states, world_env.n_actions, world_env.policy,
            transition_expert, transition_non_expert, m_expert
        )
    else:
        try:
            results = linear_irl.irl(
                world_env.n_states,
                world_env.n_actions,
                world_env.transition_probability,
                np.array(world_env.policy),
                m_expert,
                training_discount,
                1,
                0,
                beta=slack_variable,
                ground_r=ground_r,
                inv_v=inv_v
            )
            r = np.asarray(results["x"][: world_env.n_states], dtype=np.double)
        except:
            results = linear_irl.irl(
                world_env.n_states,
                world_env.n_actions,
                world_env.transition_probability,
                np.array(world_env.policy),
                m_expert,
                training_discount,
                1,
                0,
                beta=0,
                ground_r=ground_r,
                inv_v=inv_v
            )
            r = np.asarray(results["x"][: world_env.n_states], dtype=np.double)

    r = r.reshape((world_env.n_states,))
    V, policy = world_env.evaluate_learnt_reward(r, training_discount)
    return r, V, policy

def single_mdp(
    task: str,
    expert_fraction: float,
    reward_model: str,
    num_gammas: int,
    log_dir: str
) -> None:
    """
    Run LP-based IRL on a single (randomized) MDP instance (e.g. Gridworld or Objectworld),
    then plot and save the resulting learned reward and value functions for multiple gamma values.

    The function:
      1) Initializes the environment, retrieves the ground truth reward, expert policy, and set of
         "expert" states (determined by 'expert_fraction').
      2) Iterates over a list of gamma values, running LP-based IRL for each gamma.
      3) Compares the learned policies to the expert policy (both for the entire state space and
         only the expert states) and saves plots illustrating reward, value, and policy arrows.

    Args:
        task (str): Identifier of the environment ("gridworld", "objectworld", etc.).
        expert_fraction (float): Fraction of the total states that are considered demonstrated by an expert.
        reward_model (str): Name of the environment-specific reward model.
        num_gammas (int): Number of gamma values to iterate over (the code sets them from small to ~0.99).
        log_dir (str): Path to the directory where logs, plots, and pickle outputs will be saved.

    Returns:
        None. This function saves figures (JPG) and pickled results (P) to 'log_dir' as side effects.
    """

    # 1. Initialize environment, retrieving environment wrapper, ground truth reward, 
    #    expert policy, and set of expert states.
    world_env, ground_r, expert_policy, m_expert = lp_env_init(
        task=task,
        expert_fraction=expert_fraction,
        demo=True,
        seed=42,
        reward_model=reward_model
    )

    # Assume 'world_env' provides the grid dimension. If so, retrieve it for reshaping plots.
    grid_size = world_env.grid_size

    # Create a mask of states that are considered "expert" (for special coloring of arrows).
    expert_mask = np.array(
        [1 if state_idx in m_expert else 0 for state_idx in range(world_env.n_states)]
    ).reshape(grid_size, grid_size)

    # Prepare lists for storing results.
    results = []            # Discrepancy w.r.t. expert policy over all states.
    training_results = []   # Discrepancy w.r.t. expert policy over only expert states.
    learned_r = []          # Learned reward for each gamma.
    learned_v = []          # Learned value function for each gamma.
    eval_v = []             # Value function evaluated under the ground truth reward.

    # 2. Prepare subplots to visualize ground truth and learned results across different gammas.
    fig, axs = plt.subplots(
        nrows=4,
        ncols=7,
        layout="constrained",
        figsize=(30, 8),
        sharex=True,
        sharey=True
    )
    fig.suptitle(f"Performance for each gamma with {expert_fraction * 100:.1f}% expert coverage")

    # Generate a list of gamma values, ensuring 0.99 is included as the last one.
    gamma_list = [(i + 1) / num_gammas for i in range(num_gammas - 1)] + [0.99]

    # 3. Plot the ground truth value (opt_v) and ground truth reward in the first column of subplots.
    #    Assuming 'world_env.opt_v' holds the environment's ground-truth optimal state values.
    ax = axs[0][0]
    im = ax.pcolor(world_env.opt_v.reshape((grid_size, grid_size)))
    plt.colorbar(im, ax=ax)

    ax1 = axs[1][0]
    im1 = ax1.pcolor(ground_r.reshape((grid_size, grid_size)))
    plt.colorbar(im1, ax=ax1)

    # Arrow directions for plotting policies on the subplot grid.
    arrows = {
        0: (1, 0),   1: (1, 1),   2: (0, 1),   3: (-1, 1),
        4: (-1, 0),  5: (-1, -1), 6: (0, -1),  7: (1, -1),
        8: (0, 0)
    }
    scale = 0.25

    # Visualize the expert policy in the third row of the first column,
    # using a different color if the state is an "expert" state.
    ax2 = axs[2][0]
    color_code = {0: "black", 1: "lime"}
    expert_policy_reshaped = np.array(expert_policy).reshape(grid_size, grid_size)

    for rr in range(grid_size):
        for cc in range(grid_size):
            action = expert_policy_reshaped[rr, cc]
            ax2.arrow(
                cc, rr,
                scale * arrows[action][0],
                scale * arrows[action][1],
                head_width=0.1,
                color=color_code[expert_mask[rr, cc]]
            )

    # Evaluate the expert policy under the ground truth reward to plot its state values.
    gt_v, inv_v = value_iteration.value(
        policy=expert_policy,
        n_states=world_env.n_states,
        transition_probability=world_env.transition_probability,
        reward=ground_r,
        discount=gt_gamma
    )
    gt_v = np.array(gt_v)   # Ensure np.array for reshaping.
    inv_v = np.array(inv_v) # If needed for IRL solver.

    ax3 = axs[3][0]
    im3 = ax3.pcolor(gt_v.reshape((grid_size, grid_size)))
    plt.colorbar(im3, ax=ax3)

    # Set subplot titles for the first column.
    ax.set_title("GT V", fontsize='small')
    ax1.set_title("GT reward", fontsize='small')
    ax2.set_title("Expert policy", fontsize='small')
    ax3.set_title("GT V (expert)", fontsize='small')

    # Save an initial figure of the ground truth environment layout.
    plt.savefig(f"{log_dir}/expert_{expert_fraction:.2f}_V_R.jpg")

    # 4. Iterate over possible gamma values, running LP-based IRL.
    policies = []  # Store all learned policies for reference.
    for i, gamma in enumerate(gamma_list):
        # Use LP-based IRL to estimate reward and the corresponding value function.
        r_est, v_est_learned, policy = lp_irl(
            world_env,
            m_expert,
            gamma
        )
        learned_r.append(r_est)
        learned_v.append(v_est_learned)

        # Evaluate the learned policy under the ground truth reward with the ground-truth gamma.
        v_est, _ = value_iteration.value(
            policy=policy,
            n_states=world_env.n_states,
            transition_probability=world_env.transition_probability,
            reward=ground_r,
            discount=gt_gamma
        )
        v_est = np.array(v_est)
        eval_v.append(v_est)
        policies.append(policy)

        # Measure how many states differ from the expert policy overall:
        diff = world_env.n_states - np.sum(np.equal(policy, expert_policy))
        # And how many of the expert states differ specifically:
        expert_diff = np.sum([policy[idx] != expert_policy[idx] for idx in m_expert])

        results.append(diff)
        training_results.append(expert_diff)

        print(f"Gamma={gamma:.3f}, diff={diff} (all states), training diff={expert_diff}")

        # Optionally plot intermediate results every 12th gamma or at the final gamma.
        if (i % 12 == 0) or (i == num_gammas - 1):
            import math
            col_index = min(math.ceil(i / 12) + 1, 6)  # Columns from 1..6 in zero-based index
            # Plot the learned reward in row 2 of the subplot grid
            ax2_local = axs[1][col_index]
            im2 = ax2_local.pcolor(r_est.reshape((grid_size, grid_size)))
            plt.colorbar(im2, ax=ax2_local)

            # Plot the learned value in row 1 of the subplot grid
            ax3_local = axs[0][col_index]
            im3_local = ax3_local.pcolor(v_est_learned.reshape((grid_size, grid_size)))
            plt.colorbar(im3_local, ax=ax3_local)
            ax3_local.set_title(f"Gamma = {gamma:.2f}", fontsize='small')

            # Plot the learned policy in row 3 of the subplot grid
            p_local = np.array(policy).reshape(grid_size, grid_size)
            ax4 = axs[2][col_index]
            discrepancy = np.equal(expert_policy, policy).reshape(grid_size, grid_size)

            # Define color codes for match/mismatch vs. whether a state is in the expert set
            color_code_normal = {True: "black", False: "yellow"}
            color_code_expert = {True: "lime",  False: "red"}

            for rr in range(grid_size):
                for cc in range(grid_size):
                    act = p_local[rr, cc]
                    if expert_mask[rr, cc] == 1:
                        ax4.arrow(
                            cc, rr,
                            scale * arrows[act][0],
                            scale * arrows[act][1],
                            head_width=0.1,
                            color=color_code_expert[discrepancy[rr, cc]]
                        )
                    else:
                        ax4.arrow(
                            cc, rr,
                            scale * arrows[act][0],
                            scale * arrows[act][1],
                            head_width=0.1,
                            color=color_code_normal[discrepancy[rr, cc]]
                        )

            # Plot the evaluated value (under the ground reward) in row 4 of the subplot grid
            ax5 = axs[3][col_index]
            im5 = ax5.pcolor(v_est.reshape((grid_size, grid_size)))
            plt.colorbar(im5, ax=ax5)

    # 5. Save a final figure showing the last updates across columns for all tested gamma values.
    plt.savefig(f"{log_dir}/expert_{expert_fraction:.2f}_V_R.jpg")

    # Also pickle all relevant data for later analysis.
    with open(f"{log_dir}/expert_{expert_fraction:.2f}.p", 'wb') as fp:
        pickle.dump(
            {
                "gamma": gamma_list,
                "error": results,
                "training_error": training_results,
                "learned_r": learned_r,
                "learned_v": learned_v,
                "gt_r": ground_r,
                "gt_v": gt_v,
                "expert_policy": expert_policy,
                "policies": policies,
                "m_expert": m_expert,
                "eval_v": eval_v
            },
            fp
        )

    # 6. Generate and save an error curve plot over the tested gamma values.
    output_dir = f"{log_dir}/expert_{expert_fraction:.2f}_error_curve.jpg"
    plot_error_curve(
        output_dir,
        gamma_list=gamma_list,
        error=results,
        training_error=training_results
    )

def batch_mdps(
    task: str,
    expert_fraction: float,
    n_mdp: int,
    reward_model: str,
    num_gammas: int,
    log_dir: str
) -> None:
    """
    Run LP-based IRL on multiple (n_mdp) randomly initialized MDPs in a batch.

    For each MDP:
      1) Initialize the environment and obtain the ground truth reward, expert policy,
         and set of expert states (determined by 'expert_fraction').
      2) Iterate over 'num_gammas' different gamma values, running LP IRL each time.
      3) Evaluate the learned policy, record discrepancies (entire state space vs. 
         expert states), store intermediate data, and save results to disk.

    Args:
        task (str): Environment identifier (e.g., "gridworld", "objectworld").
        expert_fraction (float): Fraction of states assumed to have expert demonstrations.
        n_mdp (int): Number of distinct MDPs (random seeds) to process in this batch.
        reward_model (str): Specifies the environment's reward model (e.g., "hard" or another variant).
        num_gammas (int): Number of gamma values to try (range is set from 1/num_gammas to 0.99).
        log_dir (str): Directory path for saving pickled results files.

    Returns:
        None. This function saves intermediate and final results to pickle files in 'log_dir'
        as side-effects. No explicit return value.
    """

    # Generate a list of gamma values, ensuring 0.99 is the last entry.
    gamma_list = [(i + 1) / num_gammas for i in range(num_gammas - 1)] + [0.99]

    # Allocate arrays to store discrepancies across all MDPs and all gamma values.
    #  - result[i][j]: mismatch with the expert policy over the *entire* state space
    #  - training_result[i][j]: mismatch restricted to the states in the expert set
    result = np.zeros((n_mdp, num_gammas))
    training_result = np.zeros((n_mdp, num_gammas))

    # Prepare lists to accumulate learned rewards, learned values, and evaluated values across MDPs.
    learned_r = []
    learned_v = []
    evaluated_v = []

    # Set the random seed for selecting MDP seeds. Adjust as necessary for your workflow.
    rn.seed(42)

    if reward_model == "hard":
        grid_size = 15
    else:
        grid_size = 10

    
    seeds = rn.choice(range(grid_size ** 2), n_mdp, replace=False)

    # Loop over the MDPs.
    for i in range(n_mdp):
        # 1) Initialize environment with a unique seed.
        world_env, ground_r, expert_policy, m_expert = lp_env_init(
            task=task,
            expert_fraction=expert_fraction,
            seed=seeds[i],
            reward_model=reward_model
        )

        # Compute ground-truth value for the expert policy (for reference).
        gt_v, _ = value_iteration.value(
            policy=expert_policy,
            n_states=world_env.n_states,
            transition_probability=world_env.transition_probability,
            reward=ground_r,
            discount=gt_gamma
        )
        gt_v = np.array(gt_v)

        # Temporary lists to hold results across the gamma sweep for this MDP.
        r_list = []
        v_list = []
        e_v_list = []

        # 2) Iterate over each gamma value, run LP IRL, and measure discrepancies.
        for j, gamma in enumerate(gamma_list):
            r_est, v_est, policy = lp_irl(
                world_env,
                m_expert,
                gamma
            )

            # Compute policy discrepancy for the entire state space.
            diff = world_env.n_states - np.sum(np.equal(policy, expert_policy))
            # Compute policy discrepancy restricted to the expert states.
            expert_diff = np.sum([policy[idx] != expert_policy[idx] for idx in m_expert])

            # Store the learned reward and value.
            r_list.append(r_est)
            v_list.append(v_est)

            # Record the error metrics for this (i, j) = (MDP_i, gamma_j).
            result[i][j] = diff
            training_result[i][j] = expert_diff

            # Evaluate the learned reward using the expert policy or learned policy, as desired.
            # Below, we evaluate the expert policy using the learned reward (adjust if needed).
            e_v, _ = value_iteration.value(
                policy=expert_policy,
                n_states=world_env.n_states,
                transition_probability=world_env.transition_probability,
                reward=r_est,
                discount=gt_gamma
            )
            e_v = np.array(e_v)
            e_v_list.append(e_v)

            print(f"MDP {i}, gamma {gamma:.3f}, error (all states)={diff}, error (expert)={expert_diff}")

        # Append the ground truth reward and value at the end of the lists for reference.
        r_list.append(ground_r)
        v_list.append(gt_v)
        e_v_list.append(gt_v)

        learned_r.append(r_list)
        learned_v.append(v_list)
        evaluated_v.append(e_v_list)

        # 3) Save intermediate results for each MDP to a pickle. 
        #    This overwrites the same file at each iteration; adjust naming if desired.
        with open(f"{log_dir}/{n_mdp}_expert_{expert_fraction}.p", "wb") as fp:
            pickle.dump(
                {
                    "gamma": gamma_list,
                    "error": result,
                    "training error": training_result,
                    "learned_r": learned_r,
                    "learned_v": learned_v,
                    "evaluated_v": evaluated_v,
                    "n_mdp": i + 1
                },
                fp
            )

    # 4) Plot a batch error curve over all tested gamma values, summarizing different error metrics.
    output_file = f"{log_dir}/{n_mdp}_MDPs_expert_{expert_fraction}_error_curve.jpg"
    plot_batch_error_curve(
        output_file,
        gamma_list=gamma_list,
        error=result
    )


def cross_validate_mdps(
    task: str,
    expert_fraction: float,
    n_mdp: int,
    reward_model: str,
    num_gammas: int,
    log_dir: str
) -> None:
    """
    Perform cross-validation to select the discount factor (gamma) across multiple MDPs.

    For each of the MDPs:
      1) Initialize the environment using lp_env_init, which returns:
         - The environment (world_env)
         - Ground truth reward (ground_r)
         - Expert policy (expert_policy)
         - A set of "expert" states (m_expert)
         - Training (training) and validation (validation) subsets of expert states.
      2) Loop over 'num_gammas' discount factors, using LP-based IRL (lp_irl) to learn a
         reward and policy from the training subset.
      3) Evaluate each learned policy by measuring policy discrepancy:
         - gt_error: Discrepancy over the entire state space
         - expert_error: Discrepancy over all expert states
         - validate_error: Discrepancy on the hold-out validation subset
      4) Save the intermediate results (learned reward, policy, etc.) to disk.
      5) Plot a cross-validation curve summarizing the errors across gamma values.

    Args:
        task (str): Name of the environment (e.g., "gridworld", "objectworld").
        expert_fraction (float): Fraction of total states that have expert demonstrations.
        n_mdp (int): Number of distinct MDPs (with different random seeds) to process.
        reward_model (str): Specifies the environment's reward function model.
        num_gammas (int): Number of gamma values to test (from 1/num_gammas up to 0.99).
        log_dir (str): Directory in which results and plots will be saved.

    Returns:
        None. The function saves results and cross-validation plots to 'log_dir'.
    """

    # Generate the list of gamma values, ensuring 0.99 is included last.
    gamma_list = [(i + 1) / num_gammas for i in range(num_gammas - 1)] + [0.99]

    # Arrays to store errors across MDPs and gamma values:
    #   gt_error[i][j] : The mismatch over the entire state space for MDP i, gamma j
    #   expert_error[i][j] : Mismatch restricted to the full expert set for MDP i, gamma j
    #   validate_error[i][j] : Mismatch over only the validation subset for MDP i, gamma j
    gt_error = np.zeros((n_mdp, num_gammas))
    expert_error = np.zeros((n_mdp, num_gammas))
    validate_error = np.zeros((n_mdp, num_gammas))

    # Lists to accumulate learned objects for all MDPs:
    #   learned_r   : List of learned reward functions for each (MDP, gamma)
    #   learned_v   : List of value functions for each (MDP, gamma)
    #   evaluated_v : Any additional evaluation metrics over learned policies
    learned_r = []
    learned_v = []
    evaluated_v = []

    # Fix the random seed for MDP selection. Adjust seeding as needed for your workflow.
    rn.seed(42)

    if reward_model == "hard":
        grid_size = 15
    else:
        grid_size = 10

    seeds = rn.choice(range(grid_size ** 2), n_mdp, replace=False)
    

    # Loop over each MDP (identified by a distinct random seed).
    for i in range(n_mdp):
        # Initialize the environment for cross-validation, obtaining training/validation splits.
        (world_env, ground_r, expert_policy, m_expert,
         training, validation) = lp_env_init(
             task=task,
             expert_fraction=expert_fraction,
             seed=seeds[i],
             cross_validate=True,
             reward_model=reward_model
         )

        # Compute the ground-truth value for the expert policy (for reference).
        gt_v, _ = value_iteration.value(
            policy=expert_policy,
            n_states=world_env.n_states,
            transition_probability=world_env.transition_probability,
            reward=ground_r,
            discount=gt_gamma
        )
        gt_v = np.array(gt_v)

        # Temporary lists to record the learned reward, value, and evaluation for each gamma.
        r_list = []
        v_list = []
        e_v_list = []

        # Iterate through all gamma values, running LP-based IRL.
        for j, gamma in enumerate(gamma_list):
            # Learn reward and policy using only the training subset.
            r_est, v_est, policy = lp_irl(world_env, training, gamma)

            # Compute policy discrepancies.
            expert_diff = np.sum(policy[idx] != expert_policy[idx] for idx in m_expert)
            val_diff = np.sum(policy[idx] != expert_policy[idx] for idx in validation)
            gt_diff = world_env.n_states - np.sum(np.equal(policy, expert_policy))

            # Store errors in the corresponding arrays.
            gt_error[i][j] = gt_diff
            expert_error[i][j] = expert_diff
            validate_error[i][j] = val_diff

            # Append learned reward and value estimates.
            r_list.append(r_est)
            v_list.append(v_est)

            # Evaluate the expert policy using the learned reward (or any evaluation you want).
            e_v, _ = value_iteration.value(
                policy=expert_policy,
                n_states=world_env.n_states,
                transition_probability=world_env.transition_probability,
                reward=r_est,
                discount=gt_gamma
            )
            e_v = np.array(e_v)
            e_v_list.append(e_v)

            print(
                f"MDP={i}, gamma={gamma:.3f}, "
                f"val_err={val_diff}, expert_err={expert_diff}, gt_err={gt_diff}"
            )

        # Append ground truth references to the lists for consistency.
        r_list.append(ground_r)
        v_list.append(gt_v)
        e_v_list.append(gt_v)

        # Collect all learned objects across gamma values for this MDP.
        learned_r.append(r_list)
        learned_v.append(v_list)
        evaluated_v.append(e_v_list)

        # Save partial results for each MDP in a pickle file. Adjust the filename pattern if needed.
        with open(f"{log_dir}/{n_mdp}_expert_{expert_fraction}.p", "wb") as fp:
            pickle.dump(
                {
                    "gamma": gamma_list,
                    "gt_error": gt_error,
                    "val_error": validate_error,
                    "expert_error": expert_error,
                    "learned_r": learned_r,
                    "learned_v": learned_v,
                    "evaluated_v": evaluated_v
                },
                fp
            )

    # Plot a cross-validation curve over all tested gamma values, summarizing different error metrics.
    output_file = f"{log_dir}/{n_mdp}_MDPs_expert_{expert_fraction}_error_curve.jpg"
    plot_cross_validation_curve(
        output_file,
        gamma_list=gamma_list,
        gt_error=gt_error,
        val_error=validate_error,
        expert_error=expert_error
    )

def run_experiments(task_env, reward_model, method, n_gammas, n_mdps, log_base_dir):
    """
    Dispatch experiments for a list of expert fractions, either as a single MDP,
    batch of MDPs, or cross-validation runs.

    Args:
        task_env (str): Name of the environment (e.g., "gridworld", "objectworld").
        reward_model (str): The reward model (e.g., "linear", "nonlinear").
        method (str): Which IRL method to run ("single", "batch", or "cross").
        n_gammas (int): Number of gamma values to try.
        n_mdps (int): Number of MDPs for batch/cross runs.
        log_base_dir (str): Directory path for saving logs/results.

    Returns:
        None
    """
    expert_list = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    if n_mdps is None:
        n_mdps = 10  # Number of MDPs for batch/cross runs

    for expert_frac in expert_list:
        if method == "single":
            single_mdp(
                task_env, float(expert_frac),
                reward_model, n_gammas,
                log_base_dir
            )
        elif method == "batch":
            batch_mdps(
                task_env, float(expert_frac),
                n_mdps, reward_model, n_gammas,
                log_base_dir
            )
        elif method == "cross":
            cross_validate_mdps(
                task_env, float(expert_frac),
                n_mdps, reward_model, n_gammas,
                log_base_dir
            )
        else:
            print(f"Unknown method: {method}")
            break

def main(argv):
    """
    Theoretically, F in lp has additional information of all expert demonstrations in the states,
    therefore, having a larger gamma better approximates the ground-truth mapping from R to V,
    which accounts for the long-term effect of taking an action. In other words, with lp, using
    a small gamma folds much of the V into R (the R becomes 'shaped'). Large gamma => good performance,
    small gamma => shaped R.

    In contrast, lp1 uses partial expert information to approximate F, assigning a zero vector to
    undemonstrated states. If we look at the constraints formed by the Bellman equation:
        V^{π} = R + gamma T^{π} V^{π},
    1) V(s) = R(s) + gamma T^{π}[s] V^{π} = R(s) for s not demonstrated => the R is shaped, 
       equivalent to V(s).
    2) V(s) = R(s) + gamma T^{π}[s] V^{π} for s in the demonstrations => unaffected.

    Downside: now F-hat is estimated from partial expert demonstrations, so V^{π} ~ (I - gamma F-hat)^{-1} R,
    and the constraints are not strictly forced.

    To demonstrate a U-shaped curve, we need:
      1) A complex model, and
      2) Few data. Complexity of the model scales with the horizon.

    For simple tasks (e.g., gridworld), the decision boundary is clear, so having few data can
    still extrapolate well. We may need to make the decision boundary more complex.

    This main function:
      1) Parses command-line arguments (task_env, reward_model, method, n_gammas).
      2) Creates output directories.
      3) Runs the chosen IRL method (single, batch, or cross) on multiple expert fractions.
    """
    task_env, reward_model, method, n_gammas, n_mdps = lp_parse(argv)

    # Build output directory based on arguments
    log_base_dir = f"./output/{task_env}/lp/{method}/reward_model_{reward_model}"
    make_dir(log_base_dir)

    # Dispatch experiments based on 'method'
    run_experiments(task_env, reward_model, method, n_gammas, n_mdps, log_base_dir)

if __name__ == "__main__":
    main(sys.argv)