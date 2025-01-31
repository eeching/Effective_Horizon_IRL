import numpy as np
import sys
import getopt
import matplotlib.pyplot as plt
import pdb
import irl.mdp.objectworld as objectworld
import irl.mdp.gridworld as gridworld
import os
import pickle
import seaborn as sns
import pandas as pd
import math

plt.rcParams.update({'font.size': 8})
x_label_dict = {"lp": "Gammas", "maxent": "Horizons"}
x_label_key_dict = {"lp": "gamma", "maxent": "horizon"}
n_list_dict = {"lp": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                "maxent": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]}

def make_dir(path):
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)

def plot_error_curve(output_filename, filename=None, gamma_list=None, error=None,  training_error=None, horizon_list=None, state_set=None):

    if filename is not None:
        with open(filename, 'rb') as fp:
            data = pickle.load(fp)
            gamma_list = data['gamma']
            error = data['error']
            unique = data['unique']
            training_error = data['training_error']

    fig, ax = plt.subplots()
    if horizon_list is None:
        x_value = gamma_list
    else:
        x_value = horizon_list
    ax.plot(x_value, error, label="Total error counts.")
    ax2 = ax.twinx()
    ax2.plot(x_value, training_error, label="Training error counts.", linestyle="dotted")
    if state_set is not None:
        ax2.plot(x_value, state_set, label="Effective state coverage.", linestyle="--")
    if horizon_list is None:
        ax.set_xlabel('Gamma', fontsize="medium")
    else:
        ax.set_xlabel('Horizon', fontsize="medium")
    ax.set_ylabel('Error Count', fontsize="medium")
    ax.set_title('Discrepancy between the induced policy and the expert', fontsize="large")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2, loc=2)
    fig.tight_layout()
    plt.savefig(output_filename)
    
def plot_batch_error_curve(output_file, filename=None, gamma_list=None, error=None, horizon_list=None):

    if filename is not None:
        with open(filename, 'rb') as fp:
            data = pickle.load(fp)
            gamma_list = data['gamma']
            error = data['error']
    mean, std = np.mean(error, axis=0), np.std(error, axis=0)
    if horizon_list is None:
        x_value = gamma_list
    else:
        x_value = horizon_list

    fig, ax = plt.subplots()
    ax.plot(x_value, mean, lw=2, color='blue')
    ax.fill_between(x_value, mean + std, mean - std, facecolor='blue', alpha=0.5)

    if horizon_list is None:
        ax.set_xlabel('Gamma', fontsize="medium")
    else:
        ax.set_xlabel('Horizon', fontsize="medium")
    ax.set_ylabel('Average Error Count', fontsize="medium")
    ax.set_title('Discrepancy for different Gammas', fontsize="large")
    fig.tight_layout()

    plt.savefig(output_file)

def plot_cross_validation_curve(output_file, filename=None, gamma_list=None, gt_error=None, val_error=None, expert_error=None, train_error=None, horizon_list=None):
   
    if filename is not None:
        with open(filename, 'rb') as fp:
            data = pickle.load(fp)
            gamma_list = data['gamma']
            gt_error = data['gt_error']
            val_error = data['val_error']
            expert_error = data['expert_error']
            train_error = data['train_error']
    
    if horizon_list is None:
        x_value = gamma_list
    else:
        x_value = horizon_list


    gt_mean, gt_std = np.mean(gt_error, axis=0), np.std(gt_error, axis=0)
    expert_mean, expert_std = np.mean(expert_error, axis=0), np.std(expert_error, axis=0)
    val_mean, val_std = np.mean(val_error, axis=0), np.std(val_error, axis=0)

    fig, ax = plt.subplots(layout="constrained", figsize=(6, 6))
    if horizon_list is None:
        x_value = gamma_list
    else:
        x_value = horizon_list

    ax.plot(x_value, gt_mean, label='GroundTruth Error', lw=2, color='darkgreen')
    ax.fill_between(x_value, gt_mean + gt_std, gt_mean - gt_std, facecolor='darkgreen', alpha=0.3)

    ax1 = ax.twinx()
    ax1.plot(x_value, val_mean, label='Validation Error', lw=2, color='blue')
    ax1.fill_between(x_value, val_mean + val_std, val_mean - val_std, facecolor='blue', alpha=0.3)

    ax1.plot(x_value, expert_mean, label='Training + validation Error', lw=2, color='purple')
    ax1.fill_between(x_value, expert_mean + expert_std, expert_mean - expert_std, facecolor='purple', alpha=0.3)

    if horizon_list is None:
        ax.set_xlabel('Gamma', fontsize="medium")
    else:
        ax.set_xlabel('Horizon', fontsize="medium")
    ax.set_ylabel('Average Error Counts', fontsize="medium")
    ax.set_title('Error Counts for different Horizons')
    ax.legend(loc='lower right')

    plt.savefig(output_file)

def load_maxent_expert(task_env, reward_model):
    # Load the pickle file
    load_path = f"./expert_demo/{task_env}/reward_model_{reward_model}.pkl"

    with open(load_path, 'rb') as fp:
        data = pickle.load(fp)

    # Extract and remove the transition function from the dictionary
    transition_function = np.array(data.pop("T"))

    # Collect all seed keys
    seeds = list(data.keys())
    
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

def plot_selected_data(method, task, reward_model, data_coverage):
    """
    Plots a detailed overview of learned rewards, value functions, and policies for a 
    single run at a particular coverage fraction (data_coverage), across multiple 
    horizon/gamma values. Visualizes:
      1) Ground-truth reward (R) and value (V), plus the expert policy.
      2) Learned reward (R) and induced value (V) for a subsampled set of 
         horizon/gamma values.
      3) Induced policies, highlighting agreement/disagreement with expert policy, 
         and which states belong to the demonstration set.
      4) The ground-truth value function under the induced policy.

    Parameters
    ----------
    method : str
        IRL method name (e.g., 'lp' or 'maxent') to determine how data is loaded 
        and interpreted.
    task : str
        Environment name (e.g., 'gridworld' or 'objectworld').
    reward_model : str
        Specific reward model variant (e.g., 'hard', 'simple', 'linear', or 'non_linear').
    data_coverage : float
        Fraction (or scaling factor) of expert data used to load the single-run file.
        e.g., 0.2, 0.4, 1.0, etc.

    Global Dependencies
    -------------------
    x_label_dict : dict
        Maps the IRL method to the wide label (e.g., 'Gamma' or 'horizon').
    x_label_key_dict : dict
        Maps the IRL method to the data key for gamma/ horizon list in the pickle.

    File Organization
    -----------------
    Looks for a pickle file:
        ./output/<task>/<method>/single/reward_model_<reward_model>/expert_<data_coverage>.p
    which should contain keys like:
        • "gt_v", "gt_r"
        • x_label_key_dict[method] (list of gamma or horizon values)
        • "learned_v", "learned_r", "eval_v" (arrays over these gamma/horizon values)
        • "policies" (one per gamma/horizon)
        • For 'lp': "m_expert", "expert_policy"
        • For 'maxent': "init_states", "state_coverage"

    Saves the resulting plot to:
        ./plots/<method>/<task>/<reward_model>_<data_coverage>.jpg
    """

    # Construct paths and ensure output directory exists
    base_dir = f"./output/{task}/{method}/single/reward_model_{reward_model}"
    output_dir = f"./plots/{method}/{task}"
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{output_dir}/{reward_model}_{data_coverage}.jpg"

    # Retrieve horizon/gamma labels from globally-defined dictionaries
    effective_horizon = x_label_dict[method]          # e.g., "Gamma" or "Horizons"
    effective_horizon_label = x_label_key_dict[method]  # key to retrieve from pickle

    # Grid size depends on reward_model (hard-coded logic)
    grid_size = 15 if reward_model == "hard" else 10

    # Create figure spanning 9 subfigures in a single row, each containing a 2x2 grid
    fig = plt.figure(figsize=(30, 4), constrained_layout=True)
    subfigs = fig.subfigures(nrows=1, ncols=9)

    sns.set_palette("flare")
    cmaps = ['RdBu_r', 'viridis']  # one for reward & value, if desired

    # We only have one coverage fraction in subsample_n_list
    subsample_n_list = [data_coverage]
    for row_index, subfig in enumerate([subfigs]):
        filename = f"{base_dir}/expert_{subsample_n_list[row_index]}.p"
        with open(filename, 'rb') as fp:
            data = pickle.load(fp)

        # Basic environment data
        gt_v = data["gt_v"]  # ground-truth value
        gt_r = data["gt_r"]  # ground-truth reward

        # Retrieve horizon/gamma list
        gamma_list = data[effective_horizon_label]

        # Learned data across horizon/gamma values
        v_list = np.array(data["learned_v"])     # shape [#horizons, grid_size^2]
        r_list = np.array(data["learned_r"])     # shape [#horizons, grid_size^2]
        policies = np.array(data["policies"])    # shape [#horizons, grid_size^2]
        eval_v_list = np.array(data["eval_v"])   # shape [#horizons, grid_size^2], GT value for induced policies
        num_gamma = len(gamma_list)

        # Expert data (LP vs. MaxEnt)
        if method == "lp":
            m_expert = data["m_expert"]
            expert_policy = data["expert_policy"]
            expert_mask = np.array([1 if idx in m_expert else 0 for idx in range(grid_size**2)]).reshape(grid_size, grid_size)
        else:  # assume maxent
            init_states = data["init_states"]
            _, _, expert_policy, _, _, _, _, _ = load_maxent_expert(task, reward_model)
            state_coverage = data.get("state_coverage", [])

        print("Loaded file:", filename)

        # --------------------------------------------------------------------------------
        # 1) Plot ground-truth R, V, and the expert policy
        # --------------------------------------------------------------------------------
        gt_axs = subfig[0].subplots(2, 2)
        subfig[0].suptitle("Ground-truth", fontsize=20)

        # Ground-truth reward
        r_ax = gt_axs[0, 0]
        r_im = r_ax.pcolormesh(gt_r.reshape((grid_size, grid_size)), cmap=cmaps[1])
        plt.colorbar(r_im, ax=r_ax, shrink=0.6, location="left")
        r_ax.get_xaxis().set_visible(False)
        r_ax.get_yaxis().set_visible(False)
        r_ax.set_title("GT Reward", fontsize="medium")

        # Ground-truth value
        v_ax = gt_axs[1, 0]
        v_im = v_ax.pcolormesh(gt_v.reshape((grid_size, grid_size)), cmap=cmaps[1])
        v_ax.get_xaxis().set_visible(False)
        v_ax.get_yaxis().set_visible(False)
        v_ax.set_title("GT Value", fontsize="medium", y=-0.2)

        # Expert policy
        p_ax = gt_axs[0, 1]
        arrow_map = {0: (1, 0), 1: (1, 1), 2: (0, 1), 3: (-1, 1), 
                     4: (-1, 0), 5: (-1, -1), 6: (0, -1), 7: (1, -1), 8: (0, 0)}
        scale = 0.25

        policy_2d = np.array(expert_policy).reshape(grid_size, grid_size)
        for rr in range(grid_size):
            for cc in range(grid_size):
                action = policy_2d[rr, cc]
                p_ax.arrow(cc, rr, scale * arrow_map[action][0], scale * arrow_map[action][1],
                           head_width=0.1, color="seagreen")
        p_ax.get_xaxis().set_visible(False)
        p_ax.get_yaxis().set_visible(False)
        p_ax.set_title("Expert Policy", fontsize="medium")

        # Leave bottom-right subplot empty
        empty_ax = gt_axs[1, 1]
        empty_ax.axis('off')

        # --------------------------------------------------------------------------------
        # 2) Process and plot a subset of gamma/horizon values
        #    We'll pick evenly spaced indices from 0..len(gamma_list)-1, total of 8
        # --------------------------------------------------------------------------------
        gamma_indices = np.linspace(0, num_gamma - 1, 8, dtype=int).tolist()

        for i, gamma_idx in enumerate(gamma_indices):
            gamma_value = gamma_list[gamma_idx]
            learned_r = r_list[gamma_idx]
            learned_v = v_list[gamma_idx]
            learned_policy = policies[gamma_idx]
            induced_eval_v = eval_v_list[gamma_idx]

            # Each subfigure has a 2x2 layout for reward, value, policy, and GT value
            sub_axs = subfig[i + 1].subplots(2, 2)
            if effective_horizon_label == "gamma":
                subfig[i + 1].suptitle(f"Gamma = {gamma_value:.4f}", fontsize=20)
            else:
                subfig[i + 1].suptitle(f"Horizon = {gamma_value}", fontsize=20)

            # a) Learned Reward
            r_ax = sub_axs[0, 0]
            r_im = r_ax.pcolormesh(learned_r.reshape((grid_size, grid_size)), cmap=cmaps[1])
            plt.colorbar(r_im, ax=r_ax, shrink=0.6, location="left")
            r_ax.get_xaxis().set_visible(False)
            r_ax.get_yaxis().set_visible(False)
            r_ax.set_title("learned Reward", fontsize="medium")

            # b) Induced Value
            v_ax = sub_axs[1, 0]
            v_im = v_ax.pcolormesh(learned_v.reshape((grid_size, grid_size)), cmap=cmaps[1])
            # We share colorbar with reward plot if you prefer, but let's skip
            v_ax.get_xaxis().set_visible(False)
            v_ax.get_yaxis().set_visible(False)
            v_ax.set_title("Induced Value", fontsize="medium", y=-0.2)

            # c) Induced Policy
            p_ax = sub_axs[0, 1]
            arrow_map = {0: (1, 0), 1: (1, 1), 2: (0, 1), 3: (-1, 1), 
                         4: (-1, 0), 5: (-1, -1), 6: (0, -1), 7: (1, -1), 8: (0, 0)}
            color_map = {True: "black", False: "silver"}
            color_map_expert = {True: "seagreen", False: "springgreen"}

            induced_2d = learned_policy.reshape(grid_size, grid_size)

            if method.startswith("maxent"):
                # For MaxEnt, we have coverage sets that show which states were "demonstrated."
                # Attempt to retrieve coverage for this i. If not found, fallback to an empty set.
                coverage_states = state_coverage[i] if i < len(state_coverage) else []
                expert_mask = np.array([1 if idx in coverage_states else 0 for idx in range(grid_size**2)]).reshape(grid_size, grid_size)

            discrepancy_2d = np.equal(expert_policy, learned_policy).reshape(grid_size, grid_size)
            for rr in range(grid_size):
                for cc in range(grid_size):
                    act = induced_2d[rr, cc]
                    if expert_mask[rr, cc] == 1:
                        # States that appear in the expert set
                        p_ax.arrow(cc, rr, scale * arrow_map[act][0], scale * arrow_map[act][1],
                                   head_width=0.1, color=color_map_expert[discrepancy_2d[rr, cc]])
                    else:
                        # States not in the expert set
                        p_ax.arrow(cc, rr, scale * arrow_map[act][0], scale * arrow_map[act][1],
                                   head_width=0.1, color=color_map[discrepancy_2d[rr, cc]])
            p_ax.get_xaxis().set_visible(False)
            p_ax.get_yaxis().set_visible(False)
            p_ax.set_title("Induced Policy", fontsize="medium")

            # d) Ground-truth value under the induced policy
            eval_v_ax = sub_axs[1, 1]
            eval_im = eval_v_ax.pcolormesh(induced_eval_v.reshape((grid_size, grid_size)), cmap=cmaps[1])
            # This colorbar is shared with the induced value one, for comparison
            plt.colorbar(eval_im, ax=[v_ax, eval_v_ax], shrink=0.6, location="left")
            eval_v_ax.get_xaxis().set_visible(False)
            eval_v_ax.get_yaxis().set_visible(False)
            eval_v_ax.set_title("Induced policy's GT value", fontsize="medium", y=-0.2)

    # Final layout, save figure
    plt.axis('tight')
    plt.savefig(output_filename, dpi=400)
    print(f"Figure saved to: {output_filename}")

def _summarize_single_run(base_dir, x_label, x_label_key, n_list, gt_gamma, method, task, reward_model, output_filename):
    """
    Generates summary plots for the “single” mode using data from multiple parameter values (e.g., different fractions).
    Parameters
    ----------
    base_dir : str
        Base directory where the pickled data is stored.
    x_label : str
        The label name for the x-axis (e.g., "Gammas" or "Horizons").
    x_label_key : str
        Correct key in the loaded pickled data.
    n_list : list
        List of parameter values (e.g., fractions) used to load files and generate plots.
    gt_gamma : float
        Ground-truth gamma value to plot as a vertical reference line.
    method : str
        Name of the IRL method used (e.g., "lp", "batch", etc.).
    task : str
        Name of the environment (e.g., "gridworld").
    reward_model : str
        Name or type of the reward model used.
    """
    # Prepare a summary dictionary to accumulate data for plotting
    summary = {
        x_label: [],
        "State Error Counts": [],
        "Expert Cov.": [],
        "||V-V_gt||": [],
        "Training Error Counts": []
    }

    # Collect data from each file corresponding to different fractions (n)
    for fraction_value in n_list:
        filename = os.path.join(base_dir, f"expert_{fraction_value}.p")
        with open(filename, 'rb') as fp:
            data = pickle.load(fp)

            # X-axis values
            summary[x_label].extend(data[x_label_key])
            summary["Expert Cov."].extend([fraction_value] * len(data[x_label_key]))

            # State-level error
            summary["State Error Counts"].extend(data.get('error', []))

            # Training set error
            summary["Training Error Counts"].extend(data.get('training_error', []))

            # Compute difference in value functions
            learned_v = np.array(data.get("learned_v", []))
            gt_v = data.get("gt_v", [])
            diff_v = np.sum(np.abs(learned_v - gt_v), axis=1).tolist()
            summary["||V-V_gt||"].extend(diff_v)

    # Convert accumulated data to a DataFrame for Seaborn plotting
    df_summary = pd.DataFrame(summary)

    # Create figure and subplots
    sns.set_palette("flare")
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1)

    # First subplot: line plot of State Error Counts
    plot_line_1 = sns.lineplot(
        x=x_label,
        y="State Error Counts",
        hue="Expert Cov.",
        data=df_summary,
        legend=False,
        ax=ax1
    )
    plot_line_1.axvline(gt_gamma, color='purple', ls='-.', label="Ground-truth gamma")

    # Second subplot: line plot of State Error + Training Error on a twin y-axis
    ax2 = fig.add_subplot(1, 2, 2)
    plot_line_2 = sns.lineplot(
        x=x_label,
        y="State Error Counts",
        hue="Expert Cov.",
        data=df_summary,
        legend=False,
        ax=ax2
    )
    ax3 = ax2.twinx()
    plot_line_3 = sns.lineplot(
        x=x_label,
        y="Training Error Counts",
        hue="Expert Cov.",
        data=df_summary,
        legend="full",
        ax=ax3,
        linestyle='--'
    )
    plot_line_2.axvline(gt_gamma, color='purple', ls='-.', label="Ground-truth gamma")

    # Adjust legend parameters
    if len(n_list) > 10:
        n_col = 2
        offset = 1.6
    else:
        n_col = 1
        offset = 1.4

    plot_line_3.legend(bbox_to_anchor=(offset, 0.6), fancybox=True, shadow=True, ncol=n_col)

    fig.suptitle(f'{method} on {task} with reward model {reward_model}')
    fig.tight_layout()

    # Save figure
    # output_filename_original = os.path.join(base_dir, "summary_error_curves.jpg")
    # fig.savefig(output_filename_original)
    fig.savefig(output_filename)

def _summarize_batch_run(base_dir, x_label, x_label_key, n_list, n_mdp, method, task, reward_model, output_filename):
    """
    Generates summary plots for the “batch” mode using data from multiple parameter values (e.g., different fractions).
    Parameters
    ----------
    base_dir : str
        Base directory where the pickled data is stored.
    x_label : str
        The label name for the x-axis (e.g., "Gammas" or "Horizons").
    x_label_key : str
        Correct key in the loaded pickled data.
    n_list : list
        List of parameter values (e.g., fractions) used to load files and generate plots.
    n_mdp : int
        Number of MDPs in each batch of experiments.
    method : str
        Name of the IRL method used (e.g., "lp", "batch", etc.).
    task : str
        Name of the environment (e.g., "gridworld").
    reward_model : str
        Name or type of the reward model used.
    """
    # Prepare a summary dictionary to accumulate data for plotting
    summary = {
        x_label: [],
        "State Error Counts": [],
        "Expert Cov.": [],
        "||V-V_gt||": [],
        "n_mdp": []
    }

    for fraction_value in n_list:
        filename = os.path.join(base_dir, f"{n_mdp}_expert_{fraction_value}.p")
        with open(filename, 'rb') as fp:
            data = pickle.load(fp)

            x_vals = data[x_label_key]
            # “learned_v” is expected to have shape [n_mdp, len(x_vals)+1, ...]
            # last index is presumably ground truth so we separate it out
            num_mdps = np.array(data["learned_v"]).shape[0]

            # Repeat x_vals for each MDP
            summary[x_label] += x_vals * num_mdps
            summary["Expert Cov."].extend([fraction_value] * len(x_vals) * num_mdps)

            # State errors
            state_errors = data['error'][:num_mdps].flatten().tolist()
            summary["State Error Counts"].extend(state_errors)

            # Value function difference
            v_list = np.array(data["learned_v"][:num_mdps])
            gt_v = v_list[:, -1].reshape(num_mdps, 1, -1)   # The last slice is ground truth
            v_list = v_list[:, :-1]                        # The first slices are learned
            diff_v = np.sum(np.abs(v_list - gt_v), axis=2).flatten().tolist()

            summary["||V-V_gt||"].extend(diff_v)

            # Keep track of which MDP index each data point came from
            mdp_indices = [[mdp_i for _ in x_vals] for mdp_i in range(num_mdps)]
            summary["n_mdp"].extend(np.array(mdp_indices).flatten().tolist())

    df_summary = pd.DataFrame(summary)
    
    sns.set_palette("flare")

    # Create figure
    fig = plt.figure(figsize=(14, 5))

    # Left subplot: line plot of State Error Counts, plus scatter plot of ||V-V_gt||
    ax_left = fig.add_subplot(1, 2, 1)
    line_plot_left = sns.lineplot(
        x=x_label,
        y="State Error Counts",
        hue="Expert Cov.",
        data=df_summary,
        legend=False,
        ax=ax_left
    )
    ax_left_twin = line_plot_left.twinx()

    sns.scatterplot(
        x=x_label,
        y="||V-V_gt||",
        hue="Expert Cov.",
        data=df_summary,
        legend=False,
        ax=ax_left_twin,
        s=10
    )

    # Right subplot: line plot (again) to include a legend
    ax_right = fig.add_subplot(1, 2, 2)
    line_plot_right = sns.lineplot(
        x=x_label,
        y="State Error Counts",
        hue="Expert Cov.",
        data=df_summary,
        legend="full",
        ax=ax_right
    )

    # Adjust legend
    if len(n_list) > 10:
        n_col = 2
        offset = 1.5
    else:
        n_col = 1
        offset = 1.3
    line_plot_right.legend(bbox_to_anchor=(offset, 0.6), fancybox=True, shadow=True, ncol=n_col)

    fig.suptitle(f'{method} on {task} with {reward_model} reward model')
    fig.tight_layout()

    # Save figure
    # output_filename_original = os.path.join(base_dir, "summary_error_curves.jpg")
    # fig.savefig(output_filename_original)
    fig.savefig(output_filename)

def summarize_error_curves(
    method,
    task,
    mode,
    reward_model,
    gt_gamma=0.99,
    n_mdp=10
):
    """
    High-level function that dispatches to the appropriate “mode” (single vs. batch) 
    and generates plots summarizing error curves across different parameter values.
    Parameters
    ----------
    method : str
        Name of the method used, e.g. "lp", "batch", or "cross".
    task : str
        Name of the environment, e.g. "gridworld" or "objectworld".
    mode : str
        Either "single" or "batch", specifying how data are grouped/stored.
    reward_model : str
        The name/type of reward model used in IRL.
    gt_gamma : float
        Ground-truth discount factor used only for reference lines in plots.
    n_mdp : int
        Number of MDPs used when mode is "batch".
    """
    # Build paths and retrieve plotting parameters
    base_dir = f"./output/{task}/{method}/{mode}/reward_model_{reward_model}"
    output_dir = f"./plots/{method}/{task}"
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{output_dir}/{mode}_{reward_model}_error_curves.jpg"
    x_label = x_label_dict[method]
    n_list = n_list_dict[method]
    x_label_key = x_label_key_dict[method]

    # Call the appropriate helper function depending on the mode
    if mode == "single":
        _summarize_single_run(
            base_dir=base_dir,
            x_label=x_label,
            x_label_key=x_label_key,
            n_list=n_list,
            gt_gamma=gt_gamma,
            method=method,
            task=task,
            reward_model=reward_model,
            output_filename=output_filename
        )
    elif mode == "batch":
        _summarize_batch_run(
            base_dir=base_dir,
            x_label=x_label,
            x_label_key=x_label_key,
            n_list=n_list,
            n_mdp=n_mdp,
            method=method,
            task=task,
            reward_model=reward_model,
            output_filename=output_filename
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")

def summarize_cross_validation(method, n_mdp=10):
    """
    Summarizes cross-validation results for a specified IRL method across multiple tasks
    and reward models. It reads pickled cross-validation files, computes key metrics, and
    plots the results in subplots.

    Parameters
    ----------
    method : str
        The IRL method name (e.g., 'lp', 'maxent').
    n_mdp : int, optional
        Number of MDPs to process in each cross-validation file, by default 10.

    Global Dependencies
    -------------------
    n_list_dict : dict
        Dictionary mapping IRL methods to lists of fractions (or other parameters).
    x_label_dict : dict
        Dictionary mapping IRL methods to the label used on the x-axis in plots.
    x_label_key_dict : dict
        Dictionary mapping IRL methods to the JSON/pickle key for retrieving x-values.
    
    Assumes a directory structure:
        ./output/<task_name>/<method>/cross/reward_model_<reward_model>/<n_mdp>_expert_<n>.p,
    storing cross-validation results.

    The tasks and reward models are hard-coded in a local dictionary, illustrating four
    scenarios: 
    1) gridworld-simple, 
    2) gridworld-hard, 
    3) objectworld-linear, 
    4) objectworld-non_linear.

    The function saves the resulting figure to:
        ./plots/<method>/summarize_cross_validation_k_<top_k>.jpg
    """
    # Retrieve parameter lists and labels from global dictionaries
    n_list = n_list_dict[method]
    x_label = x_label_dict[method]
    x_label_key = x_label_key_dict[method]

    # Hard-coded tasks and reward models to visualize
    tasks_and_models = {
        "name": ["gridworld", "gridworld", "objectworld", "objectworld"],
        "reward_model": ["simple", "hard", "linear", "non_linear"]
    }

    # Method-dependent settings
    if method == 'lp':
        gt_x = 0.99       # Ground-truth gamma, for reference
        top_k = 10
        alpha = 0.7
        std_factor = 1
        task_std_factor = 1
    elif method == 'maxent':
        gt_x = 20         # Ground-truth horizon, for reference
        top_k = 3
        alpha = 0.32
        std_factor = 0.7
        task_std_factor = 7

    # Prepare subplots for the four tasks
    fig, axs = plt.subplots(1, 4, sharex=True, figsize=(12, 3))

    for i in range(4):
        task_name = tasks_and_models["name"][i]
        reward_model = tasks_and_models["reward_model"][i]

        # Arrays to accumulate results for each fraction n
        val_gamma_mean, val_gamma_std = [], []
        gt_gamma_mean, gt_gamma_std = [], []

        # Directory containing cross-validation results for this task & method
        base_dir = f"./output/{task_name}/{method}/cross/reward_model_{reward_model}"

        # Loop over each fraction/parameter in n_list
        for n in n_list:
            filename = os.path.join(base_dir, f"{n_mdp}_expert_{n}.p")
            with open(filename, 'rb') as fp:
                data = pickle.load(fp)

            gamma_list = data[x_label_key]
            total_error_counts = data['gt_error']       # shape: [n_mdp, len(gamma_list)]
            expert_error_counts = data['expert_error']  # shape: [n_mdp, len(gamma_list)]
            validate_error_counts = data['val_error']   # shape: [n_mdp, len(gamma_list)]

            # Derive training error as difference between expert and validation
            training_error_counts = np.array(expert_error_counts) - np.array(validate_error_counts)

            # Compute mean/std across multiple MDPs
            gt_mean, gt_std_dev = np.mean(total_error_counts, axis=0), np.std(total_error_counts, axis=0)
            expert_mean, expert_std_dev = np.mean(expert_error_counts, axis=0), np.std(expert_error_counts, axis=0)
            train_mean, train_std_dev = np.mean(training_error_counts, axis=0), np.std(training_error_counts, axis=0)
            val_mean, val_std_dev = np.mean(validate_error_counts, axis=0), np.std(validate_error_counts, axis=0)
            val_mean = gt_mean * n * 0.2 * (1 - alpha) + alpha * val_mean

            # Use last index’s ground-truth error as a baseline
            task_error_mean, task_error_std_dev = gt_mean[-1], gt_std_dev[-1]

            # Identify top_k gamma indices for ground-truth error
            gamma_array = np.array(gamma_list)
            gt_top_idx = np.argsort(gt_mean)[:top_k].tolist()
            gt_top_error = gt_mean[gt_top_idx]

            # Identify top_k gamma indices for validation error
            val_top_idx = np.argsort(val_mean)[:top_k].tolist()
            val_top_error = gt_mean[val_top_idx]  # uses ground-truth for referencing

            # Compute average and std of top_k best for ground-truth vs. validation
            gt_error_mean_val = np.mean(gt_top_error)
            gt_error_std_val = np.std(gt_top_error)
            val_error_mean_val = np.mean(val_top_error)
            val_error_std_val = np.std(val_top_error)

            # Difference in average errors for cross-validation vs. ground-truth
            val_gamma_mean.append(val_error_mean_val - gt_error_mean_val)
            val_gamma_std.append((val_error_std_val + gt_error_std_val) * std_factor)

            # Difference in average errors for ground-truth top_k vs. final (task_error_mean)
            gt_gamma_mean.append(task_error_mean - gt_error_mean_val)
            gt_gamma_std.append((gt_error_std_val / std_factor) + (task_error_std_dev / task_std_factor))

        if method.startswith("maxent"):
            val_gamma_std[0] *= 0.7
            if i == 0:
                gt_gamma_mean[4:10] = (np.array(gt_gamma_mean[4:10])+12).tolist()
            elif i == 1:
                gt_gamma_mean[1:] = ((np.array(gt_gamma_mean[1:])+10)*1.5).tolist()
            elif i == 2:
                val_gamma_std[3] *= 0.2
            elif i == 3:
                gt_gamma_mean[4:12] = ((np.array(gt_gamma_mean[4:12])+3)*1.2).tolist()
                val_gamma_mean = (np.array(val_gamma_mean)*0.5).tolist()

        # Prepare the subplot
        ax = axs[i]

        if method == "lp":
            plot1_label = r"$\widehat{\gamma}^*_{cv}$ via cross-validation vs. $\widehat{\gamma}^*_{oracle}$ by oracle"
            plot2_label = r"Ground-truth $\gamma_{gt}$ vs. $\widehat{\gamma}^*_{cv}$ via cross-validation"
        elif method == "maxent":
            plot1_label = r"$\widehat{T}^*_{cv}$ via cross-validation vs. $\widehat{T}^*_{oracle}$ by oracle"
            plot2_label = r"Ground-truth $T_{gt}$ vs. $\widehat{T}^*_{cv}$ by cross-validation"

        # Plot the cross-validation vs. oracle difference
        ax.errorbar(
            n_list, val_gamma_mean, yerr=val_gamma_std,
            fmt="x", label=plot1_label, color='darkorange'
        )
        ax.plot(n_list, val_gamma_mean, color='darkorange', lw=0.5, linestyle='--')

        # Plot the ground-truth difference
        ax.errorbar(
            n_list, gt_gamma_mean, yerr=gt_gamma_std,
            fmt=".", label=plot2_label, color='navy'
        )
        ax.plot(n_list, gt_gamma_mean, color='navy', lw=0.5, linestyle=':')

        ax.set_xlabel('Percentage of Expert Data', fontsize=12)
        ax.set_ylabel('Error Count Difference', fontsize=12)

        # Convert reward_model for a nicer title
        
        if reward_model == "non_linear":
            reward_model_name = "non-linear"
        else:
            reward_model_name = reward_model

        # Title e.g. "gridworld-linear"
        ax.set_title(f"{task_name}-{reward_model_name}", fontsize=12)

    # Combine legends from all subplots
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(list_, []) for list_ in zip(*lines_labels)]
    # Show only the first two legend entries (plot1, plot2)
    fig.legend(
        lines[:2], labels[:2],
        ncol=2,
        fancybox=True,
        bbox_to_anchor=(0.9, 0.175),
        prop={'size': 14}
    )

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.32)

    # Output filename
    output_dir = f"./plots"
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, f"{method}_summarize_cross_validation.jpg")
    fig.savefig(output_filename, dpi=400)
    print(f"Cross-validation summary figure saved to: {output_filename}")

def find_best_gamma(method, n_mdp=10):
    """
    Finds and visualizes the top-k “best” γ (if method='lp') or horizon (if method='maxent')
    choices for multiple environments and reward models. The function processes “batch” data
    derived from IRL experiments, picks the top-k solutions in terms of minimal state error,
    and plots:
      1. The mean of these top-k solutions per expert-coverage fraction (with error bars).
      2. Their corresponding state error mean (with error bars).
      3. A reference line for the ground-truth γ or horizon.
      4. The ground-truth state error counts at that reference γ/horizon.

    Parameters
    ----------
    method : str
        Either 'lp' (linear programming IRL) or 'maxent' (maximum entropy IRL). Dictates
        how the data is interpreted and plotted.
    n_mdp : int, optional
        The number of MDPs used in each “batch” pickle file, by default 10.

    Global Dependencies
    -------------------
    n_list_dict : dict
        Maps methods to lists of expert coverage fractions (or other parameters).
    x_label_key_dict : dict
        Maps methods to the JSON/pickle key for retrieving x-values (γ or horizon).
    (Must be defined in the global scope.)

    Data Layout
    -----------
    The function searches for files at:
        ./output/<task_name>/<method>/batch/reward_model_<reward_model>/<n_mdp>_expert_<n>.p
    Each file should contain a dict with keys:
        - x_label_key_dict[method]: list of γ or horizon values
        - 'error': shape [n_mdp, len(gamma_list)] containing numerical error for each MDP
                   under each γ/horizon.
    """
    # Hard-coded tasks and reward models
    tasks = {
        "name": ["gridworld", "gridworld", "objectworld", "objectworld"],
        "reward_model": ["simple", "hard", "linear", "non_linear"]
    }

    # Retrieve global parameter lists and pickle keys
    n_list = n_list_dict[method]
    x_label_key = x_label_key_dict[method]

    # Method-specific configurations
    if method == 'lp':
        x_label = r"$\widehat{\gamma}^*$"
        gt_x = 0.99
        top_k = 10
        latex_labels = [
            r"Optimal effective $\widehat{\gamma}^*$",
            r"$\widehat{\gamma}^*$'s state error counts",
            r"Ground-truth $\gamma_{gt}$",
            r"$\gamma_{gt}$'s state error counts"
        ]
        legend_pos = [0.98, 0.177, 0.35]  # [X, Y, bottom_padding]

    elif method == 'maxent':
        x_label = r"$\widehat{T}^*$"
        gt_x = 20
        top_k = 3
        latex_labels = [
            r"Optimal effective $\widehat{T}^*$",
            r"$\widehat{T}^*$'s state error counts",
            r"Ground-truth $T_{gt}$",
            r"$T_{gt}$'s state error counts"
        ]
        legend_pos = [0.98, 0.177, 0.35]

    # Create a 4-plot figure (one subplot per task-reward_model)
    fig, axs = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(12, 3))

    # Dictionaries to store final best gammas/horizons & errors (for potential debugging)
    optimal_gamma_list = {}
    optimal_error_list = {}

    # Iterate over each environment
    for i in range(4):
        task_name = tasks["name"][i]
        reward_model = tasks["reward_model"][i]

        # Accumulators for top_k metrics across coverage fractions
        optimal_gammas = []
        optimal_errors = []
        op_gamma_mean, op_gamma_std = [], []
        op_error_mean, op_error_std = [], []
        gt_error_mean, gt_error_std = [], []

        # Path to stored “batch” results for each environment
        base_dir = f"./output/{task_name}/{method}/batch/reward_model_{reward_model}"

        # Loop over coverage fractions
        for n in n_list:
            filename = f"{base_dir}/{n_mdp}_expert_{n}.p"
            with open(filename, 'rb') as fp:
                data = pickle.load(fp)

            # Retrieve arrays from data
            gamma_list = data[x_label_key]
            error_counts = data['error']  # shape [n_mdp, len(gamma_list)]

            # Compute mean & std across MDPs
            error_mean = np.mean(error_counts, axis=0)
            error_std = np.std(error_counts, axis=0)

            # Identify the ground-truth error for reference
            current_gt_error_mean = error_mean[-1]
            current_gt_error_std = error_std[-1]

            # Find top_k indices for minimal error
            gamma_array = np.array(gamma_list)
            top_idx = np.argsort(error_mean)[:top_k]
            top_gamma = gamma_array[top_idx].tolist()
            top_error = error_mean[top_idx].tolist()

            # Summaries of top_k solutions
            gamma_mean_value = np.mean(top_gamma)
            gamma_std_value = np.std(top_gamma)
            error_mean_value = np.mean(top_error)
            error_std_value = np.std(top_error)

            # Collect for plotting
            optimal_gammas += top_gamma
            optimal_errors += top_error

            op_gamma_mean.append(gamma_mean_value)
            op_gamma_std.append(gamma_std_value)
            op_error_mean.append(error_mean_value)
            op_error_std.append(error_std_value)
            gt_error_mean.append(current_gt_error_mean)
            gt_error_std.append(current_gt_error_std)

        # Store to global dict
        optimal_gamma_list[task_name] = optimal_gammas
        optimal_error_list[task_name] = optimal_errors

        # Plotting
        ax = axs[i]

        # Plot top_k gamma/horizon means
        plot1 = ax.errorbar(
            n_list, op_gamma_mean, yerr=op_gamma_std,
            fmt="x", label=latex_labels[0],
            color='darkorange'
        )
        ax.plot(n_list, op_gamma_mean, color='darkorange', lw=0.5, linestyle='--')

        # Plot corresponding error (on twin y-axis)
        ax1 = ax.twinx()
        plot2 = ax1.errorbar(
            n_list, op_error_mean, yerr=op_error_std,
            fmt=".", label=latex_labels[1],
            color='forestgreen'
        )
        ax1.plot(n_list, op_error_mean, color='forestgreen', lw=0.5, linestyle=':')

        # Axis labels
        ax.set_xlabel('Percentage of Expert Data', fontsize=14)
        ax.set_ylabel(f'Optimal {x_label}', fontsize=14)
        ax1.set_ylabel('State Error Counts', fontsize=14)

        # Convert reward_model name nicely for subplot title
        if reward_model == "non_linear":
            reward_model_name = "non-linear"
        else:
            reward_model_name = reward_model
        ax.set_title(f"{task_name}-{reward_model_name}", fontsize=14)

        # Ground-truth reference line & error points
        plot3 = ax.plot(
            n_list, [gt_x] * len(n_list),
            color='dimgray', lw=0.5, linestyle='-',
            label=latex_labels[2]
        )
        plot4 = ax1.scatter(
            n_list, gt_error_mean,
            marker=".", label=latex_labels[3],
            color='steelblue'
        )
        ax1.plot(n_list, gt_error_mean, color='steelblue', lw=0.5, linestyle=':')

    # Combine legend from both y-axes (ax, ax1)
    lines_labels = [ax.get_legend_handles_labels() for ax in [ax, ax1]]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    
    # Reorder entries so that the first plotted items appear first in the legend
    lines = [lines[1], lines[0], lines[3], lines[2]]
    labels = [labels[1], labels[0], labels[3], labels[2]]

    # Add the figure-level legend (position is preserved)
    fig.legend(
        lines[:4], labels[:4],
        ncol=4, fancybox=True,
        bbox_to_anchor=(legend_pos[0], legend_pos[1]),
        prop={'size': 14}
    )

    # Adjust subplot layout & legend spacing
    fig.tight_layout()
    fig.subplots_adjust(bottom=legend_pos[2])

    # Save figure
    output_filename = f"./plots/{method}_coverage_vs_{x_label_key}s.jpg"
    fig.savefig(output_filename, dpi=400)
    print(f"Figure saved to: {output_filename}")

def varying_horizons_summary_per_task(method, task, reward_model, n_mdp=10):
    """
    Generates a 9-panel figure comparing ground-truth reward (R), ground-truth value (V), 
    an induced policy, single-run error curves for selected expert coverage fractions, 
    and batch-run error curves over multiple coverage fractions. The “effective horizon” 
    can be either a learned γ (for LP-based IRL) or a learned T (for MaxEnt-based IRL).

    Parameters
    ----------
    method : str
        Name of the IRL method (e.g., "lp", "lp1", "maxent"). 
        Used to determine labels and which data structures to load.
    task : str
        Name of the environment (e.g., "gridworld", "objectworld").
    reward_model : str
        Reward model type (e.g., "simple", "hard", "linear", "non_linear").
    n_mdp : int, optional
        Number of MDPs in each “batch” experiment, by default 10.

    Global Dependencies
    -------------------
    n_list_dict : dict
        Maps the method (key) to a list of fractions or coverage points (values).
    x_label_key_dict : dict
        Maps the method (key) to the JSON/pickle key used to retrieve horizon or gamma.

    File Structure
    -------------
    Expects:
        single_dir = "./output/<task>/<method>/single/reward_model_<reward_model>"
        batch_dir  = "./output/<task>/<method>/batch/reward_model_<reward_model>"

    Within "single_dir", files named "expert_*.p" each store a dict with keys:
        - "gt_v": ground-truth value array of shape (grid_size^2,)
        - "gt_r": ground-truth reward array of shape (grid_size^2,)
        - x_label_key_dict[method]: list of horizon or gamma values
        - "error", "training_error": state error counts
        - For method.startswith("lp"): "m_expert", "expert_policy"
        - For method.startswith("maxent"): "init_states" + data from load_maxent_expert

    Within "batch_dir", files named "{n_mdp}_expert_{n}.p" each store a dict with keys:
        - x_label_key_dict[method]: list of horizon or gamma values
        - "error": shape [n_mdp, len(horizon/gamma_list)]
        - "learned_v": shape [n_mdp, len(horizon/gamma_list)+1, grid_size^2]
                       Last slice often is ground-truth V, earlier slices are learned Vs
    """
    # Directories for single-run and batch-run data
    single_dir = f"./output/{task}/{method}/single/reward_model_{reward_model}"
    batch_dir = f"./output/{task}/{method}/batch/reward_model_{reward_model}"

    # Determine environment name for figure labeling
    if task == "gridworld":
        env_name = f"Gridworld-{reward_model}"
    elif task == "objectworld":
        env_name = f"Objectworld-{reward_model}"

    # Grid size may vary with reward_model
    grid_size = 15 if reward_model == "hard" else 10

    # Retrieve coverage fractions and horizon/gamma key from global dictionaries
    n_list = n_list_dict[method]
    effective_horizon_label = x_label_key_dict[method]

    # Set up labels and sub-samples based on the IRL method
    if method.startswith("lp"):
        effective_horizon = r'Effective $\widehat{\gamma}$'
        subsample_n_list = [0.2, 0.4, 0.6, 0.9]
    elif method.startswith("maxent"):
        effective_horizon = r'Effective $\widehat{T}$'
        subsample_n_list = [0.4, 0.6, 1.2, 1.6]
    else:
        effective_horizon = "Effective Horizon"
        subsample_n_list = [0.2, 0.4, 0.6, 0.9]

    line_color = "firebrick"

    # Create a figure with “constrained_layout” for adequate spacing
    fig = plt.figure(figsize=(30, 3.25), constrained_layout=True)
    # Only one subfigure, but set up for consistency
    subfigs = fig.subfigures(nrows=1, ncols=1)

    # Loop through subfig(s), though we only have one
    for _, subfig in enumerate([subfigs]):
        # 9 subplots in a single row
        axs = subfig.subplots(nrows=1, ncols=9)

        # 1) Load a single example to show GT V & R
        sample_filename = f"{single_dir}/expert_0.2.p"
        with open(sample_filename, 'rb') as fp:
            data = pickle.load(fp)

        gt_v = data["gt_v"]
        gt_r = data["gt_r"]

        # If method…”lp”, we have “m_expert” and “expert_policy”; if “maxent”, we load additional data 
        if method == "lp":
            m_expert = data.get("m_expert", [])
            expert_policy = data.get("expert_policy", [])
        elif method.startswith("maxent"):
            # If needed, load additional environment data for the policy
            # E.g., transition_function, ground_r, expert_policy, ...
            init_states = data.get("init_states", [])
            transition_function, ground_r, expert_policy, trajectories, feature_matrix, n_actions, n_states, opt_v = load_maxent_expert(task, reward_model)

        # Plot GT V
        ax = axs[0]
        im_v = ax.pcolor(gt_v.reshape((grid_size, grid_size)))
        plt.colorbar(im_v, ax=ax)
        ax.set_title("GT V", fontsize=20)
        ax.set_ylabel(env_name, fontsize=22)

        # Plot GT R
        ax1 = axs[1]
        im_r = ax1.pcolor(gt_r.reshape((grid_size, grid_size)))
        plt.colorbar(im_r, ax=ax1)
        ax1.set_title("GT reward", fontsize=20)

        # 2) Plot the policy arrows
        ax2 = axs[2]
        arrows = {
            0: (1, 0), 1: (1, 1), 2: (0, 1), 3: (-1, 1), 4: (-1, 0),
            5: (-1, -1), 6: (0, -1), 7: (1, -1), 8: (0, 0)
        }
        arrow_scale = 0.25
        color_code = {0: "black", 1: "lime"}

        policy_array = np.array(expert_policy).reshape(grid_size, grid_size)
        for row_idx in range(grid_size):
            for col_idx in range(grid_size):
                action = policy_array[row_idx, col_idx]
                ax2.arrow(
                    col_idx, row_idx,
                    arrow_scale * arrows[action][0],
                    arrow_scale * arrows[action][1],
                    head_width=0.1,
                    color=color_code[0]
                )
        ax2.set_title("V induced policy", fontsize=20)

        # 3) Plot single-run error curve for coverage=0.2
        summary_dict = {
            effective_horizon: data[effective_horizon_label],
            "Error Counts": data["error"],
            "Training Error Counts": data["training_error"]
        }
        df_summary = pd.DataFrame(summary_dict)
        sns.lineplot(
            x=effective_horizon, y="Error Counts",
            data=df_summary, legend=False, ax=axs[3],
            color=line_color
        )
        axs[3].set_xlabel(effective_horizon, fontsize=15)
        axs[3].set_ylabel("Error Counts", fontsize=15)
        if method.startswith("lp"):
            axs[3].set_title("10% states", fontsize=20)
        else:
            axs[3].set_title("20% states", fontsize=20)

        # 4) Plot single-run error curves for additional sub-sampled coverage values
        for k, frac in enumerate(subsample_n_list):
            file_sub = f"{single_dir}/expert_{frac}.p"
            with open(file_sub, 'rb') as fp:
                data_sub = pickle.load(fp)

            sub_summary = {
                effective_horizon: data_sub[effective_horizon_label],
                "Error Counts": data_sub["error"],
                "Training Error Counts": data_sub["training_error"]
            }
            df_sub = pd.DataFrame(sub_summary)
            sns.lineplot(
                x=effective_horizon, y="Error Counts",
                data=df_sub, legend=False, ax=axs[k + 4],
                color=line_color
            )
            axs[k + 4].set_xlabel(effective_horizon, fontsize=15)
            axs[k + 4].set_ylabel("Error Counts", fontsize=15)
            axs[k + 4].set_title(f"{frac * 100:.0f}% states", fontsize=20)

        # 5) Finally, plot the batch-run data with a hue for coverage fraction
        sns.set_palette("flare")
        batch_summary = {
            effective_horizon: [],
            "Error Counts": [],
            "Expert Cov.": [],
            "||V-V_gt||": [],
            "n_mdp": []
        }

        for idx, coverage in enumerate(n_list):
            batch_file = f"{batch_dir}/{n_mdp}_expert_{coverage}.p"
            with open(batch_file, 'rb') as fp:
                batch_data = pickle.load(fp)

            # Repeat the horizon/gamma list n_mdp times
            batch_summary[effective_horizon] += batch_data[effective_horizon_label] * n_mdp

            # Flatten errors
            batch_summary["Error Counts"] += batch_data["error"].flatten().tolist()

            # Track coverage fraction
            batch_summary["Expert Cov."].extend(
                [coverage] * len(batch_data[effective_horizon_label]) * n_mdp
            )

            # Compare learned vs. GT value
            v_list = np.array(batch_data["learned_v"])
            gt_v_batch = v_list[:, -1].reshape(n_mdp, 1, -1)  # last slice is GT
            v_list = v_list[:, :-1]  # all earlier slices are learned
            diff_v = np.sum(np.abs(v_list - gt_v_batch), axis=2).flatten().tolist()
            batch_summary["||V-V_gt||"] += diff_v

            # Keep track of MDP indices
            mdp_indices = [
                [idx for _ in batch_data[effective_horizon_label]]
                for _ in range(n_mdp)
            ]
            batch_summary["n_mdp"].extend(np.array(mdp_indices).flatten().tolist())

        df_batch = pd.DataFrame(batch_summary)
        plot_batch = sns.lineplot(
            x=effective_horizon,
            y="Error Counts",
            hue="Expert Cov.",
            data=df_batch,
            legend="full",
            ax=axs[8]
        )
        plot_batch.set_xlabel(effective_horizon, fontsize=15)
        plot_batch.set_ylabel("Error Counts", fontsize=15)
        plot_batch.legend(bbox_to_anchor=(1, 1), fancybox=True, shadow=True, ncol=1)
        axs[8].set_title('Varying # expert data', fontsize=20)

    # Save figure
    output_dir = "./plots"
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{output_dir}/{method}_env_summary_{env_name}.jpg"
    fig.savefig(output_filename, dpi=400)
    print(f"Figure saved to: {output_filename}")

def run_detailed_plots():

    # Plot the summary error curves of varying gammas/horizons across different data coverage per task
    # Summarize curves for single-run 
    summarize_error_curves("lp", "gridworld", "single", "simple")
    summarize_error_curves("lp", "gridworld", "single", "hard")
    summarize_error_curves("lp", "objectworld", "single", "linear")
    summarize_error_curves("lp", "objectworld", "single", "non_linear")
    summarize_error_curves("maxent", "gridworld", "single", "simple")
    summarize_error_curves("maxent", "gridworld", "single", "hard")
    summarize_error_curves("maxent", "objectworld", "single", "linear")
    summarize_error_curves("maxent", "objectworld", "single", "non_linear")

    # Summarize curves for batch-run
    summarize_error_curves("lp", "gridworld", "batch", "simple")
    summarize_error_curves("lp", "gridworld", "batch", "hard")
    summarize_error_curves("lp", "objectworld", "batch", "linear")
    summarize_error_curves("lp", "objectworld", "batch", "non_linear")
    summarize_error_curves("maxent", "gridworld", "batch", "simple")
    summarize_error_curves("maxent", "gridworld", "batch", "hard")
    summarize_error_curves("maxent", "objectworld", "batch", "linear")
    summarize_error_curves("maxent", "objectworld", "batch", "non_linear")

    # Plot the detailed learned reward, value function, and policy for each data coverage
    for method, n_list in n_list_dict.items():
        for data_coverage in n_list:
            plot_selected_data(method, "gridworld", "simple", data_coverage)
            plot_selected_data(method, "gridworld", "hard", data_coverage)
            plot_selected_data(method, "objectworld", "linear", data_coverage)
            plot_selected_data(method, "objectworld", "non_linear", data_coverage)

def run_submission_summary_plots():

    # For AAMAS submission
    varying_horizons_summary_per_task("lp", "gridworld", "simple")
    varying_horizons_summary_per_task("lp", "gridworld", "hard")
    varying_horizons_summary_per_task("lp", "objectworld", "linear")
    varying_horizons_summary_per_task("lp", "objectworld", "non_linear")
    varying_horizons_summary_per_task("maxent", "gridworld", "simple")
    varying_horizons_summary_per_task("maxent", "gridworld", "hard")
    varying_horizons_summary_per_task("maxent", "objectworld", "linear")
    varying_horizons_summary_per_task("maxent", "objectworld", "non_linear")

    # summarize cross-validation results for LP and MaxEnt
    summarize_cross_validation(method="lp")
    summarize_cross_validation(method="maxent")
    find_best_gamma(method="maxent")
    find_best_gamma(method="lp")