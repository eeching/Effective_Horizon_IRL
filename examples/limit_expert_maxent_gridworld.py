"""
Run linear programming inverse reinforcement learning on the gridworld MDP.

Matthew Alger, 2015
matthew.alger@anu.edu.au
"""

import numpy as np
import matplotlib.pyplot as plt

import irl.maxent as maxent
import irl.mdp.gridworld_random as gridworld
from irl.value_iteration import optimal_value, find_policy
import pdb
import pickle
import random
import tqdm

# use all expert demonstrations given, evaluate when comparing to the full expert_demonstrations
def test(grid_size, expert_fraction, epochs=200, learning_rate=0.01):
    """
    Run maxent inverse reinforcement learning on the gridworld MDP.

    Plots the reward function.

    grid_size: Grid size. int.
    discount: MDP discount factor. float.
    """
    # construct the env and get expert demonstrations.
    with open(f'./maxent_expert/gridworld_expert_length_8.pkl', 'rb') as fp:
        data = pickle.load(fp)
        goal_pos = list(data.keys())[0]
        demo = data[goal_pos]

    ground_r = demo["gt_r"]
    expert_policy = demo["expert_policy"]
    trajectories, idx_list, _, _ = demo["trajectories"]
    trajectories = trajectories[:idx_list[int(grid_size**2*expert_fraction)]+1]
    feature_matrix = demo["feature_matrix"]
    transition_function = demo["transition_function"]
    n_actions = demo["n_actions"]
    n_states = demo["n_states"]
    opt_v = demo["opt_v"]

    result = []

    fig, axs = plt.subplots(2, 11, layout="constrained", figsize=(35, 5), sharex=True, sharey=True)
    fig.suptitle(f"Performance for each gamma with {expert_fraction*100}% of expert coverage")
    gamma_list = [0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

    # plot the ground truth v and r
    ax = axs[0][0]
    im = ax.pcolor(opt_v.reshape((grid_size, grid_size)))
    plt.colorbar(im, ax=ax)

    ax1 = axs[1][0]
    im1 = ax1.pcolor(ground_r.reshape((grid_size, grid_size)))
    plt.colorbar(im1, ax=ax1)
    ax.set_title("GT V", fontsize='small')
    ax1.set_title("GT reward", fontsize='small')

    for i, gamma in enumerate(gamma_list):

        r, learned_V, policy = maxent_irl(feature_matrix, n_states, n_actions, gamma, transition_function, trajectories, epochs, learning_rate)
        diff = n_states - np.sum(np.equal(policy, expert_policy))
        result.append(diff)

        print("Diff", diff)

        ax2 = axs[1][i+1]
        im2 = ax2.pcolor(r.reshape((grid_size, grid_size)))
        plt.colorbar(im2, ax=ax2)

        ax3 = axs[0][i+1]
        im3 = ax3.pcolor(learned_V.reshape((grid_size, grid_size)))
        plt.colorbar(im3, ax=ax3)
        ax3.set_title(f"Gamma = {gamma}", fontsize='small')

    plt.savefig(f"./maxent_result/expert_{expert_fraction}_V_R_gridworld_length_8.jpg")
    gamma_list.reverse()
    result.reverse()
    print(result)
    with open(f'./output/gridworld/maxent/single_mdp/expert_{expert_fraction}_gridworld_length_8.pkl', 'wb') as fp:
        pickle.dump({"gamma": gamma_list, "error": result}, fp)
    plot_error_curve(expert_fraction, gamma_list=gamma_list, error=result)


# use all expert demonstrations given, evaluate when comparing to the full expert_demonstrations
def batch_test(grid_size, expert_fraction, n_mdp, num_gamma, epochs=200, learning_rate=0.01):

    with open(f'./maxent_expert/gridworld_expert_length_8.pkl', 'rb') as fp:
        data = pickle.load(fp)
        goal_poses_itr = iter(list(data.keys()))

    gamma_list = [(i+1)/num_gamma for i in range(num_gamma-1)] + [0.99]
    result = np.zeros((n_mdp, num_gamma))

    for i in range(n_mdp):
        goal_pos = next(goal_poses_itr)
        demo = data[goal_pos]
        expert_policy = demo["expert_policy"]
        trajectories, idx_list, _,  _ = demo["trajectories"]
        trajectories = trajectories[:idx_list[int(grid_size ** 2 * expert_fraction)] + 1]
        feature_matrix = demo["feature_matrix"]
        transition_function = demo["transition_function"]
        n_actions = demo["n_actions"]
        n_states = demo["n_states"]

        for j, gamma in enumerate(gamma_list):
            # run lp irl
            _, _, policy = maxent_irl(feature_matrix, n_states, n_actions, gamma, transition_function, trajectories, epochs, learning_rate)
            diff = n_states - np.sum(np.equal(policy, expert_policy))
            result[i][j] = diff
            print(f"MDP {i}, gamma {gamma}, error {diff}")

        with open(f'./output/gridworld/maxent/batch/{n_mdp}_expert_{expert_fraction}_length_8.p', 'wb') as fp:
            pickle.dump({"gamma": gamma_list, "error": result, "n_mdp": i+1}, fp)

    print(result)
    plot_batch_error_curve(expert_fraction, gamma_list=gamma_list, error=result, batch=n_mdp)

# use a fraction of the given expert demonstration, choose gamma using the validation set, evaluate using the full expert
# (0.8, 0.2) training and validating splits
def cross_validate(grid_size, expert_fraction, n_mdp, num_gamma, epochs=200, learning_rate=0.01):

    with open(f'./maxent_expert/gridworld_expert_length_8.pkl', 'rb') as fp:
        data = pickle.load(fp)
        goal_poses_itr = iter(list(data.keys()))

    gamma_list = [(i + 1) / num_gamma for i in range(num_gamma - 1)] + [0.99]
    gt_error, expert_error, validate_error = np.zeros((n_mdp, num_gamma)), np.zeros((n_mdp, num_gamma)), np.zeros((n_mdp, num_gamma))

    for i in range(n_mdp):
        goal_pos = next(goal_poses_itr)
        demo = data[goal_pos]
        expert_policy = demo["expert_policy"]
        trajectories, idx_list, cached_num_state_list, _ = demo["trajectories"]
        # trajectories = trajectories[:idx_list[int(grid_size ** 2 * expert_fraction)] + 1]
        training_traj = trajectories[:idx_list[int(grid_size ** 2 * expert_fraction*0.8)] + 1]
        # validate_traj = trajectories[idx_list[int(grid_size ** 2 * expert_fraction*0.8)] + 1 : idx_list[int(grid_size ** 2 * expert_fraction)+1]]
        traj_states = cached_num_state_list[int(grid_size ** 2 * expert_fraction)]
        train_states = cached_num_state_list[int(grid_size ** 2 * expert_fraction*0.8)]
        validate_states = traj_states - train_states
        feature_matrix = demo["feature_matrix"]
        transition_function = demo["transition_function"]
        n_actions = demo["n_actions"]
        n_states = demo["n_states"]

        for j, gamma in enumerate(gamma_list):
            # run lp irl
            _, _, policy = maxent_irl(feature_matrix, n_states, n_actions, gamma, transition_function, training_traj, epochs, learning_rate)
            expert_diff = np.sum([policy[i] != expert_policy[i] for i in traj_states])
            val_diff = np.sum([policy[i] != expert_policy[i] for i in validate_states])
            gt_diff = n_states - np.sum(np.equal(policy, expert_policy))
            gt_error[i][j] = gt_diff
            expert_error[i][j] = expert_diff
            validate_error[i][j] = val_diff
            print(f"MDP {i}, gamma {gamma}, val error {val_diff}, expert_error {expert_diff}, gt_error {gt_diff}")

        with open(f'./output/gridworld/maxent/cross_validate/{n_mdp}_expert_{expert_fraction}_length_8.p', 'wb') as fp:
            pickle.dump(
                {"gamma": gamma_list, "gt_error": gt_error, "val_error": validate_error, "expert_error": expert_error},
                fp)

    plot_cross_validation_curve(expert_fraction, n_states, gamma_list=gamma_list, gt_error=gt_error, val_error=validate_error, expert_error=expert_error,  batch=n_mdp)


def maxent_irl(feature_matrix, n_states, n_actions, training_discount, transition_function, trajectories, epochs, learning_rate):

    r = maxent.irl(feature_matrix, n_actions, training_discount,
                   transition_function, trajectories, epochs, learning_rate)
    value = optimal_value(n_states, n_actions, transition_function, r, training_discount)
    policy = find_policy(n_states, n_actions, transition_function, r, training_discount, threshold=1e-2, v=value,
                         stochastic=False)
    return r, value, policy


def plot_error_curve(expert_fraction, filename=None, gamma_list=None, error=None):

    if filename is not None:
        with open(filename, 'rb') as fp:
            data = pickle.load(fp)
            gamma_list = data['gamma']
            error = data['error']

    fig, ax = plt.subplots()
    ax.plot(gamma_list, error)
    ax.set_xlabel('Gamma', fontsize="medium")
    ax.set_ylabel('Error Count', fontsize="medium")
    ax.set_title('Discrepancy between the induced policy and the expert for different Gammas', fontsize="large")
    fig.tight_layout()
    plt.savefig(f"./output/gridworld/maxent/single_mdp/expert_{expert_fraction}_error_curve_length_8.jpg")
    plt.show()


def plot_batch_error_curve(expert_fraction, filename=None, gamma_list=None, error=None, batch=0):

    if filename is not None:
        with open(filename, 'rb') as fp:
            data = pickle.load(fp)
            gamma_list = data['gamma']
            error = data['error']
    mean, std = np.mean(error, axis=0), np.std(error, axis=0)

    fig, ax = plt.subplots()
    ax.plot(gamma_list, mean, lw=2, color='blue')
    ax.fill_between(gamma_list, mean + std, mean - std, facecolor='blue', alpha=0.5)

    ax.set_xlabel('Gamma', fontsize="medium")
    ax.set_ylabel('Average Error Count', fontsize="medium")
    ax.set_title('Discrepancy between the induced policy and the expert for different Gammas', fontsize="large")
    fig.tight_layout()

    plt.savefig(f"./output/gridworld/maxent/batch/{batch}_MDPs_expert_{expert_fraction}_error_curve_length_8.jpg")


def plot_cross_validation_curve(expert_fraction, n_states, filename=None, gamma_list=None, gt_error=None, val_error=None, expert_error=None, batch=0):

    if filename is not None:
        with open(filename, 'rb') as fp:
            data = pickle.load(fp)
            gamma_list = data['gamma']
            gt_error = data['gt_error']
            val_error = data['val_error']
            expert_error = data['expert_error']

    gt_mean, gt_std = np.mean(gt_error/n_states, axis=0), np.std(gt_error/n_states, axis=0)

    m_expert = int(expert_fraction*n_states)
    expert_mean, expert_std = np.mean(expert_error / m_expert, axis=0), np.std(expert_error / m_expert, axis=0)

    k_val = m_expert - int(m_expert*0.8)
    val_mean, val_std = np.mean(val_error / k_val, axis=0), np.std(val_error / k_val, axis=0)

    fig, ax = plt.subplots(layout="constrained", figsize=(6, 6))
    ax.plot(gamma_list, val_mean, label='Validation Error', lw=2, color='blue')
    ax.fill_between(gamma_list, val_mean + val_std, val_mean - val_std, facecolor='blue', alpha=0.3)

    ax.plot(gamma_list, expert_mean, label='Training + validation Error', lw=2, color='purple')
    ax.fill_between(gamma_list, expert_mean + expert_std, expert_mean - expert_std, facecolor='purple', alpha=0.3)

    ax.plot(gamma_list, gt_mean, label='GroundTruth Error', lw=2, color='darkgreen')
    ax.fill_between(gamma_list, gt_mean + gt_std, gt_mean - gt_std, facecolor='darkgreen', alpha=0.3)

    ax.set_xlabel('Gamma', fontsize="medium")
    ax.set_ylabel('Percentage Error', fontsize="medium")
    ax.set_title('Percentage of Error for different Gammas')
    ax.legend(loc='lower right')
    # fig.tight_layout()
    plt.savefig(f"./output/gridworld/maxent/cross_validate/{batch}_MDPs_expert_{expert_fraction}_error_curve_length_8.jpg")

    # plt.show()


def cache_expert_demo(grid_size, n_mdp):
    np.random.seed(0)
    goal_states = np.random.choice(range(grid_size ** 2), 20, replace=False)
    demo = {}

    for i in tqdm(range(n_mdp)):
        goal_pos = goal_states[i]
        gw = gridworld.GridworldRandom(grid_size, 0.1, 0.99, V=True, goal_pos=goal_pos)
        ground_r = np.array([gw.reward(s) for s in range(gw.n_states)])
        trajectory_length = 50
        trajectories, cached_idx_list, cached_num_state_list, state_num_list = gw.generate_trajectories(gw.n_states, trajectory_length)
        feature_matrix = gw.feature_matrix()
        demo[goal_pos] = {"gt_r": ground_r, "expert_policy": gw.policy, "trajectories": [trajectories, cached_idx_list, cached_num_state_list, state_num_list], "feature_matrix":
                          feature_matrix, "transition_function": gw.transition_probability, "n_actions": gw.n_actions, "n_states": gw.n_states, "opt_v": gw.opt_v}

        with open(f'./maxent_expert/gridworld_expert_gridsize_{grid_size}_traj_len_{trajectory_length}.pkl', 'wb') as fp:
            pickle.dump(demo, fp)

if __name__ == '__main__':

    # MDP grid size, gt_gamma, expert_fraction, n_mdps, n_gamma
    grid_size = 50
    cache_expert_demo(grid_size, 20)

    # test(10, 1)
    # batch_test(grid_size, 1, 10, 10, epochs=200, learning_rate=0.01)
    # cross_validate(grid_size, 1, 10, 10, epochs=200, learning_rate=0.01)
