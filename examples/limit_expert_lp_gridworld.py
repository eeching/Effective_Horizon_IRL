"""
Run linear programming inverse reinforcement learning on the gridworld MDP.

Matthew Alger, 2015
matthew.alger@anu.edu.au
"""

import numpy as np
import matplotlib.pyplot as plt

import irl.partial_expert_lp as linear_irl
import irl.mdp.gridworld_random as gridworld
import pdb
import pickle
import random

# use all expert demonstrations given, evaluate when comparing to the full expert_demonstrations
def test(grid_size, gt_discount, expert_fraction):
    """
    Run linear programming inverse reinforcement learning on the gridworld MDP.

    Plots the reward function.

    grid_size: Grid size. int.
    discount: MDP discount factor. float.
    """

    # construct the env and get expert demonstrations.
    gw = gridworld.GridworldRandom(grid_size, 0.1, gt_discount, V=True, seed=0)
    ground_r = np.array([gw.reward(s) for s in range(gw.n_states)])
    expert_policy = gw.policy
    expert_idx = int(gw.n_states * expert_fraction)
    m_expert = gw.generate_expert_demonstrations(expert_idx)

    result = []

    fig, axs = plt.subplots(2, 11, layout="constrained", figsize=(35, 5), sharex=True, sharey=True)
    fig.suptitle(f"Performance for each gamma with {expert_fraction*100}% of expert coverage")
    gamma_list = [0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

    # plot the ground truth v and r
    ax = axs[0][0]
    im = ax.pcolor(gw.opt_v.reshape((grid_size, grid_size)))
    plt.colorbar(im, ax=ax)

    ax1 = axs[1][0]
    im1 = ax1.pcolor(ground_r.reshape((grid_size, grid_size)))
    plt.colorbar(im1, ax=ax1)
    ax.set_title("GT V", fontsize='small')
    ax1.set_title("GT reward", fontsize='small')

    for i, gamma in enumerate(gamma_list):
        # run lp irl
        r, learned_V, policy = lp_irl(gw, m_expert, gamma)
        diff = gw.n_states - np.sum(np.equal(policy, expert_policy))
        result.append(diff)

        print("Diff", diff)

        ax2 = axs[1][i+1]
        im2 = ax2.pcolor(r.reshape((grid_size, grid_size)))
        plt.colorbar(im2, ax=ax2)

        ax3 = axs[0][i+1]
        im3 = ax3.pcolor(learned_V.reshape((grid_size, grid_size)))
        plt.colorbar(im3, ax=ax3)
        ax3.set_title(f"Gamma = {gamma}", fontsize='small')
    plt.savefig(f"expert_{expert_fraction}_V_R.jpg")
    # plt.show()
    gamma_list.reverse()
    result.reverse()
    print(result)
    with open(f'./output/gridworld/lp/single_mdp/expert_{expert_fraction}.p', 'wb') as fp:
        pickle.dump({"gamma": gamma_list, "error": result}, fp)
    plot_error_curve(expert_fraction, gamma_list=gamma_list, error=result)

# use all expert demonstrations given, evaluate when comparing to the full expert_demonstrations
def batch_test(grid_size, gt_discount, expert_fraction, n_mdp, num_gamma):
    wind = 0.1
    gamma_list = [(i+1)/num_gamma for i in range(num_gamma-1)] + [0.99]
    result = np.zeros((n_mdp, num_gamma))

    for i in range(n_mdp):
        gw = gridworld.GridworldRandom(grid_size, wind, gt_discount)
        expert_policy = gw.policy
        expert_idx = int(gw.n_states * expert_fraction)
        m_expert = gw.generate_expert_demonstrations(expert_idx)

        for j, gamma in enumerate(gamma_list):
            # run lp irl
            _, _, policy = lp_irl(gw, m_expert, gamma)
            diff = gw.n_states - np.sum(np.equal(policy, expert_policy))
            result[i][j] = diff
            print(f"MDP {i}, gamma {gamma}, error {diff}")

        with open(f'./output/gridworld/lp/batch/{n_mdp}_expert_{expert_fraction}.p', 'wb') as fp:
            pickle.dump({"gamma": gamma_list, "error": result, "n_mdp": i+1}, fp)

    print(result)
    plot_batch_error_curve(expert_fraction, gamma_list=gamma_list, error=result, batch=n_mdp)

# use a fraction of the given expert demonstration, choose gamma using the validation set, evaluate using the full expert
# (0.8, 0.2) training and validating splits
def cross_validate(grid_size, gt_discount, expert_fraction, n_mdp, num_gamma):

    wind = 0.1
    gamma_list = [(i + 1) / num_gamma for i in range(num_gamma - 1)] + [0.99]
    gt_error, expert_error, validate_error = np.zeros((n_mdp, num_gamma)), np.zeros((n_mdp, num_gamma)), np.zeros((n_mdp, num_gamma))

    for i in range(n_mdp):
        gw = gridworld.GridworldRandom(grid_size, wind, gt_discount)
        expert_policy = gw.policy
        expert_idx = int(gw.n_states * expert_fraction)
        m_expert, training, validation = gw.generate_expert_demonstrations(expert_idx, cross_validate=0.8)

        for j, gamma in enumerate(gamma_list):
            # run lp irl
            _, _, policy = lp_irl(gw, training, gamma)
            expert_diff = np.sum([policy[i] != expert_policy[i] for i in m_expert])
            val_diff = np.sum([policy[i] != expert_policy[i] for i in validation])
            gt_diff = gw.n_states - np.sum(np.equal(policy, expert_policy))
            gt_error[i][j] = gt_diff
            expert_error[i][j] = expert_diff
            validate_error[i][j] = val_diff
            print(f"MDP {i}, gamma {gamma}, val error {val_diff}, expert_error {expert_diff}, gt_error {gt_diff}")

        with open(f'./output/gridworld/lp/cross_validate/{n_mdp}_expert_{expert_fraction}.p', 'wb') as fp:
            pickle.dump(
                {"gamma": gamma_list, "gt_error": gt_error, "val_error": validate_error, "expert_error": expert_error, "m_expert": expert_idx},
                fp)

    plot_cross_validation_curve(expert_fraction, gw.n_states, gamma_list=gamma_list, gt_error=gt_error, val_error=validate_error, expert_error=expert_error,  batch=n_mdp)


def lp_irl(gw, m_expert, training_discount):

    r = linear_irl.irl(gw.n_states, gw.n_actions, gw.transition_probability, gw.policy, m_expert, training_discount, 1, 1)
    V, policy = gw.evaluate_learnt_reward(r, training_discount)
    return r, V, policy


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
    plt.savefig(f"./output/gridworld/lp/single_mdp/expert_{expert_fraction}_error_curve.jpg")
    # plt.show()


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

    plt.savefig(f"./output/gridworld/lp/batch/{batch}_MDPs_expert_{expert_fraction}_error_curve.jpg")

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
    plt.savefig(f"./output/gridworld/lp/cross_validate/{batch}_MDPs_expert_{expert_fraction}_error_curve.jpg")

    # plt.show()


if __name__ == '__main__':

    # MDP grid size, gt_gamma, expert_fraction, n_mdps, n_gamma

    batch_test(10, 0.99, 1, 20, 12)
    # cross_validate(10, 0.99, 0.2, 20, 12)
