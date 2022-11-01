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
def batch_test(grid_size, gt_discount, expert_fraction, n_mdp):
    wind = 0.1
    gamma_list = [i/20 for i in range(20)]
    result = [0] * len(gamma_list)

    for i in range(n_mdp):
        gw = gridworld.GridworldRandom(grid_size, wind, gt_discount)
        expert_policy = gw.policy
        expert_idx = int(gw.n_states * expert_fraction)
        m_expert = gw.generate_expert_demonstrations(expert_idx)

        for j, gamma in enumerate(gamma_list):
            # run lp irl
            _, _, policy = lp_irl(gw, m_expert, gamma)
            diff = gw.n_states - np.sum(np.equal(policy, expert_policy))
            result[j] += diff
            print(f"MDP {i}, gamma {gamma}, error {diff}")

    result[:] = [x / n_mdp for x in result]
    print(result)
    with open(f'./batch/{n_mdp}_expert_{expert_fraction}.p', 'wb') as fp:
        pickle.dump({"gamma": gamma_list, "error": result}, fp, protocol=pickle.HIGHEST_PROTOCOL)
    plot_error_curve(expert_fraction, gamma_list=gamma_list, error=result, batch=n_mdp)

# use all expert demonstrations given, evaluate when comparing to the full expert_demonstrations
def test(grid_size, gt_discount, expert_fraction):
    """
    Run linear programming inverse reinforcement learning on the gridworld MDP.

    Plots the reward function.

    grid_size: Grid size. int.
    discount: MDP discount factor. float.
    """

    # construct the env and get expert demonstrations.
    wind = 0.1
    gw = gridworld.GridworldRandom(grid_size, wind, gt_discount, seed=0)
    ground_r = np.array([gw.reward(s) for s in range(gw.n_states)])
    expert_policy = gw.policy
    expert_idx = int(gw.n_states * expert_fraction)
    m_expert = gw.generate_expert_demonstrations(expert_idx)
    # plt.show()
    # expert_return = gw.evaluate_expert_policy()
    # print(expert_return)

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
    plt.show()
    gamma_list.reverse()
    result.reverse()
    print(result)
    with open(f'expert_{expert_fraction}.p', 'wb') as fp:
        pickle.dump({"gamma": gamma_list, "error": result}, fp, protocol=pickle.HIGHEST_PROTOCOL)
    plot_error_curve(expert_fraction, gamma_list=gamma_list, error=result)

# use a fraction of the given expert demonstration, choose gamma using the validation set, evaluate using the full expert
# (0.8, 0.2) training and validating splits
def cross_validate(grid_size, gt_discount, expert_fraction, n_mdp):

    wind = 0.1
    gamma_list = [i / 20 for i in range(1, 20)]
    gt_error, expert_error, validate_error = [0] * len(gamma_list), [0] * len(gamma_list), [0] * len(gamma_list)

    for _ in range(n_mdp):
        gw = gridworld.GridworldRandom(grid_size, wind, gt_discount)
        expert_policy = gw.policy
        expert_idx = int(gw.n_states * expert_fraction)
        m_expert = gw.generate_expert_demonstrations(expert_idx).tolist()
        training = set(random.sample(m_expert, int(expert_idx*0.8)))
        validation = list(set(m_expert) - training)
        training = list(training)
        validation.sort()
        training.sort()


        for j, gamma in enumerate(gamma_list):
            # run lp irl
            _, _, policy = lp_irl(gw, training, gamma)
            expert_diff = np.sum([policy[i] != expert_policy[i] for i in m_expert])
            val_diff = np.sum([policy[i] != expert_policy[i] for i in validation])
            gt_diff = gw.n_states - np.sum(np.equal(policy, expert_policy))
            gt_error[j] += gt_diff
            expert_error[j] += expert_diff
            validate_error[j] += val_diff
            print(f"MDP {j}, gamma {gamma}, val error {val_diff}, expert_error {expert_diff}, gt_error {gt_diff}")

    gt_error[:] = [x / (n_mdp * gw.n_states)  for x in gt_error]
    validate_error[:] = [x / (n_mdp * len(validation)) for x in validate_error]
    expert_error[:] = [x / (n_mdp * len(m_expert)) for x in expert_error]

    with open(f'./batch/{n_mdp}_expert_{expert_fraction}.p', 'wb') as fp:
        pickle.dump({"gamma": gamma_list, "gt_error": gt_error, "val_error": validate_error, "expert_error": expert_error}, fp, protocol=pickle.HIGHEST_PROTOCOL)
    plot_cross_validation_curve(expert_fraction, gamma_list=gamma_list, gt_error=gt_error, val_error=validate_error, expert_error=expert_error,  batch=n_mdp)


def lp_irl(gw, m_expert, training_discount):

    r = linear_irl.irl(gw.n_states, gw.n_actions, gw.transition_probability, gw.policy, m_expert, training_discount, 1, 1)
    V, policy = gw.evaluate_learnt_reward(r, training_discount)
    return r, V, policy


def plot_error_curve(expert_fraction, filename=None, gamma_list=None, error=None, batch=0):

    if filename is not None:
        with open(filename, 'wb') as fp:
            data = pickle.load(fp, protocol=pickle.HIGHEST_PROTOCOL)
            gamma_list = data['gamma']
            error = data['error']

    fig, ax = plt.subplots()
    ax.plot(gamma_list, error)
    ax.set_xlabel('Gamma', fontsize="medium")
    ax.set_ylabel('Error Count', fontsize="medium")
    ax.set_title('Discrepancy between the induced policy and the expert for different Gammas', fontsize="large")
    fig.tight_layout()
    if batch > 0:
        plt.savefig(f"./batch/{batch}_MDPs_expert_{expert_fraction}_error_curve.jpg")
    else:
        plt.savefig(f"expert_{expert_fraction}_error_curve.jpg")
    # plt.show()

def plot_cross_validation_curve(expert_fraction, filename=None, gamma_list=None, gt_error=None, val_error=None, expert_error=None, batch=0):

    if filename is not None:
        with open(filename, 'wb') as fp:
            data = pickle.load(fp, protocol=pickle.HIGHEST_PROTOCOL)
            gamma_list = data['gamma']
            gt_error = data['gt_error']
            val_error = data['val_error']
            expert_error = data['expert_error']

    fig, axes = plt.subplots(3, 1, layout="constrained", figsize=(10, 10), sharex=True)
    axes[0].plot(gamma_list, val_error)
    axes[0].set_ylabel('Validation Error', fontsize="medium")
    axes[0].set_title('Percentage of error in validation set', fontsize="medium")

    axes[1].plot(gamma_list, expert_error)
    axes[1].set_ylabel('Training + Validation Error', fontsize="medium")
    axes[1].set_title('Percentage of error in the given expert set', fontsize="medium")

    axes[2].plot(gamma_list, gt_error)
    axes[2].set_xlabel('Gamma', fontsize="medium")
    axes[2].set_ylabel('GroundTruth Error', fontsize="medium")
    axes[2].set_title('Percentage of error for the complete expert', fontsize="medium")

    fig.tight_layout()
    plt.savefig(f"./cross_validate/{batch}_MDPs_expert_{expert_fraction}_error_curve.jpg")

    # plt.show()


if __name__ == '__main__':
    # test(10, 0.99, 0.10)
    # batch_test(10, 0.99, 0.1, 100)
    cross_validate(10, 0.99, 0.2, 100)

