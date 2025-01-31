"""
Implements maximum entropy inverse reinforcement learning (Ziebart et al., 2008)

Matthew Alger, 2015
matthew.alger@anu.edu.au
"""

from itertools import product
import jax.numpy as np
from . import value_iteration_jax as value_iteration
from tqdm import tqdm
from jax import jit, random, vmap
import pdb
from functools import partial
import numpy
import math


def irl(feature_matrix, n_actions, discount, transition_probability,
        trajectories, epochs, learning_rate, finite_horizon=None, gt_alpha=None):
    """
    Find the reward function for the given trajectories.

    feature_matrix: Matrix with the nth row representing the nth state. NumPy
        array with shape (N, D) where N is the number of states and D is the
        dimensionality of the state.
    n_actions: Number of actions A. int.
    discount: Discount factor of the MDP. float.
    transition_probability: NumPy array mapping (state_i, action, state_k) to
        the probability of transitioning from state_i to state_k under action.
        Shape (N, A, N).
    trajectories: 3D array of state/action pairs. States are ints, actions
        are ints. NumPy array with shape (T, L, 2) where T is the number of
        trajectories and L is the trajectory length.
    epochs: Number of gradient descent steps. int.
    learning_rate: Gradient descent learning rate. float.
    -> Reward vector with shape (N,).
    """

    n_states, d_states = feature_matrix.shape

    # Initialise weights.
    key = random.PRNGKey(0)
    # initialize 10 random alphas
    alphas = random.uniform(key, (10, d_states,))

    # Calculate the feature expectations \tilde{phi}.
    feature_matrix = np.array(feature_matrix)
    transition_probability = np.array(transition_probability)
    trajectories = np.array(trajectories)
    feature_expectations, p_start_state = find_feature_expectations(feature_matrix,
                                                     trajectories, n_states)

    results = vmap(run_irl, in_axes = (0, None, None, None, None, None, None, None, None, None, None, None), out_axes=0)(alphas, feature_matrix, n_states, n_actions, discount, transition_probability, trajectories, p_start_state, finite_horizon, feature_expectations, epochs, learning_rate)

    r = results[1][np.argmin(results[0])]

    if gt_alpha is not None:
        pdb.set_trace()
        gt_r = np.matmul(feature_matrix, gt_alpha)
        expected_svf = find_expected_svf(n_states, r, n_actions, discount, transition_probability, trajectories, p_start_state, finite_horizon)
        grad = feature_expectations - feature_matrix.T.dot(expected_svf)

        print(f"gt alpha with error {np.sum(np.abs(grad))}")
    return r
        
def run_irl(alpha, feature_matrix, n_states, n_actions, discount, transition_probability, trajectories, p_start_state, finite_horizon, feature_expectations, epochs, learning_rate):

    for i in tqdm(range(epochs)):
        r = np.matmul(feature_matrix, alpha)
        expected_svf = find_expected_svf(n_states, r, n_actions, discount, transition_probability, trajectories, p_start_state, finite_horizon)
        grad = feature_expectations - feature_matrix.T.dot(expected_svf)
        alpha += learning_rate * grad
       
    r = np.matmul(feature_matrix, alpha).reshape((n_states,))
    return (np.sum(np.abs(grad)), r)

@partial(jit, static_argnames=['n_states'])
def find_feature_expectations(feature_matrix, trajectories, n_states):
    """
    Find the feature expectations for the given trajectories. This is the
    average path feature vector.

    feature_matrix: Matrix with the nth row representing the nth state. NumPy
        array with shape (N, D) where N is the number of states and D is the
        dimensionality of the state.
    trajectories: 3D array of state/action pairs. States are ints, actions
        are ints. NumPy array with shape (T, L, 2) where T is the number of
        trajectories and L is the trajectory length.
    -> Feature expectations vector with shape (D,).
    """

    n_trajectories = trajectories.shape[0]
    feature_expectations = np.zeros(feature_matrix.shape[1])
    start_state_count = np.zeros(n_states)

    for trajectory in trajectories:
        start_state_count = start_state_count.at[trajectory[0, 0]].add(1)
        for state, _ in trajectory:
            feature_expectations += feature_matrix[state]

    feature_expectations /= n_trajectories
    p_start_state = start_state_count/n_trajectories

    return feature_expectations, p_start_state

@partial(jit, static_argnames=['n_states', 'n_actions', 'discount', 'finite_horizon'],)
def find_expected_svf(n_states, r, n_actions, discount,
                      transition_probability, trajectories, p_start_state, finite_horizon):
    """
    Find the expected state visitation frequencies using algorithm 1 from
    Ziebart et al. 2008.

    n_states: Number of states N. int.
    alpha: Reward. NumPy array with shape (N,).
    n_actions: Number of actions A. int.
    discount: Discount factor of the MDP. float.
    transition_probability: NumPy array mapping (state_i, action, state_k) to
        the probability of transitioning from state_i to state_k under action.
        Shape (N, A, N).
    trajectories: 3D array of state/action pairs. States are ints, actions
        are ints. NumPy array with shape (T, L, 2) where T is the number of
        trajectories and L is the trajectory length.
    -> Expected state visitation frequencies vector with shape (N,).
    """

    n_trajectories = trajectories.shape[0]
    trajectory_length = trajectories.shape[1]

    # policy = find_policy(n_states, r, n_actions, discount,
    #                                 transition_probability)

    v = value_iteration.optimal_value(n_states, n_actions, transition_probability, r, discount, T=finite_horizon)

    policy = value_iteration.find_policy(n_states, n_actions,
                                         transition_probability, r, discount, v, T=finite_horizon)
    

    # start_state_count = np.zeros(n_states)
    # for trajectory in trajectories:
    #     start_state_count[trajectory[0, 0]] += 1
    # p_start_state = start_state_count/n_trajectories

    expected_svf = np.tile(p_start_state, (trajectory_length, 1)).T
    for t in range(1, trajectory_length):
        value = np.matmul(transition_probability.transpose(2, 0, 1).reshape(n_states, -1), np.matmul(np.diag(expected_svf[:, t-1]), policy).flatten())
        expected_svf = expected_svf.at[:, t].set(value)

       
    return expected_svf.sum(axis=1)

@partial(jit, static_argnames=['n_states', 'n_actions', 'discount', 'finite_horizon'],)
def expected_value_difference(n_states, n_actions, transition_probability,
    reward, discount, p_start_state, optimal_value, true_reward, finite_horizon):
    """
    Calculate the expected value difference, which is a proxy to how good a
    recovered reward function is.

    n_states: Number of states. int.
    n_actions: Number of actions. int.
    transition_probability: NumPy array mapping (state_i, action, state_k) to
        the probability of transitioning from state_i to state_k under action.
        Shape (N, A, N).
    reward: Reward vector mapping state int to reward. Shape (N,).
    discount: Discount factor. float.
    p_start_state: Probability vector with the ith component as the probability
        that the ith state is the start state. Shape (N,).
    optimal_value: Value vector for the ground reward with optimal policy.
        The ith component is the value of the ith state. Shape (N,).
    true_reward: True reward vector. Shape (N,).
    -> Expected value difference. float.
    """

    policy = value_iteration.find_policy(n_states, n_actions,
        transition_probability, reward, discount, T=finite_horizon)
    value = value_iteration.value(policy.argmax(axis=1), n_states,
        transition_probability, true_reward, discount, T=finite_horizon)

    evd = np.matmul(optimal_value, p_start_state) - np.matmul(value, p_start_state)
    return evd