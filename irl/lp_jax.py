import random

import numpy as np
from cvxopt import matrix, solvers
import pdb
from jax import jit, random, vmap
import jax.numpy as jnp
from functools import partial

def irl(n_states, n_actions, transition_probability, policy, states, discount, Rmax,
        l1, ground_r=None, beta=0, inv_v=None):
    """
    Modified version of inverse RL as described in Ng & Russell, 2000.
    Given only limited expert demonstrations
    Find a reward function

    n_states: Number of states. int.
    n_actions: Number of actions. int.
    transition_probability: NumPy array mapping (state_i, action, state_k) to
        the probability of transitioning from state_i to state_k under action.
        Shape (N, A, N).
    policy: Vector mapping state ints to action ints. Shape (N,).
    discount: Discount factor. float.
    Rmax: Maximum reward. float.
    l1: l1 regularisation. float.
    -> Reward vector
    """

    A = set(range(n_actions))  # Set of actions to help manage reordering
    S_prime = list(set(range(n_states)) - set(states))

    
    transition_expert = transition_probability[np.arange(n_states), policy, :]
    transition_expert[S_prime,:] = 0
    F = np.matmul(np.repeat(transition_expert, n_actions, axis=0).reshape(n_states, n_actions, n_states)-transition_probability, np.linalg.inv(np.eye(n_states) - discount*transition_expert)) # n_states, n_actions, n_states
    F[S_prime, :, :] = 0

    # Minimise c . x.
    c = -np.hstack([np.zeros(n_states), np.ones(n_states), -l1*np.ones(n_states)]) # (0,0,0..., 1,1,1, ..., l1, l1, l1 ...) each repeat for |n_states| times
    zero_stack1 = np.zeros((n_states*(n_actions-1), n_states)) # size: |n_states| x |actions-1|, |n_states|

    T_stack = np.vstack([
        -F[s, a]
        for s in range(n_states) # n-states
        for a in A - {policy[s]}
    ])

    I_stack1 = np.vstack([
        np.eye(1, n_states, s) # vector of size (1, n_states), only 1 at s-th state
        for s in range(n_states) # n_states
        for a in A - {policy[s]}
    ]) # size is (|actions-1|x|states|, n_states)
    I_stack2 = np.eye(n_states) # size: (n_states, n_states)
    zero_stack2 = np.zeros((n_states, n_states)) # size: (n_states, n_states)

    D_left = np.vstack([T_stack, T_stack, -I_stack2, I_stack2])
    D_middle = np.vstack([I_stack1, zero_stack1, zero_stack2, zero_stack2])
    D_right = np.vstack([zero_stack1, zero_stack1, -I_stack2, -I_stack2])

    D = np.hstack([D_left, D_middle, D_right])
    
    if beta == 0:
        b = np.zeros((n_states*(n_actions-1)*2 + 2*n_states, 1))
    else:
        b_left = np.zeros((n_states*(n_actions-1), 1))
        b_middle = - np.ones((n_states, 1))*beta
        b_middle[S_prime] = 0
        b_middle = np.repeat(b_middle, n_actions-1, axis=0)
        b_right = np.zeros((2*n_states, 1))
        b = np.vstack([b_left, b_middle, b_right])
    bounds = np.array([(None, None)]*2*n_states + [(-Rmax, Rmax)]*n_states)
    
    # We still need to bound R. To do this, we just add
    # -I R <= Rmax 1
    # I R <= Rmax 1
    # So to D we need to add -I and I, and to b we need to add Rmax 1 and Rmax 1
    D_bounds = np.hstack([
        np.vstack([
            -np.eye(n_states),
            np.eye(n_states)]),
        np.vstack([
            np.zeros((n_states, n_states)),
            np.zeros((n_states, n_states))]),
        np.vstack([
            np.zeros((n_states, n_states)),
            np.zeros((n_states, n_states))])])
    b_bounds = np.vstack([Rmax*np.ones((n_states, 1))]*2)
    D = np.vstack((D, D_bounds))
    b = np.vstack((b, b_bounds))
    A_ub = matrix(D)
    b = matrix(b)
    c = matrix(c)
    results = solvers.lp(c, A_ub, b)
    
    return results

@partial(jit, static_argnames=['n_states', 'n_actions', 'feature_dimension'], )
def v_tensor(value, expert_transition, expert_off_transition, feature_dimension, n_states, n_actions):
    """
    Finds the v tensor used in large linear IRL.

    value: NumPy matrix for the value function. The (i, j)th component
        represents the value of the jth state under the ith basis function.
    transition_probability: NumPy array mapping (state_i, action, state_k) to
        the probability of transitioning from state_i to state_k under action.
        Shape (N, A, N).
    feature_dimension: Dimension of the feature matrix. int.
    n_states: Number of states sampled. int.
    n_actions: Number of actions. int.
    policy: NumPy array mapping state ints to action ints.
    -> v helper tensor.
    """
    # v = np.zeros((n_states, n_actions-1, feature_dimension)) # N, A-1, D
    v = jnp.zeros((n_states, n_actions-1, feature_dimension)) # M, A-1, D

    exp_on_policy = jnp.matmul(expert_transition, value.T) # M D
    exp_off_policy = jnp.matmul(expert_off_transition, value.T) # M A-1 D
    v = exp_on_policy.reshape(n_states, 1, feature_dimension) - exp_off_policy

    return v

def large_irl(value, feature_matrix, n_states,
              n_actions, policy, expert_transition, expert_off_transition, m_expert):
    """
    Find the reward in a large state space.

    value: NumPy matrix for the value function. The (i, j)th component
        represents the value of the jth state under the ith basis function.
    transition_probability: NumPy array mapping (state_i, action, state_k) to
        the probability of transitioning from state_i to state_k under action.
        Shape (N, A, N).
    feature_matrix: Matrix with the nth row representing the nth state. NumPy
        array with shape (N, D) where N is the number of states and D is the
        dimensionality of the state.
    n_states: Number of states sampled. int.
    n_actions: Number of actions. int.
    policy: NumPy array mapping state ints to action ints.
    -> Reward for each state in states.
    """

    D = feature_matrix.shape[1]

    # First, calculate v, which is just a helper tensor.
    v = v_tensor(value, expert_transition, expert_off_transition, D, n_states, n_actions) # M
    v = np.array([v[i] for i in m_expert])

    n_states = len(m_expert)

    # Now we can calculate c, G, h, A, and b.
    # x = [z y_i^+ y_i^- a], which is a [N (K-1)*N (K-1)*N D] vector.
    x_size = n_states + (n_actions-1)*n_states*2 + D

    # c is a big stack of ones and zeros; there's N ones and the rest is zero.
    c = -np.hstack([np.ones(n_states), np.zeros(x_size - n_states)])
    assert c.shape[0] == x_size

    # A is [0 I_j -I_j -v^T_{ij}] and j NOT EQUAL TO policy(i).
    # I believe this is accounted for by the structure of v.
    A = np.hstack([
        np.zeros((n_states*(n_actions-1), n_states)),
        np.eye(n_states*(n_actions-1)),
        -np.eye(n_states*(n_actions-1)),
        np.vstack([v[i, j].T for i in range(n_states)
                             for j in range(n_actions-1)])])
    assert A.shape[1] == x_size

    # b is just zeros!
    b = np.zeros(A.shape[0])

    # Break G up into the bottom row and other rows to construct it.
    bottom_row = np.vstack([
                    np.hstack([
                        np.ones((n_actions-1, 1)).dot(np.eye(1, n_states, l)),
                        np.hstack([-np.eye(n_actions-1) if i == l
                                   else np.zeros((n_actions-1, n_actions-1))
                         for i in range(n_states)]),
                        np.hstack([2*np.eye(n_actions-1) if i == l
                                   else np.zeros((n_actions-1, n_actions-1))
                         for i in range(n_states)]),
                        np.zeros((n_actions-1, D))])
                    for l in range(n_states)])
    assert bottom_row.shape[1] == x_size
    G = np.vstack([
            np.hstack([
                np.zeros((D, n_states)),
                np.zeros((D, n_states*(n_actions-1))),
                np.zeros((D, n_states*(n_actions-1))),
                np.eye(D)]),
            np.hstack([
                np.zeros((D, n_states)),
                np.zeros((D, n_states*(n_actions-1))),
                np.zeros((D, n_states*(n_actions-1))),
                -np.eye(D)]),
            np.hstack([
                np.zeros((n_states*(n_actions-1), n_states)),
                -np.eye(n_states*(n_actions-1)),
                np.zeros((n_states*(n_actions-1), n_states*(n_actions-1))),
                np.zeros((n_states*(n_actions-1), D))]),
            np.hstack([
                np.zeros((n_states*(n_actions-1), n_states)),
                np.zeros((n_states*(n_actions-1), n_states*(n_actions-1))),
                -np.eye(n_states*(n_actions-1)),
                np.zeros((n_states*(n_actions-1), D))]),
            bottom_row])
    assert G.shape[1] == x_size

    h = np.vstack([np.ones((D*2, 1)),
                   np.zeros((n_states*(n_actions-1)*2+bottom_row.shape[0], 1))])

    from cvxopt import matrix, solvers
    c = matrix(c)
    G = matrix(G)
    h = matrix(h)
    A = matrix(A)
    b = matrix(b)
    results = solvers.lp(c, G, h, A, b)
    alpha = np.asarray(results["x"][-D:], dtype=np.double)
    return np.dot(feature_matrix, -alpha)
