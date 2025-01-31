import jax.numpy as np
import pdb
from jax import jit, random, vmap
from functools import partial
import jax.lax as lax
import numpy
import matplotlib.pyplot as plt

@partial(jit, static_argnames=['n_states', 'discount', 'threshold', 'T'],)
def value(policy, n_states, transition_probability, reward, discount,
                    threshold=1e-4, T=None):
    """
    Find the value function associated with a policy.

    policy: List of action ints for each state.
    n_states: Number of states. int.
    transition_probability: Function taking (state, action, state) to
        transition probabilities.
    reward: Vector of rewards for each state.
    discount: MDP discount factor. float.
    threshold: Convergence threshold, default 1e-2. float.
    -> Array of values for each state
    """
    v = np.zeros(n_states)
    if T is None:
        diff = np.inf
        transition_expert = transition_probability[np.arange(n_states), policy, :]

        inv_v = np.matmul(np.linalg.inv(np.eye(n_states) - discount* transition_expert), reward)
        
        def cond(arg):
            transition_expert, reward, discount, v, diff, threshold = arg
            return diff > threshold
        
        def body(arg):
            transition_expert, rewards, discount, v, diff, threshold = arg
            updated_v = np.matmul(transition_expert, reward + discount*v)
            diff = np.max(np.abs(v - updated_v))
            v = updated_v
            return (transition_expert, reward, discount, v, diff, threshold)

        transition_expert, reward, discount, v, diff, threshold = lax.while_loop(
                    cond,
                    body,
                    (transition_expert, reward, discount, v, diff, threshold))
    else:
        t = 0
        transition_expert = transition_probability[np.arange(n_states), policy, :]

        inv_v = np.matmul(np.linalg.inv(np.eye(n_states) - discount* transition_expert), reward)
        
        def cond(arg):
            transition_expert, reward, v, t, T = arg
            return t < T
        
        def body(arg):
            transition_expert, rewards, v, t, T = arg
            updated_v = np.matmul(transition_expert, reward + v)
            t += 1
            v = updated_v
            return (transition_expert, reward, v, t, T)

        transition_expert, reward, v, t, T = lax.while_loop(
                    cond,
                    body,
                    (transition_expert, reward, v, t, T))

    return v, inv_v

@partial(jit, static_argnames=['n_states', 'discount', 'threshold'],)
def value_matrix(n_states, transition_expert, feature_matrix, discount, threshold=1e-4):
    
    vec_value_i = vmap(value_i,in_axes=(None, None, 1, None, None))
    v = vec_value_i(transition_expert, n_states, feature_matrix, discount, threshold)
    return v

@partial(jit, static_argnames=['n_states', 'discount', 'threshold'],)
def value_i(transition_expert, n_states, feature_i, discount,
                    threshold=1e-4):
   
    v = np.zeros(n_states)
    diff = np.inf
        
    def cond(arg):
        transition_expert, feature_i, discount, v, diff, threshold = arg
        return diff > threshold
        
    def body(arg):
        transition_expert, feature_i, discount, v, diff, threshold = arg
        updated_v = np.matmul(transition_expert, feature_i + discount*v)
        diff = np.max(np.abs(v - updated_v))
        v = updated_v
        return (transition_expert, feature_i, discount, v, diff, threshold)

    transition_expert, feature_i, discount, v, diff, threshold = lax.while_loop(
                    cond,
                    body,
                    (transition_expert, feature_i, discount, v, diff, threshold))
    return v

@partial(jit, static_argnames=['n_states', 'n_actions', 'discount', 'threshold', 'T'],)
def optimal_value(n_states, n_actions, transition_probability, reward,
                  discount, threshold=1e-2, T=None):
    """
    Find the optimal value function.

    n_states: Number of states. int.
    n_actions: Number of actions. int.
    transition_probability: Function taking (state, action, state) to
        transition probabilities.
    reward: Vector of rewards for each state.
    discount: MDP discount factor. float.
    threshold: Convergence threshold, default 1e-2. float.
    -> Array of values for each state
    """

    v = np.zeros(n_states)

    if T is None: # infinite horizon

        diff = np.inf

        def cond(arg):
            transition_probability, reward, discount, v, diff, threshold = arg
            return diff > threshold
        
        def body(arg):
            transition_probability, rewards, discount, v, diff, threshold = arg
            max_v = np.max(np.matmul(transition_probability, reward + discount*v), axis=1)
            diff = np.max(np.abs(v - max_v))
            v = max_v
            return (transition_probability, reward, discount, v, diff, threshold)

        transition_probability, reward, discount, v, diff, threshold = lax.while_loop(
                    cond,
                    body,
                    (transition_probability, reward, discount, v, diff, threshold))
    else:

        t = 0

        def cond(arg):
            transition_probability, reward, v, t, T = arg
            return t < T
        
        def body(arg):
            transition_probability, rewards, v, t, T = arg
            max_v = np.max(np.matmul(transition_probability, reward + v), axis=1)
            v = max_v
            t += 1
            return (transition_probability, reward, v, t, T)

        transition_probability, reward, v, t, T = lax.while_loop(
                    cond,
                    body,
                    (transition_probability, reward, v, t, T))


    return v

def find_policy(n_states, n_actions, transition_probability, reward, discount, v,
                threshold=1e-2, stochastic=True, T=None):
        
    if T is not None:
        discount = 1

    if stochastic:
        Q = np.matmul(transition_probability, reward + discount*v)
        Q -= Q.max(axis=1).reshape((n_states, 1))  # For numerical stability.
        Q = np.exp(Q)/np.exp(Q).sum(axis=1).reshape((n_states, 1))
        return Q

    v_update = np.matmul(transition_probability, reward + discount * v)
    policy = np.argmax(v_update, axis=1)
    # check if the policy is unique
    v_opt = -np.sort(-v_update, axis=1)
    rep = np.sum(v_opt[:, 0] == v_opt[:,1])
    
    return policy, rep

@partial(jit, static_argnames=['n_states', 'n_actions', 'discount', 'threshold', 'T'],)
def optimal_value_ra(n_states, n_actions, transition_probability, reward,
                  discount, threshold=1e-2, T=None):
    """
    Find the optimal value function.

    n_states: Number of states. int.
    n_actions: Number of actions. int.
    transition_probability: Function taking (state, action, state) to
        transition probabilities.
    reward: Vector of rewards for each state.
    discount: MDP discount factor. float.
    threshold: Convergence threshold, default 1e-2. float.
    -> Array of values for each state
    """

    v = np.zeros(n_states)

    if T is None: # infinite horizon

        diff = np.inf

        def cond(arg):
            transition_probability, reward, discount, v, diff, threshold = arg
            return diff > threshold
        
        def body(arg):
            transition_probability, rewards, discount, v, diff, threshold = arg
            max_v = np.max(np.add(reward, np.matmul(transition_probability, discount*v)), axis=1)
            diff = np.max(np.abs(v - max_v))
            v = max_v
            return (transition_probability, reward, discount, v, diff, threshold)

        transition_probability, reward, discount, v, diff, threshold = lax.while_loop(
                    cond,
                    body,
                    (transition_probability, reward, discount, v, diff, threshold))
    else:

        t = 0

        def cond(arg):
            transition_probability, reward, v, t, T = arg
            return t < T
        
        def body(arg):
            transition_probability, rewards, v, t, T = arg
            max_v = np.max(np.matmul(transition_probability, reward + v), axis=1)
            v = max_v
            t += 1
            return (transition_probability, reward, v, t, T)

        transition_probability, reward, v, t, T = lax.while_loop(
                    cond,
                    body,
                    (transition_probability, reward, v, t, T))


    return v

def find_policy_ra(n_states, n_actions, transition_probability, reward, discount, v,
                threshold=1e-2, stochastic=True, T=None):
        
    if T is not None:
        discount = 1

    if stochastic:
        Q = np.add(reward, np.matmul(transition_probability, discount*v))
        Q -= Q.max(axis=1).reshape((n_states, 1))  # For numerical stability.
        Q = np.exp(Q)/np.exp(Q).sum(axis=1).reshape((n_states, 1))
        return Q

    v_update = np.add(reward, np.matmul(transition_probability, discount * v))
    policy = np.argmax(v_update, axis=1)
    # check if the policy is unique
    v_opt = -np.sort(-v_update, axis=1)
    rep = np.sum(v_opt[:, 0] == v_opt[:,1])
    
    return policy, rep