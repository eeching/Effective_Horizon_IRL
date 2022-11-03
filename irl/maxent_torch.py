from itertools import product

import numpy as np

import torch
from tqdm import tqdm
import pdb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def irl(feature_matrix, n_actions, discount, transition_probability,
        trajectories, epochs, learning_rate):
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
    alpha = torch.rand(d_states).to(device)
    # data to torch
    feature_matrix_torch = torch.from_numpy(feature_matrix.astype(np.float32)).to(device)
    trajectories_torch = torch.from_numpy(trajectories.astype(np.int64)).to(device)
    transition_probability_torch = torch.from_numpy(transition_probability.astype(np.float32)).to(device)
    n_states_torch, n_actions_torch, discount_torch = torch.tensor(n_states), torch.tensor(n_actions), torch.tensor(discount).to(device)

    pdb.set_trace()
    # Calculate the feature expectations \tilde{phi}.
    feature_expectations = find_feature_expectations(feature_matrix_torch,
                                                     trajectories_torch)
    # Gradient descent on alpha.
    for i in tqdm(range(epochs)):
        r = torch.matmul(feature_matrix_torch, alpha)
        expected_svf = find_expected_svf(n_states_torch, r, n_actions_torch, discount_torch,
                                         transition_probability_torch, trajectories_torch)
        grad = feature_expectations - torch.matmul(torch.transpose(feature_matrix_torch, 0, 1), expected_svf)

        alpha += learning_rate * grad

    reward = torch.matmul(feature_matrix_torch, alpha)
    v = optimal_value(n_states_torch, n_actions_torch, transition_probability_torch, reward, discount_torch)
    policy = find_policy_value_iteration(n_states_torch, n_actions_torch,
                                         transition_probability_torch, reward, discount_torch, stochastic=False)

    return torch.matmul(feature_matrix, alpha).view((n_states,)).numpy(), v.numpy(), policy.numpy()

# def find_svf(n_states, trajectories):
#     """
#     Find the state visitation frequency from trajectories.
#
#     n_states: Number of states. int.
#     trajectories: 3D array of state/action pairs. States are ints, actions
#         are ints. NumPy array with shape (T, L, 2) where T is the number of
#         trajectories and L is the trajectory length.
#     -> State visitation frequencies vector with shape (N,).
#     """
#
#     svf = np.zeros(n_states)
#
#     for trajectory in trajectories:
#         for state, _, _ in trajectory:
#             svf[state] += 1
#
#     svf /= trajectories.shape[0]
#
#     return svf

def find_feature_expectations(feature_matrix, trajectories):
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
    all_states = torch.flatten(trajectories[:, :, 1], start_dim=0)
    feature_expectations = torch.sum(torch.index_select(feature_matrix, 0, all_states), 0)/trajectories.shape[0]

    return feature_expectations

def find_expected_svf(n_states, r, n_actions, discount,
                      transition_probability, trajectories):
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

    policy = find_policy_value_iteration(n_states, n_actions,
                                         transition_probability, r, discount)

    start_state_count = torch.zeros(n_states, 1).to(device)
    init_states = torch.flatten(trajectories[:, 0, 0], start_dim=0)
    for s in init_states:
        start_state_count[s, 0] += 1

    p_start_state = start_state_count/n_trajectories
    expected_svf = p_start_state.expand(-1, trajectory_length)

    for t in range(1, trajectory_length):
        expected_svf[:, t] = 0
        for i, j, k in torch.cartesian_prod(torch.tensor(range(n_states)).to(device), torch.tensor(range(n_actions)).to(device), torch.tensor(range(n_states)).to(device)):
            expected_svf[k, t] += (expected_svf[i, t-1] *
                                  policy[i, j] * # Stochastic policy
                                  transition_probability[i, j, k])

    return torch.sum(expected_svf, 1)

# def softmax(x1, x2):
#     """
#     Soft-maximum calculation, from algorithm 9.2 in Ziebart's PhD thesis.
#
#     x1: float.
#     x2: float.
#     -> softmax(x1, x2)
#     """
#
#     max_x = max(x1, x2)
#     min_x = min(x1, x2)
#     return max_x + np.log(1 + np.exp(min_x - max_x))

def optimal_value(n_states, n_actions, transition_probabilities, reward,
                  discount, threshold=1e-2):
    """
    Find the optimal value function.

    n_states: Number of states. int.
    n_actions: Number of actions. int.
    transition_probabilities: Function taking (state, action, state) to
        transition probabilities.
    reward: Vector of rewards for each state.
    discount: MDP discount factor. float.
    threshold: Convergence threshold, default 1e-2. float.
    -> Array of values for each state
    """

    v = torch.zeros(n_states).to(device)

    diff = torch.tensor(float("inf")).to(device)
    threshold = torch.tensor(threshold).to(device)
    n_inf = torch.tensor(float("-inf")).to(device)
    while diff > threshold:
        diff = torch.tensor(0).to(device)
        for s in range(n_states):
            max_v = n_inf
            for a in range(n_actions):
                tp = transition_probabilities[s, a, :]
                max_v = torch.maximum(max_v, torch.matmul(tp, reward + discount*v))

            new_diff = torch.abs(v[s] - max_v)
            if new_diff > diff:
                diff = new_diff
            v[s] = max_v

    return v


def find_policy_value_iteration(n_states, n_actions, transition_probabilities, reward, discount,
                threshold=1e-2, v=None, stochastic=True):
    """
    Find the optimal policy.

    n_states: Number of states. int.
    n_actions: Number of actions. int.
    transition_probabilities: Function taking (state, action, state) to
        transition probabilities.
    reward: Vector of rewards for each state.
    discount: MDP discount factor. float.
    threshold: Convergence threshold, default 1e-2. float.
    v: Value function (if known). Default None.
    stochastic: Whether the policy should be stochastic. Default True.
    -> Action probabilities for each state or action int for each state
        (depending on stochasticity).
    """

    if v is None:
        v = optimal_value(n_states, n_actions, transition_probabilities, reward,
                          discount, threshold)

    if stochastic:
        # Get Q using equation 9.2 from Ziebart's thesis.
        Q = torch.zeros((n_states, n_actions)).to(device)
        for i in range(n_states):
            for j in range(n_actions):
                p = transition_probabilities[i, j, :]
                Q[i, j] = torch.matmul(p, reward + discount*v)

        Q -= torch.max(Q, 1).values.view(n_states, 1)
        Q = torch.exp(Q)/torch.sum(torch.exp(Q), 1).view(n_states, 1).expand(-1, n_actions)
        return Q

    def _policy(s):
        return torch.max(torch.tensor([sum(transition_probabilities[s, a, k] *
                                     (reward[k] + discount * v[k])
                                     for k in range(n_states)) for a in range(n_actions)]), 0).indices
    policy = torch.tensor([_policy(s) for s in range(n_states)])
    pdb.set_trace()
    return policy

# def find_policy(n_states, r, n_actions, discount,
#                            transition_probability):
#     """
#     Find a policy with linear value iteration. Based on the code accompanying
#     the Levine et al. GPIRL paper and on Ziebart's PhD thesis (algorithm 9.1).
#
#     n_states: Number of states N. int.
#     r: Reward. NumPy array with shape (N,).
#     n_actions: Number of actions A. int.
#     discount: Discount factor of the MDP. float.
#     transition_probability: NumPy array mapping (state_i, action, state_k) to
#         the probability of transitioning from state_i to state_k under action.
#         Shape (N, A, N).
#     -> NumPy array of states and the probability of taking each action in that
#         state, with shape (N, A).
#     """
#
#     # V = value_iteration.value(n_states, transition_probability, r, discount)
#
#     # NumPy's dot really dislikes using inf, so I'm making everything finite
#     # using nan_to_num.
#     V = np.nan_to_num(np.ones((n_states, 1)) * float("-inf"))
#
#     diff = np.ones((n_states,))
#     while (diff > 1e-4).all():  # Iterate until convergence.
#         new_V = r.copy()
#         for j in range(n_actions):
#             for i in range(n_states):
#                 new_V[i] = softmax(new_V[i], r[i] + discount*
#                     np.sum(transition_probability[i, j, k] * V[k]
#                            for k in range(n_states)))
#
#         # # This seems to diverge, so we z-score it (engineering hack).
#         new_V = (new_V - new_V.mean())/new_V.std()
#
#         diff = abs(V - new_V)
#         V = new_V
#
#     # We really want Q, not V, so grab that using equation 9.2 from the thesis.
#     Q = np.zeros((n_states, n_actions))
#     for i in range(n_states):
#         for j in range(n_actions):
#             p = np.array([transition_probability[i, j, k]
#                           for k in range(n_states)])
#             Q[i, j] = p.dot(r + discount*V)
#
#     # Softmax by row to interpret these values as probabilities.
#     Q -= Q.max(axis=1).reshape((n_states, 1))  # For numerical stability.
#     Q = np.exp(Q)/np.exp(Q).sum(axis=1).reshape((n_states, 1))
#     return Q
#
# def expected_value_difference(n_states, n_actions, transition_probability,
#     reward, discount, p_start_state, optimal_value, true_reward):
#     """
#     Calculate the expected value difference, which is a proxy to how good a
#     recovered reward function is.
#
#     n_states: Number of states. int.
#     n_actions: Number of actions. int.
#     transition_probability: NumPy array mapping (state_i, action, state_k) to
#         the probability of transitioning from state_i to state_k under action.
#         Shape (N, A, N).
#     reward: Reward vector mapping state int to reward. Shape (N,).
#     discount: Discount factor. float.
#     p_start_state: Probability vector with the ith component as the probability
#         that the ith state is the start state. Shape (N,).
#     optimal_value: Value vector for the ground reward with optimal policy.
#         The ith component is the value of the ith state. Shape (N,).
#     true_reward: True reward vector. Shape (N,).
#     -> Expected value difference. float.
#     """
#
#     policy = value_iteration.find_policy(n_states, n_actions,
#         transition_probability, reward, discount)
#     value = value_iteration.value(policy.argmax(axis=1), n_states,
#         transition_probability, true_reward, discount)
#
#     evd = optimal_value.dot(p_start_state) - value.dot(p_start_state)
#     return evd



