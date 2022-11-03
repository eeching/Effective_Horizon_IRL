"""
Implements the gridworld MDP.

Matthew Alger, 2015
matthew.alger@anu.edu.au
"""

import numpy as np
import numpy.random as rn
from irl.value_iteration import optimal_value, find_policy
import pdb
import random

class GridworldRandom(object):
    """
    Customized Gridworld MDP.
    1/ Random Goal
    2/ Sparse reward: Goal = 1, 0 otherwise
    3/ Random Obstacles
    """

    def __init__(self, grid_size, wind, discount, goal_pos=None, seed=None, V=False):
        """
        grid_size: Grid size. int.
        wind: Chance of moving randomly. float.
        discount: MDP discount. float.
        -> Gridworld
        """

        self.actions = ((1, 0), (0, 1), (-1, 0), (0, -1), (0, 0))
        self.n_actions = len(self.actions)
        self.n_states = grid_size**2
        self.grid_size = grid_size
        self.wind = wind
        self.discount = discount
        self.max_steps = grid_size * 3
        if goal_pos is None:
            if seed is not None:
                np.random.seed(seed)
            self.goal_pos = np.random.randint(grid_size**2)
        else:
            self.goal_pos = goal_pos

        # Preconstruct the transition probability array.
        self.transition_probability = np.array(
            [[[self._transition_probability(i, j, k)
               for k in range(self.n_states)]
              for j in range(self.n_actions)]
             for i in range(self.n_states)])
        self.opt_v = None
        if V:
            self.opt_v = optimal_value(self.n_states,
                                  self.n_actions,
                                  self.transition_probability,
                                  [self.reward(s) for s in range(self.n_states)],
                                  self.discount)

            print("finished computing V-value")
            self.policy = find_policy(self.n_states, self.n_actions, self.transition_probability, [self.reward(s) for s in range(self.n_states)], self.discount,
                        threshold=1e-2, v=self.opt_v, stochastic=False)
            print("finished computing the expert policy")

    def __str__(self):
        return "Gridworld({}, {}, {})".format(self.grid_size, self.wind,
                                              self.discount)

    def feature_vector(self, i, feature_map="ident"):
        """
        Get the feature vector associated with a state integer.

        i: State int.
        feature_map: Which feature map to use (default ident). String in {ident,
            coord, proxi}.
        -> Feature vector.
        """

        if feature_map == "coord":
            f = np.zeros(self.grid_size)
            x, y = i % self.grid_size, i // self.grid_size
            f[x] += 1
            f[y] += 1
            return f
        if feature_map == "proxi":
            f = np.zeros(self.n_states)
            x, y = i % self.grid_size, i // self.grid_size
            for b in range(self.grid_size):
                for a in range(self.grid_size):
                    dist = abs(x - a) + abs(y - b)
                    f[self.point_to_int((a, b))] = dist
            return f
        # Assume identity map.
        f = np.zeros(self.n_states)
        f[i] = 1
        return f

    def feature_matrix(self, feature_map="ident"):
        """
        Get the feature matrix for this gridworld.

        feature_map: Which feature map to use (default ident). String in {ident,
            coord, proxi}.
        -> NumPy array with shape (n_states, d_states).
        """

        features = []
        for n in range(self.n_states):
            f = self.feature_vector(n, feature_map)
            features.append(f)
        return np.array(features)

    def int_to_point(self, i):
        """
        Convert a state int into the corresponding coordinate.

        i: State int.
        -> (x, y) int tuple.
        """

        return (i % self.grid_size, i // self.grid_size)

    def point_to_int(self, p):
        """
        Convert a coordinate into the corresponding state int.

        p: (x, y) tuple.
        -> State int.
        """

        return p[0] + p[1]*self.grid_size

    def neighbouring(self, i, k):
        """
        Get whether two points neighbour each other. Also returns true if they
        are the same point.

        i: (x, y) int tuple.
        k: (x, y) int tuple.
        -> bool.
        """

        return abs(i[0] - k[0]) + abs(i[1] - k[1]) <= 1

    def _transition_probability(self, i, j, k):
        """
        Get the probability of transitioning from state i to state k given
        action j.

        i: State int.
        j: Action int.
        k: State int.
        -> p(s_k | s_i, a_j)
        """

        xi, yi = self.int_to_point(i)
        xj, yj = self.actions[j]
        xk, yk = self.int_to_point(k)

        if not self.neighbouring((xi, yi), (xk, yk)):
            return 0.0

        # Is k the intended state to move to?
        if (xi + xj, yi + yj) == (xk, yk):
            return 1 - self.wind + self.wind/self.n_actions

        # If these are not the same point, then we can move there by wind.
        if (xi, yi) != (xk, yk):
            return self.wind/self.n_actions

        # If these are the same point, we can only move here by either moving
        # off the grid or being blown off the grid. Are we on a corner or not?
        if (xi, yi) in {(0, 0), (self.grid_size-1, self.grid_size-1),
                        (0, self.grid_size-1), (self.grid_size-1, 0)}:
            # Corner.
            # Can move off the edge in two directions.
            # Did we intend to move off the grid?
            if not (0 <= xi + xj < self.grid_size and
                    0 <= yi + yj < self.grid_size):
                # We intended to move off the grid, so we have the regular
                # success chance of staying here plus an extra chance of blowing
                # onto the *other* off-grid square.
                return 1 - self.wind + 2*self.wind/self.n_actions
            else:
                # We can blow off the grid in either direction only by wind.
                return 2*self.wind/self.n_actions
        else:
            # Not a corner. Is it an edge?
            if (xi not in {0, self.grid_size-1} and
                yi not in {0, self.grid_size-1}):
                # Not an edge.
                return 0.0

            # Edge.
            # Can only move off the edge in one direction.
            # Did we intend to move off the grid?
            if not (0 <= xi + xj < self.grid_size and
                    0 <= yi + yj < self.grid_size):
                # We intended to move off the grid, so we have the regular
                # success chance of staying here.
                return 1 - self.wind + self.wind/self.n_actions
            else:
                # We can blow off the grid only by wind.
                return self.wind/self.n_actions

    def reward(self, state_int):
        """
        Reward for being in state state_int.

        state_int: State integer. int.
        -> Reward.
        """

        if state_int == self.goal_pos:
            return 1
        return 0

    def average_reward(self, n_trajectories, trajectory_length, policy):
        """
        Calculate the average total reward obtained by following a given policy
        over n_paths paths.

        policy: Map from state integers to action integers.
        n_trajectories: Number of trajectories. int.
        trajectory_length: Length of an episode. int.
        -> Average reward, standard deviation.
        """

        trajectories = self.generate_trajectories(n_trajectories,
                                             trajectory_length, policy)
        rewards = [[r for _, _, r in trajectory] for trajectory in trajectories]
        rewards = np.array(rewards)

        # Add up all the rewards to find the total reward.
        total_reward = rewards.sum(axis=1)

        # Return the average reward and standard deviation.
        return total_reward.mean(), total_reward.std()

    def generate_trajectories(self, n_states, trajectory_length):
        """
        Generate n_trajectories trajectories with length trajectory_length,
        following the given policy.

        n_trajectories: Number of trajectories. int.
        trajectory_length: Length of an episode. int.
        policy: Map from state integers to action integers.
        random_start: Whether to start randomly (default False). bool.
        -> [[(state int, action int, reward float)]]
        """

        expert_m_itr = iter([int(x*n_states)for x in [0.08, 0.1, 0.2, 0.25, 0.4, 0.5, 0.6, 0.75, 0.8, 1]])
        curr_expert_m = next(expert_m_itr)
        uncovered_states = set(np.arange(n_states))
        state_occupancy_list = []
        trajectories = []
        cached_traj_idx = {}
        cached_states_sets = {}

        num_covered = n_states - len(uncovered_states)
        while num_covered < n_states:
            print(f"{len(trajectories)} trajs and {num_covered} states")
            state_int = random.sample(uncovered_states, 1)[0]
            trajectory = []
            for _ in range(trajectory_length):

                action = self.actions[self.policy[state_int]]
                sx, sy = self.int_to_point(state_int)
                if (0 <= sx + action[0] < self.grid_size and
                        0 <= sy + action[1] < self.grid_size):
                    next_sx = sx + action[0]
                    next_sy = sy + action[1]
                else:
                    next_sx = sx
                    next_sy = sy

                action_int = self.actions.index(action)
                if state_int in uncovered_states:
                    uncovered_states.remove(state_int)
                trajectory.append((state_int, action_int))
                state_int = self.point_to_int((next_sx, next_sy))
            num_covered = n_states - len(uncovered_states)
            state_occupancy_list.append(num_covered)
            trajectories.append(trajectory)
            if num_covered >= curr_expert_m:
                cached_traj_idx[curr_expert_m] = len(trajectories) - 1
                cached_states_sets[curr_expert_m] = set(np.arange(n_states)) - uncovered_states
                try:
                    curr_expert_m = next(expert_m_itr)
                except Exception:
                    pass

        return np.array(trajectories), cached_traj_idx, cached_states_sets, state_occupancy_list

    def generate_expert_demonstrations(self, m_expert, cross_validate=None):
        """
        Generate the expert demonstrations of n states
        following the given policy.
        policy: Map from state integers to action integers.
        random_start: Whether to start randomly (default False). bool.
        -> [[(state int, action int, reward float)]]
        """
        m_states = rn.choice(range(self.n_states), m_expert, replace=False)
        if cross_validate is None:
            return np.sort(m_states).reshape(m_expert)
        else:
            k_training = int(m_expert*cross_validate)
            training = np.sort(m_states[: k_training]).reshape(k_training)
            validate = np.sort(m_states[k_training:]).reshape(m_expert-k_training)
            return m_states, training, validate

        # if policy is None:
        #     policy = self.policy
        # actions = np.array([policy[s] for s in states]).reshape(m_expert, 1)
        # demo = np.concatenate((states, actions), axis=1)
        # partial_T = np.array(
        #     [[self._transition_probability(i, policy[i], k) for k in range(self.n_states)]
        #      for i in range(m_expert)])
        # return demo, partial_T, states.reshape(m_expert)

    def evaluate_expert_policy(self, n_trajectory=10, random_action=False, random_start=True):

        result = []

        for i in range(n_trajectory):
            steps = 0

            if random_start:
                sx, sy = rn.randint(self.grid_size), rn.randint(self.grid_size)
                print(f"start pos {sx}, {sy}")
            else:
                sx, sy = 0, 0

            traj = []
            while steps < self.max_steps:
                if random_action is True and rn.random() < self.wind:
                    action = self.actions[rn.randint(0, 4)]
                else:
                    # pdb.set_trace()
                    action = self.actions[self.policy[self.point_to_int((sx, sy))]]

                if (0 <= sx + action[0] < self.grid_size and
                        0 <= sy + action[1] < self.grid_size):
                    next_sx = sx + action[0]
                    next_sy = sy + action[1]
                else:
                    next_sx = sx
                    next_sy = sy
                traj.append([(sx, sy), action, (next_sx, next_sy)])
                next_state_int = self.point_to_int((next_sx, next_sy))
                steps += 1
                sx = next_sx
                sy = next_sy

                if next_state_int == self.goal_pos:
                    reward = self.discount ** steps
                    result.append(reward)
                    # print(traj)
                    # pdb.set_trace()
                    break
                elif steps == self.max_steps:
                    result.append(0)
                    # print(traj)
                    # pdb.set_trace()
                    break
        return result

    def evaluate_learnt_reward(self, reward, discount, n_states=None, n_actions=None, transition_prob=None):

        if n_states is None:
            n_states, n_actions, transition_prob  = self.n_states, self.n_actions, self.transition_probability

        value = optimal_value(n_states,
                              n_actions,
                              transition_prob,
                              reward,
                              discount)

        policy = find_policy(n_states, n_actions, transition_prob, reward, discount, threshold=1e-2, v=value, stochastic=False)

        return value, policy







