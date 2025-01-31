from irl.value_iteration_jax import optimal_value, find_policy
import pdb
import numpy as np
import numpy.random as rn


class Gridworld(object):
    """
    Customized Gridworld MDP.
    1/ Random Goal
    2/ Sparse reward: Goal = 1, 0 otherwise
    3/ Random Obstacles
    """

    def __init__(self, wind=0.1, discount=0.99, grid_size=10, demo=None, seed=None, V=True, T=None, reward_model=None, finite_horizon=None, objectworld=False):
        """
        wind: Chance of moving randomly. float.
        discount: MDP discount. float.
        -> Gridworld
        """
       
        self.actions = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (0, 0))
        self.n_actions = len(self.actions)
        self.reward_model = reward_model
        
        if reward_model == "hard":
            self.grid_size = 15
        elif reward_model in ["simple", "linear", "non_linear"]:
            self.grid_size = 10
        else:
            self.grid_size = grid_size
    
        self.n_states = self.grid_size**2
        self.wind = wind
        self.discount = discount
        self.max_steps = self.grid_size * 3
        self.seed = seed
        self.finite_horizon = finite_horizon
        self.rn = rn
        self.rn.seed(self.seed)


        if objectworld is False:
            if reward_model == "simple":
                if demo:
                    self.goal_pos = [2*self.grid_size+2, 7*self.grid_size+3, 4*self.grid_size+8, 8*self.grid_size+8]  
                else: 
                    self.goal_pos = self.rn.choice(range(self.grid_size ** 2), 4, replace=False)
            elif reward_model == "hard":
                if demo:
                    self.goal_pos = [2*self.grid_size+2, 9*self.grid_size+2, 6*self.grid_size+7, 9*self.grid_size+10, 12*self.grid_size+13, 1*self.grid_size+13]
                else:
                    self.goal_pos = self.rn.choice(range(self.grid_size ** 2), 6, replace=False)
            else:
                self.goal_pos = self.rn.randint(self.grid_size**2)

        if T is None:
            self.transition_probability = np.array(
            [[[self._transition_probability(i, j, k)
               for k in range(self.n_states)]
              for j in range(self.n_actions)]
             for i in range(self.n_states)])
        else:
            self.transition_probability = T
        
        self.opt_v = None
        
        if V and objectworld is False:
            # self.reward_array = np.array([self.reward(s) for s in range(self.n_states)])
            self.reward_array = self.get_reward_array()
            self.opt_v = optimal_value(self.n_states,
                                  self.n_actions,
                                  self.transition_probability,
                                  self.reward_array,
                                  self.discount,
                                  T=self.finite_horizon)

            print("finished computing V-value")
            self.policy, rep = find_policy(self.n_states, self.n_actions, self.transition_probability, self.reward_array, self.discount,
                        self.opt_v, stochastic=False, T=self.finite_horizon)

            if rep > 0:
                print(f"Gt policy ---, No uniquely optimal policy, equally optimal actions number: {rep}")
            print("finished computing the expert policy")

    def __str__(self):
        return "Gridworld({}, {}, {})".format(self.grid_size, self.wind,
                                              self.discount)

    def feature_vector(self, i, feature_map):
        """
        Get the feature vector associated with a state integer.

        i: State int.
        feature_map: Which feature map to use (default ident).
        -> Feature vector.
        """
        # Assume identity map.
        f = np.zeros(self.n_states)
        f[i] = 1
        return f

    def feature_matrix(self, feature_map="ident"):
        """
        Get the feature matrix for this gridworld.

        feature_map: Which feature map to use (default ident). 
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
        
        return np.abs(i[0] - k[0]) <= 1 and  np.abs(i[1] - k[1]) <= 1
         
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
        
        corner_spare = 4
        edge_spare = 6

        if not self.neighbouring((xi, yi), (xk, yk)):
            return 0.0
       
       # Is k the intended state to move to?
        if (xi + xj, yi + yj) == (xk, yk):
            return 1 - self.wind

        # if xi, yi is a corner 
        if (xi, yi) in {(0, 0), (self.grid_size-1, self.grid_size-1),
                        (0, self.grid_size-1), (self.grid_size-1, 0)}:
            # if we are moving off grid
            if not (0 <= xi + xj < self.grid_size and
                    0 <= yi + yj < self.grid_size):
                        # for diag action, only four actions are feasible, randomly to anyone
                        return 1/corner_spare
            # not moving off the grid
            else:
                # only move in 3 directions
                return self.wind/(corner_spare-1)
        # if it is an edge        
        elif xi in {0, self.grid_size-1} or yi in {0, self.grid_size-1}:
            # if we try to move off the grid, 6 direction as remaining
            if not (0 <= xi + xj < self.grid_size and
                    0 <= yi + yj < self.grid_size):
                return 1/edge_spare
            else:
                # we try not to move off, some 5 remaining actions
                return self.wind/(edge_spare-1)
        else:
            return self.wind/(self.n_actions-1)

    def get_optimal_policy(self):
        return self.policy
    
    def get_optimal_value(self):
        return self.opt_v
        
    def get_reward_array(self):
    
        if self.reward_model in ["simple", "hard"]:
            reward = np.zeros(self.n_states)
            reward[self.goal_pos] = 1
        
            return reward
    
        unit_dist = self.reward_model/np.sqrt(self.grid_size**2*2)        

        row = np.repeat(np.arange(self.grid_size), self.grid_size, axis=0).reshape(self.grid_size, self.grid_size)
        col = np.repeat(np.arange(self.grid_size), self.grid_size, axis=0).reshape(self.grid_size, self.grid_size).T
        reward = - np.sqrt(row**2+col**2)*unit_dist + self.reward_model
        reward = reward.flatten()
        reward[self.goal_pos] = self.reward_model*1.1
        return reward

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
        total_reward = np.sum(rewards, axis=1)

        # Return the average reward and standard deviation.
        return np.mean(total_reward), np.std(total_reward)

    def generate_all_trajectories(self, n_states, trajectory_length):

        trajectories = {}
        for i in range(n_states):
            print(f"{len(trajectories)} trajs")
            state_int = i
            trajectory = []
            for _ in range(trajectory_length):
                if state_int not in trajectories:
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
                    trajectory.append((state_int, action_int))
                    state_int = self.point_to_int((next_sx, next_sy))
                else:
                    trajectory += trajectories[state_int][:trajectory_length-len(trajectory)]
                    break
            trajectories[i] = trajectory
           
        return trajectories

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
        trajectories = []
        state_occupancy_list = []

        # edges_list = np.hstack([np.arange(self.grid_size), np.arange(self.n_states-self.grid_size, self.n_states), np.arange(self.grid_size, self.n_states-self.grid_size, self.grid_size), np.arange(self.grid_size*2-1, self.n_states-self.grid_size, self.grid_size)])
        # init_states = rn.choice(edges_list, len(edges_list), replace=False)

        init_states = rn.choice(np.arange(n_states), 100, replace=False)
        print("init states", init_states[:10])

        # collect at most 80 trajectories
        for i in range(100):
            print(f"{len(trajectories)} trajs")
            state_int = init_states[i]
            trajectory = []
            state_coverage = np.zeros(n_states)
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
                trajectory.append((state_int, action_int))
                state_coverage[state_int] = 1
                state_int = self.point_to_int((next_sx, next_sy))

            num_covered = np.sum(state_coverage)
            state_occupancy_list.append(num_covered)
            trajectories.append(trajectory)
           
        return trajectories, state_occupancy_list

    def generate_expert_demonstrations(self, m_expert, cross_validate_ratio=None):
        """
        Generate the expert demonstrations of n states
        following the given policy.
        policy: Map from state integers to action integers.
        random_start: Whether to start randomly (default False). bool.
        -> [[(state int, action int, reward float)]]
        """
        
        rn.seed(self.seed)
        states_itr = iter(rn.choice(range(self.n_states), self.n_states, replace=False))

        expert_demo = []
        # state_int = 90

        if cross_validate_ratio is None:
            m_training = m_expert
            m_validate = 0
        else:
            m_training = int(m_expert*cross_validate_ratio)
            m_validate = m_expert - m_training


        while len(expert_demo) < m_training:
            state_int = next(states_itr)
            while state_int not in expert_demo and len(expert_demo) < m_training:
                expert_demo.append(state_int)
                action = self.actions[self.policy[state_int]]
                sx, sy = self.int_to_point(state_int)
                if (0 <= sx + action[0] < self.grid_size and
                        0 <= sy + action[1] < self.grid_size):
                    next_sx = sx + action[0]
                    next_sy = sy + action[1]
                else:
                    next_sx = sx
                    next_sy = sy
                state_int = self.point_to_int((next_sx, next_sy))

        val_start_idx = len(expert_demo)

        while len(expert_demo) < val_start_idx + m_validate and len(expert_demo) <= self.n_states:
            state_int = next(states_itr)
            while state_int not in expert_demo:
                expert_demo.append(state_int)
                action = self.actions[self.policy[state_int]]
                sx, sy = self.int_to_point(state_int)
                if (0 <= sx + action[0] < self.grid_size and
                        0 <= sy + action[1] < self.grid_size):
                    next_sx = sx + action[0]
                    next_sy = sy + action[1]
                else:
                    next_sx = sx
                    next_sy = sy
                state_int = self.point_to_int((next_sx, next_sy))

        if cross_validate_ratio is None:
            return np.sort(expert_demo[:m_expert]).reshape(m_expert)
        else:
            
            training = np.sort(expert_demo[:m_training]).reshape(m_training)
            validate = np.sort(expert_demo[len(expert_demo)-m_validate:]).reshape(m_validate)
            expert = np.sort(expert_demo[:m_training] + expert_demo[len(expert_demo)-m_validate:]).reshape(m_expert)
            return expert, training, validate

    def evaluate_learnt_reward(self, reward, discount, n_states=None, n_actions=None, transition_prob=None):

        if n_states is None:
            n_states, n_actions, transition_prob  = self.n_states, self.n_actions, self.transition_probability

        value = optimal_value(n_states,
                              n_actions,
                              transition_prob,
                              reward,
                              discount)

        policy, rep = find_policy(n_states, n_actions, transition_prob, reward, discount, threshold=1e-2, v=value, stochastic=False)

        return value, policy








