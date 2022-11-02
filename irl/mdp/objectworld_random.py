"""
Implements the objectworld MDP described in Levine et al. 2011.

Matthew Alger, 2015
matthew.alger@anu.edu.au
"""

import math
from itertools import product

import numpy as np
import numpy.random as rn
from irl.value_iteration import optimal_value, find_policy
from .gridworld_random import GridworldRandom

class OWObject(object):
    """
    Object in objectworld.
    """

    def __init__(self, inner_colour, outer_colour):
        """
        inner_colour: Inner colour of object. int.
        outer_colour: Outer colour of object. int.
        -> OWObject
        """

        self.inner_colour = inner_colour
        self.outer_colour = outer_colour

    def __str__(self):
        """
        A string representation of this object.

        -> __str__
        """

        return "<OWObject (In: {}) (Out: {})>".format(self.inner_colour,
                                                      self.outer_colour)

class ObjectworldRandom(GridworldRandom):
    """
    Objectworld MDP.
    """

    def __init__(self, grid_size, n_objects, n_colours, wind, discount, V=False, goal_pos=None, seed=None):
        """
        grid_size: Grid size. int.
        n_objects: Number of objects in the world. int.
        n_colours: Number of colours to colour objects with. int.
        wind: Chance of moving randomly. float.
        discount: MDP discount. float.
        -> Objectworld
        """

        super().__init__(grid_size, wind, discount, goal_pos=goal_pos, seed=seed)

        self.n_objects = n_objects
        self.n_colours = n_colours

        # Generate objects.
        self.objects = {}
        for _ in range(self.n_objects):
            obj = OWObject(rn.randint(self.n_colours),
                           rn.randint(self.n_colours))

            while True:
                x = rn.randint(self.grid_size)
                y = rn.randint(self.grid_size)

                if (x, y) not in self.objects:
                    break

            self.objects[x, y] = obj

        if V:
            self.opt_v = optimal_value(self.n_states,
                                       self.n_actions,
                                       self.transition_probability,
                                       [self.reward(s) for s in range(self.n_states)],
                                       self.discount)

            print("finished computing V-value")
            self.policy = find_policy(self.n_states, self.n_actions, self.transition_probability,
                                      [self.reward(s) for s in range(self.n_states)], self.discount,
                                      threshold=1e-2, v=self.opt_v, stochastic=False)
            print("finished computing the expert policy")

    def feature_vector(self, i, discrete=True):
        """
        Get the feature vector associated with a state integer.

        i: State int.
        discrete: Whether the feature vectors should be discrete (default True).
            bool.
        -> Feature vector.
        """

        sx, sy = self.int_to_point(i)

        nearest_inner = {}  # colour: distance
        nearest_outer = {}  # colour: distance

        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if (x, y) in self.objects:
                    dist = math.hypot((x - sx), (y - sy))
                    obj = self.objects[x, y]
                    if obj.inner_colour in nearest_inner:
                        if dist < nearest_inner[obj.inner_colour]:
                            nearest_inner[obj.inner_colour] = dist
                    else:
                        nearest_inner[obj.inner_colour] = dist
                    if obj.outer_colour in nearest_outer:
                        if dist < nearest_outer[obj.outer_colour]:
                            nearest_outer[obj.outer_colour] = dist
                    else:
                        nearest_outer[obj.outer_colour] = dist

        # Need to ensure that all colours are represented.
        for c in range(self.n_colours):
            if c not in nearest_inner:
                nearest_inner[c] = 0
            if c not in nearest_outer:
                nearest_outer[c] = 0

        if discrete:
            state = np.zeros((2*self.n_colours*self.grid_size,))
            i = 0
            for c in range(self.n_colours):
                for d in range(1, self.grid_size+1):
                    if nearest_inner[c] < d:
                        state[i] = 1
                    i += 1
                    if nearest_outer[c] < d:
                        state[i] = 1
                    i += 1
            assert i == 2*self.n_colours*self.grid_size
            assert (state >= 0).all()
        else:
            # Continuous features.
            state = np.zeros((2*self.n_colours))
            i = 0
            for c in range(self.n_colours):
                state[i] = nearest_inner[c]
                i += 1
                state[i] = nearest_outer[c]
                i += 1

        return state

    def feature_matrix(self, discrete=True):
        """
        Get the feature matrix for this objectworld.

        discrete: Whether the feature vectors should be discrete (default True).
            bool.
        -> NumPy array with shape (n_states, n_states).
        """

        return np.array([self.feature_vector(i, discrete)
                         for i in range(self.n_states)])

    def reward(self, state_int):
        """
        Get the reward for a state int.

        state_int: State int.
        -> reward float
        """

        x, y = self.int_to_point(state_int)

        near_c0 = False
        near_c1 = False
        for (dx, dy) in product(range(-3, 4), range(-3, 4)):
            if 0 <= x + dx < self.grid_size and 0 <= y + dy < self.grid_size:
                if (abs(dx) + abs(dy) <= 3 and
                        (x+dx, y+dy) in self.objects and
                        self.objects[x+dx, y+dy].outer_colour == 0):
                    near_c0 = True
                if (abs(dx) + abs(dy) <= 2 and
                        (x+dx, y+dy) in self.objects and
                        self.objects[x+dx, y+dy].outer_colour == 1):
                    near_c1 = True

        if near_c0 and near_c1:
            return 1
        if near_c0:
            return -1
        return 0

