import abc
import numpy as np
import itertools


class BaseAgent:

    def __init__(self, gamma, n_bins, n_iterations, env):
        self.n_iterations = n_iterations
        # https://github.com/openai/gym/blob/master/gym/envs/classic_control/mountain_car.py
        self._state_bins = [
            # Cart Position
            self._discretize(lb=-2.4, ub=2.4, n_bins=n_bins),
            # Cart Velocity
            self._discretize(lb=-30.0, ub=30.0, n_bins=n_bins),
        ]

        self.n_states = n_bins ** len(self._state_bins)
        self.n_actions = 2
        self.q_table = np.ones((self.n_states, self.n_actions)) * 0.5
        self.c = np.zeros((self.n_states, self.n_actions))
        self.gamma = gamma

        self.state_to_idx = {}
        self.init_states()
        self.env = env

        self.test_scores = []

    @abc.abstractmethod
    def init_states(self):
        possible_states = list(
            itertools.product(list(self._state_bins[0]),
                              list(self._state_bins[1]))
        )

        for i, state in enumerate(possible_states):
            self.state_to_idx[state] = i

    @abc.abstractmethod
    def get_discrete_state(self, cts_state):
        cp = cts_state[0]
        cv = cts_state[1]

        cp_dis = self._find_nearest(self._state_bins[0], cp)
        cv_dis = self._find_nearest(self._state_bins[1], cv)

        state = (cp_dis, cv_dis)
        return self.state_to_idx[state]

    @staticmethod
    @abc.abstractmethod
    def _discretize(lb, ub, n_bins):
        return np.linspace(lb, ub, n_bins)

    @staticmethod
    @abc.abstractmethod
    def _find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]
