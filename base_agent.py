import abc
import numpy as np
import itertools


class BaseAgent:

    def __init__(self, gamma, n_bins, n_iterations, env):
        self.n_iterations = n_iterations
        # https://github.com/openai/gym/blob/master/gym/envs/classic_control/mountain_car.py
        self._state_bins = [
            # Cart Position
            self._discretize(lb=-1.2, ub=0.6, n_bins=n_bins),
            # Cart Velocity
            self._discretize(lb=-0.07, ub=0.07, n_bins=n_bins),
        ]

        self.n_states = n_bins ** len(self._state_bins)
        # self.n_states = 10 * 10
        self.n_actions = env.action_space.n
        self.q_table = np.random.uniform(low=-1, high=1, size = (self.n_states, self.n_actions))
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
