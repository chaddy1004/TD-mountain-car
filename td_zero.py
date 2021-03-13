import numpy as np

import itertools

import random
from collections import namedtuple

import seaborn as sns

from base_agent import BaseAgent
import pandas as pd

random.seed(a=19971124)
np.random.seed(seed=19940513)
sns.set_theme(style="darkgrid")


class TDZeroAgent(BaseAgent):

    def __init__(self, gamma, n_bins, n_iterations, env):
        super().__init__(gamma, n_bins, n_iterations, env)
        self.epsilon = 0.8
        self.MIN_EPSILON = 0.1

        self.delta_epsilon = (self.epsilon - self.MIN_EPSILON) / int(self.n_iterations * 0.8)
        self.lr = 0.2

    def policy(self, state):
        # e-greedy policy
        decision = np.random.rand()
        # print(decision, self.epsilon)
        if decision < self.epsilon:
            action = np.random.choice(self.n_actions)
        else:
            q = self.q_table[state]
            action = np.argmax(q)
        return action

    def sarsa(self):
        for ep in range(self.n_iterations):
            s_curr = self.env.reset()
            # self.epsilon -= self.delta_epsilon
            a_curr = self.policy(self.get_discrete_state(s_curr))
            done = False
            score = 0
            episode_scores = []
            if ep < (self.n_iterations * 0.8):
                self.epsilon -= self.delta_epsilon
            while not done:
                s_next, r, done, _ = self.env.step(a_curr)

                score += r

                s_curr_discrete = self.get_discrete_state(s_curr)
                s_next_discrete = self.get_discrete_state(s_next)
                a_next = self.policy(s_next_discrete)

                q_curr = self.q_table[s_curr_discrete, a_curr]
                done_mask = int(done)
                # obtaining the target q value from the action sampled from current policy
                q_next = self.q_table[s_next_discrete][a_curr]
                self.q_table[s_curr_discrete, a_curr] = q_curr + self.lr * (
                        r + (1 - done_mask) * (self.gamma * q_next) - q_curr)

                s_curr = s_next
                a_curr = a_next
                episode_scores.append(score)

            self.test_scores.append(score)
            if (ep + 1) % 500 == 0:
                print(f"Iteration {ep}: score = {score}")

    def q_learning(self):
        for ep in range(self.n_iterations):
            s_curr = self.env.reset()
            a_curr = self.policy(self.get_discrete_state(s_curr))
            done = False
            score = 0
            if ep < (self.n_iterations * 0.8):
                self.epsilon -= self.delta_epsilon
            while not done:
                s_next, r, done, _ = self.env.step(a_curr)

                score += r

                s_curr_discrete = self.get_discrete_state(s_curr)
                s_next_discrete = self.get_discrete_state(s_next)
                a_next = self.policy(s_next_discrete)

                q_curr = self.q_table[s_curr_discrete, a_curr]
                done_mask = int(done)
                # obtaining the target q value from the action sampled from current policy
                q_next = np.max(self.q_table[s_next_discrete])
                self.q_table[s_curr_discrete, a_curr] = q_curr + self.lr * (
                        r + (1 - done_mask) * (self.gamma * q_next) - q_curr)
                s_curr = s_next
                a_curr = a_next

            self.test_scores.append(score)
            if (ep + 1) % 500 == 0:
                print(f"Iteration {ep}: score = {score}")

    def expected_sarsa(self):
        for ep in range(self.n_iterations):
            s_curr = self.env.reset()
            a_curr = self.policy(self.get_discrete_state(s_curr))
            done = False
            score = 0
            episode_scores = []
            if ep < (self.n_iterations * 0.8):
                self.epsilon -= self.delta_epsilon
            while not done:
                s_next, r, done, _ = self.env.step(a_curr)
                score += r
                s_curr_discrete = self.get_discrete_state(s_curr)
                s_next_discrete = self.get_discrete_state(s_next)
                a_next = self.policy(s_next_discrete)

                q_curr = self.q_table[s_curr_discrete, a_curr]
                done_mask = int(done)
                # obtaining the target q value from the action sampled from current policy
                expected_q_next = 0
                for i in range(self.n_actions):
                    if i == np.argmax(self.q_table[s_next_discrete]):
                        p = 1 - self.epsilon + (self.epsilon / self.n_actions)
                    else:
                        p = self.epsilon / self.n_actions
                    expected_q_next += p * self.q_table[s_next_discrete, i]

                self.q_table[s_curr_discrete, a_curr] = q_curr + self.lr * (
                        r + (1 - done_mask) * (self.gamma * expected_q_next) - q_curr)
                s_curr = s_next
                a_curr = a_next
                episode_scores.append(score)

            self.test_scores.append(score)
            if (ep + 1) % 500 == 0:
                print(f"Iteration {ep}: score = {score}")

    def test(self, n_test_iterations, render):
        test_scores = []
        for i in range(n_test_iterations):
            s_curr = self.env.reset()
            done = False
            score = 0
            self.epsilon = 0.0
            while not done:
                a_curr = np.argmax(self.q_table[self.get_discrete_state(s_curr)])
                if render:
                    self.env.render()
                s_next, r, done, _ = self.env.step(a_curr)

                score += r

                s_curr = s_next
            test_scores.append(score)
            # print(f"test score: {score}")
        return sum(test_scores) / n_test_iterations

    def generate_plot(self, name):
        x_axis = [i for i in range(len(self.test_scores))]
        d = {"iteration": x_axis, "scores": self.test_scores}
        experiment = pd.DataFrame(data=d)
        plot = sns.lineplot(data=experiment, x="iteration", y="scores")
        plot.set_title(f"Score of {name} on Discretized MountainCar")
        plot.figure.savefig(f"Score_vs_Iteration_{name}.png")
