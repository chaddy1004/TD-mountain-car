import numpy as np

import itertools

import random
from collections import namedtuple

import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
from base_agent import BaseAgent

random.seed(a=19971124)
np.random.seed(seed=19940513)
sns.set_theme(style="darkgrid")


class TDNAgent(BaseAgent):

    def __init__(self, gamma, n_bins, n_iterations, env, n):
        super().__init__(gamma, n_bins, n_iterations, env)
        self.epsilon = 0.8
        self.MIN_EPSILON = 0.1
        self.n = n

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
            T = 200000
            states = []
            actions = []
            rewards = []
            tau = -1000
            t = 0
            s_curr = self.env.reset()
            a_curr = self.policy(self.get_discrete_state(s_curr))
            states.append(self.get_discrete_state(s_curr))
            actions.append(a_curr)
            rewards.append(0)
            score = 0

            while (tau != (T - 1)):
                # print(t, T-1)
                if t < T:
                    if ep < (self.n_iterations * 0.8):
                        self.epsilon -= self.delta_epsilon

                    s_next, r, done, _ = self.env.step(a_curr)
                    # if t < 201:
                    #     print(t, tau, T, done)
                    s_next_discrete = self.get_discrete_state(s_next)
                    states.append(s_next_discrete)
                    rewards.append(r)
                    score += r
                    if done:
                        T = t + 1
                        # print("T", T)
                    else:
                        a_next = self.policy(s_next_discrete)
                        actions.append(a_next)
                        a_curr = a_next
                        # episode_scores.append(score)

                tau = t - self.n + 1
                # print(t, tau, T - 1)
                if tau >= 0:
                    end = min(tau + self.n, T) + 1
                    G = 0
                    for i in range(tau + 1, end):
                        # print(ep, t, i)
                        # print(rewards)
                        G += (self.gamma ** (i - tau - 1)) * rewards[i]
                        # print("first two", (i - tau - 1), i)
                    if (tau + self.n) < T:
                        # print("last", self.n, tau+self.n)
                        # print(len(states), tau + self.n)
                        # print(len(actions))
                        # print(states[tau + self.n])
                        G += self.gamma ** (self.n) * self.q_table[states[tau + self.n], actions[tau + self.n]]
                    else:
                        # print("ELSE", tau, self.n, T, end)
                        pass

                    self.q_table[states[tau], actions[tau]] += self.lr * (G - self.q_table[states[tau], actions[tau]])
                t += 1
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
        return sum(test_scores)/n_test_iterations

    def generate_plot(self, name):
        x_axis = [i for i in range(len(self.test_scores))]
        d = {"iteration": x_axis, "scores": self.test_scores}
        experiment = pd.DataFrame(data=d)
        plt.clf()
        plot = sns.lineplot(data=experiment, x="iteration", y="scores")
        plot.set_title(f"Score of {self.n}_step SARSA on Discretized MountainCar")
        plot.figure.savefig(f"Score_vs_Iteration_{self.n}_step_SARSA.png")
