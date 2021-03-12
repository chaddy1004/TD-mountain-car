import numpy as np

import itertools

import random
from collections import namedtuple

import seaborn as sns

from base_agent import BaseAgent

random.seed(a=19971124)
np.random.seed(seed=19940513)
sns.set_theme(style="darkgrid")


class TDZeroAgent(BaseAgent):

    def __init__(self, gamma, n_bins, n_iterations, env):
        super().__init__(gamma, n_bins, n_iterations, env)
        self.epsilon = 0.1
        self.lr = 0.01

    def policy(self, state):
        decision = np.random.rand()
        if decision < self.epsilon:
            action = np.random.choice(self.n_actions)
        else:
            q = self.q_table[state]
            action = np.argmax(q)
        return action

    def sarsa(self):
        for ep in range(self.n_iterations):
            s_curr = self.env.reset()
            a_curr = self.policy(self.get_discrete_state(s_curr))
            done = False
            score = 0
            while not done:
                # env.render()
                s_next, r, done, _ = self.env.step(a_curr)
                r = r if not done or score >= 199 else - 100

                score += r

                if done:
                    score = score if score == 200 else score + 100
                    # print("St################Goal Reached###################", score)
                s_curr_discrete = self.get_discrete_state(s_curr)
                s_next_discrete = self.get_discrete_state(s_next)
                a_next = self.policy(s_next_discrete)

                q_curr = self.q_table[s_curr_discrete, a_curr]
                done = int(done)
                # obtaining the target q value from the action sampled from current policy
                q_next = self.q_table[s_next_discrete, a_next]

                self.q_table[s_curr_discrete, a_curr] = q_curr + self.lr * (r + (1 - done) * (self.gamma * q_next) - q_curr)

                s_curr = s_next
                a_curr = a_next
            score = score if score == 200 else score + 100
            self.test_scores.append(score)
            if (ep + 1) % 1 == 0:
                print(f"Iteration {ep}: score = {score}")

    def generate_plot(self):
        x_axis = [i for i in range(len(self.test_scores))]
        d = {"iteration": x_axis, "scores": self.test_scores}
        experiment = pd.DataFrame(data=d)
        plot = sns.lineplot(data=experiment, x="iteration", y="scores")
        plot.set_title("Score of Off Policy MC Control on Discretized Cartpole")
        plot.figure.savefig("Score_vs_Iteration.png")
