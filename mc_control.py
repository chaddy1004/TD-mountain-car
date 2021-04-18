import numpy as np

import random
from collections import namedtuple

import seaborn as sns

from base_agent import BaseAgent
import pandas as pd

random.seed(a=19971124)
np.random.seed(seed=19940513)
sns.set_theme(style="darkgrid")



class McControlAgent(BaseAgent):
    def __init__(self, gamma, n_bins, n_iterations, env):
        super().__init__(gamma, n_bins, n_iterations, env)
        self.epsilon = 0.8
        self.MIN_EPSILON = 0.1
        self.delta_epsilon = (self.epsilon - self.MIN_EPSILON) / int(self.n_iterations * 0.5)
        # self.delta_epsilon = 0

    def generate_episodes(self):
        done = False
        episode = []

        Sample = namedtuple('Sample', ['s', 'a', 'b_a_s', 'r', 's_next', 'done'])
        s_curr = self.env.reset()
        score = 0

        while not done:
            # action = np.random.choice(n_actions, 1, p=policy)
            random_sampled = random.uniform(0, 1)
            if random_sampled < self.epsilon:
                action_prob = self.epsilon / self.n_actions
                action = np.random.choice(self.n_actions)
            else:
                # greedy action selection
                action_prob = (1 - self.epsilon) / self.n_actions
                action = np.argmax(self.q_table[self.get_discrete_state(s_curr)])
            s_next, r, done, _ = self.env.step(int(action))

            sample = Sample(s=s_curr, a=int(action), b_a_s=action_prob, r=r, s_next=s_next, done=done)
            episode.append(sample)

            # score is reflective of how much reward the agent received
            score += r
            s_curr = s_next

        return episode

    def mc_control(self):
        for ep in range(self.n_iterations):
            if (ep + 1) % 500 == 0:
                print("asdf")
            if ep < (self.n_iterations * 0.5):
                self.epsilon -= self.delta_epsilon
            episode = self.generate_episodes()
            G_tp1 = 0  # G_{t+1}
            W = 1
            for step in reversed(episode):
                s_curr, a, b_a_s, r, s_next, done = step.s, step.a, step.b_a_s, step.r, step.s_next, step.done
                G_t = r + self.gamma * G_tp1
                s_curr_discrete = self.get_discrete_state(s_curr)
                self.c[s_curr_discrete, a] = self.c[s_curr_discrete, a] + W
                self.q_table[s_curr_discrete, a] = self.q_table[s_curr_discrete, a] + (
                        W / self.c[s_curr_discrete, a]) * (
                                                           G_t - self.q_table[s_curr_discrete, a])
                pi_st = np.argmax(self.q_table[s_curr_discrete])
                G_tp1 = G_t
                if a != pi_st:
                    break
                W = W / b_a_s

            done = False
            s_curr = self.env.reset()
            s_curr_start = s_curr
            score = 0
            while not done:
                # if len(agent.experience_replay) == agent.replay_size:
                #     env.render()
                s_curr_discrete = self.get_discrete_state(s_curr)
                action = np.argmax(self.q_table[s_curr_discrete])
                s_next, r, done, _ = self.env.step(int(action))
                score += r
                s_curr = s_next
            self.test_scores.append(score)

            if (ep + 1) % 500 == 0 or score > -200:
                print(f"Iteration {ep}: score = {score}")

    def test(self, n_test_iterations, render):
        test_scores = []
        for i in range(n_test_iterations):
            s_curr = self.env.reset()
            done = False
            score = 0
            while not done:
                s_curr_discrete = self.get_discrete_state(s_curr)
                action = np.argmax(self.q_table[s_curr_discrete])
                s_next, r, done, _ = self.env.step(int(action))
                score += r
                s_curr = s_next
            test_scores.append(score)
            print(f"test score: {score}")
        return sum(test_scores) / n_test_iterations

    def generate_plot(self, name):
        x_axis = [i for i in range(len(self.test_scores))]
        d = {"iteration": x_axis, "scores": self.test_scores}
        experiment = pd.DataFrame(data=d)
        plot = sns.lineplot(data=experiment, x="iteration", y="scores")
        plot.set_title(f"Score of {name} on Discretized MountainCar")
        plot.figure.savefig("Score_vs_Iteration_mc_control.png")
