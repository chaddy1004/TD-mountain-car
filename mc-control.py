import numpy as np

import itertools

import random
from collections import namedtuple

import seaborn as sns

from base_agent import BaseAgent

random.seed(a=19971124)
np.random.seed(seed=19940513)
sns.set_theme(style="darkgrid")


class McControlAgent(BaseAgent):
    def __init__(self, gamma, n_bins, n_iterations, env):

        super().__init__(gamma, n_bins, n_iterations, env)

    def generate_episodes(self, iteration):
        done = False
        episode = []

        Sample = namedtuple('Sample', ['s', 'a', 'b_a_s', 'r', 's_next', 'done'])
        s_curr = self.env.reset()
        score = 0
        while not done:
            # action = np.random.choice(n_actions, 1, p=policy)
            if iteration < self.n_iterations // 8:
                epsilon = 0.1
            elif self.n_iterations // 8 < iteration < self.n_iterations // 4:
                epsilon = 0.05
            else:
                epsilon = 0.001
            random_sampled = random.uniform(0, 1)
            if random_sampled < epsilon:
                action_prob = epsilon / 2
                prob_value = random.uniform(0, 1)
                if prob_value < 0.5:
                    action = 0
                else:
                    action = 1
            else:
                # greedy action selection
                action_prob = (1 - epsilon)
                action = np.argmax(self.q_table[self.get_discrete_state(s_curr)])
            s_next, r, done, _ = self.env.step(int(action))

            # reward of -100 (punishment) if the pole falls before reaching the maximum score
            r = r if not done or r >= 199 else - 100

            sample = Sample(s=s_curr, a=int(action), b_a_s=action_prob, r=r, s_next=s_next, done=done)
            episode.append(sample)

            # score is reflective of how much reward the agent received
            score += r
            s_curr = s_next

        return episode

    def mc_control(self):
        for i in range(self.n_iterations):
            episode = self.generate_episodes(iteration=i)
            G_tp1 = 0  # G_{t+1}
            W = 1
            for step in reversed(episode):
                s_curr, a, b_a_s, r, s_next, done = step.s, step.a, step.b_a_s, step.r, step.s_next, step.done
                G_t = r + self.gamma * G_tp1
                s_curr_discrete = self.get_discrete_state(s_curr)
                self.c[s_curr_discrete][a] = self.c[s_curr_discrete][a] + W
                self.q_table[s_curr_discrete][a] = self.q_table[s_curr_discrete][a] + (
                        W / self.c[s_curr_discrete][a]) * (
                                                           G_t - self.q_table[s_curr_discrete][a])
                pi_st = np.argmax(self.q_table[s_curr_discrete])
                G_tp1 = G_t
                if a != pi_st:
                    break
                W = W / b_a_s

            done = False
            s_curr = self.env.reset()
            score = 0
            while not done:
                # if len(agent.experience_replay) == agent.replay_size:
                #     env.render()
                s_curr_discrete = self.get_discrete_state(s_curr)
                action = np.argmax(self.q_table[s_curr_discrete])
                s_next, r, done, _ = self.env.step(int(action))
                r = r if not done or score >= 199 else - 100
                score += r
                s_curr = s_next
            score = score if score == 200 else score + 100
            self.test_scores.append(score)
            if (i + 1) % 500 == 0:
                print(f"Iteration {i}: score = {score}")
