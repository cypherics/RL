import math

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from cartpole_utils import plot_results
from utils import env_reset, env_step, update, update_metrics, print_metrics


def convert(x):
    return torch.tensor(x).float().unsqueeze(0)


class CartpoleNonLinearQ:
    def __init__(
        self,
        alpha,
        eps,
        gamma,
        alpha_decay,
        eps_decay,
        max_train_iterations,
        max_test_iterations,
        max_episode_length,
    ):
        self.alpha = alpha
        self.eps = eps
        self.gamma = gamma
        self.alpha_decay = alpha_decay
        self.eps_decay = eps_decay
        self.max_train_iterations = max_train_iterations
        self.max_test_iterations = max_test_iterations
        self.max_episode_length = max_episode_length

        self.env = gym.make("CartPole-v0")
        # self.env = gym.make("MountainCar-v0")

        self.num_actions = self.env.action_space.n
        self.num_observations = self.env.observation_space.shape[0]

        self.hidden = 32
        self.model = nn.Sequential(
            nn.Linear(self.num_observations, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.num_actions),
        )
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)

    def random_action(self):
        return np.random.randint(0, self.num_actions)

    def greedy_action(self, qs):
        return torch.argmax(qs)

    def policy(self, state, training):
        # TODO: Implement an epsilon-greedy policy
        # - with probability eps return a random action
        # - otherwise find the action that maximizes Q
        # - During the rollout phase, we don't need to compute the gradient!
        #   (Hint: use torch.no_grad()). The policy should return torch tensors.
        with torch.no_grad():
            if training:
                p = self.model(convert(state))
                if np.random.rand() < self.eps:
                    a = self.random_action()
                else:
                    a = self.greedy_action(p).tolist()

            else:
                p = self.model(convert(state))
                a = self.greedy_action(p).tolist()

        return convert(a)

    def compute_loss(self, state, action, reward, next_state, next_action, done):
        state = convert(state)
        next_state = convert(next_state)
        action = action.view(1, 1)
        reward = torch.tensor(reward).view(1, 1)
        done = torch.tensor(done).int().view(1, 1)

        # TODO: Compute Q(s, a) and Q(s', a') for the given state-action pairs.
        # Detach the gradient of Q(s', a'). Why do we have to do that? Think about
        # the effect of backpropagating through Q(s, a) and Q(s', a') at once!

        _q = torch.gather(self.model(state), dim=1, index=action.long())
        _q_next = (
            reward.detach()
            if done
            else (
                reward + (self.gamma * torch.max(self.model(next_state).detach()))
            ).detach()
        )

        q_network_loss = self.criterion(_q, _q_next)
        return q_network_loss

    def train_step(self, state, action, reward, next_state, next_action, done):
        loss = self.compute_loss(state, action, reward, next_state, next_action, done)
        # TODO: prepare the optimizer (reset gradients)
        self.optimizer.zero_grad()

        # TODO: compute gradients based on loss
        loss.backward()
        # TODO: update the model parameters using the optimizer
        self.optimizer.step()
        return loss.item()

    def run_episode(self, training):
        episode_reward, episode_loss = 0, 0.0
        state = env_reset(self.env, False)
        for t in range(self.max_episode_length):
            action = self.policy(state, self.eps)
            next_state, reward, done, _ = env_step(self.env, int(action.item()), False)
            next_action = None
            episode_reward += reward

            if training:
                episode_loss += self.train_step(
                    state, action, reward, next_state, next_action, done
                )
            else:
                with torch.no_grad():
                    episode_loss += self.compute_loss(
                        state, action, reward, next_state, next_action, done
                    ).item()

            state = next_state
            if done:
                break
        return dict(reward=episode_reward, loss=episode_loss / t)

    def train(self):
        self.train_metrics = dict(reward=[], loss=[])
        for it in range(self.max_train_iterations):
            episode_metrics = self.run_episode(training=True)
            update_metrics(self.train_metrics, episode_metrics)
            if it % 100 == 0:
                print_metrics(it, self.train_metrics, training=True)
            self.eps *= self.eps_decay

    def test(self):
        self.test_metrics = dict(reward=[], loss=[])
        with torch.no_grad():
            for it in range(self.max_test_iterations):
                episode_metrics = self.run_episode(training=False)
                update_metrics(self.test_metrics, episode_metrics)
        print_metrics(it + 1, self.test_metrics, training=False)
        plot_results(self.train_metrics, self.test_metrics)


if __name__ == "__main__":
    alg = CartpoleNonLinearQ(
        alpha=1e-03,
        eps=1,
        gamma=0.9,
        eps_decay=0.999,
        max_train_iterations=5000,
        alpha_decay=0.999,
        max_test_iterations=500,
        max_episode_length=2000,
    )
    alg.train()
    alg.test()
