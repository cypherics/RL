from math import gamma
import matplotlib.pyplot as plt
import numpy as np
from gym.envs.toy_text import CliffWalkingEnv
from cliffwalking_utils import plot_results
from utils import env_reset, env_step


class DeterministicSARSA:
  def __init__(self, alpha, eps, gamma, alpha_decay, eps_decay, max_train_iterations, max_episode_length):
    self.alpha = alpha
    self.eps = eps
    self.gamma = gamma
    self.alpha_decay = alpha_decay
    self.eps_decay = eps_decay
    self.max_train_iterations = max_train_iterations
    self.max_episode_length = max_episode_length

    self.env = CliffWalkingEnv()
    self.num_actions = self.env.action_space.n
    self.num_observations = self.env.observation_space.n

    self.Q = np.zeros((self.num_observations, self.num_actions))

  def policy(self, state, is_training):
    # TODO: Implement an epsilon-greedy policy
    # - with probability eps return a random action
    # - otherwise find the action that maximizes Q
    rand = np.random.random()
    if rand > self.eps:
      return np.argmax(self.get_qs(state))

    return np.random.randint(0, self.num_actions)

  def get_qs(self, state):
    reward_list = []
    for action in range(0, self.num_actions):
      if self.is_state_int(state):
        reward_list.append(self.Q[state][action])
      else:
        reward_list.append(self.Q[state[0]][action])
    
    return reward_list

  def is_state_int(self, state):
    try:
      if len(state) > 0:
        return False
    except:
      return True

  def train_step(self, state, action, reward, next_state, next_action, done):
    # TODO: Implement the SARSA update.
    # - Q(s, a) = alpha * (reward + gamma * Q(s', a') - Q(s, a))
    # - Make sure that Q(s', a') = 0 if we reach a terminal state
    if self.is_state_int(state):
      state = [state]
      
    if done:
      self.Q[next_state][next_action] = 0
    self.Q[state[0]][action] = self.alpha * (reward + self.gamma * self.Q[next_state][np.argmax(self.get_qs(next_state))] - self.Q[state[0]][action])

  def run_episode(self, training, render=False):
    episode_reward = 0
    state = env_reset(self.env, render)
    action = self.policy(state, training)
    for t in range(self.max_episode_length):

      next_state, reward, done, _, _ = env_step(self.env, action, render)
      episode_reward += reward
      next_action = self.policy(next_state, training)
      if training:
        self.train_step(state, action, reward, next_state, next_action, done)
      state, action = next_state, next_action
      if done:
        break
    return episode_reward

  def train(self):
    self.train_reward = []
    for it in range(self.max_train_iterations):
      self.train_reward.append(self.run_episode(training=True))
      self.alpha *= self.alpha_decay
      self.eps *= self.eps_decay

  def test(self, render=False):
    self.test_reward = self.run_episode(training=False, render=render)



def find_and_param_combinations(alpha_range, eps_gamma_range, test_results, epsilon=True):

    top_combinations = []
    flattened = np.array(test_results).flatten()
    sorted_flat = np.sort(flattened)
    reverse_sorted = sorted_flat[::-1]
    vals_already_handled = []
    
    for val in reverse_sorted:
      if val not in vals_already_handled:
        vals_already_handled.append(val)
        index = np.where(test_results == val)
        top_combinations.append((alpha_range[index[0][0]], eps_gamma_range[index[1][0]], val))
      
      if len(top_combinations) == 3:
        break
      
    eps_or_gamma = "epsilon" if epsilon else "gamma"
    print(f"Top 3 combinations alpha-{eps_or_gamma}-tuning")
    print("===================")
    for combi in top_combinations:
      print(f"The combination of alpha: {combi[0]} and {eps_or_gamma}: {combi[1]} yielded following reward averaged over 200 episodes: {combi[2]}")
  

def tune_alpha_eps(gamma=0.9):
  # TODO Create suitable parameter ranges (np.arange)
  alpha_range = np.linspace(0.0, 1.0, num=11)
  eps_range = np.linspace(0.0, 1.0, num=11)

  # TODO: Change `debugging` to `False` after finishing your implementation! Report the results averaged over 5 repetitions!
  debugging = True
  if debugging:
    num_repetitions = 1
  else:
    num_repetitions = 5

  train_results = np.zeros((len(alpha_range), len(eps_range), num_repetitions))
  test_results = np.zeros((len(alpha_range), len(eps_range), num_repetitions))

  for i, alpha in enumerate(alpha_range):
    print(f'alpha = {alpha:0.2f} ', end='')
    for j, eps in enumerate(eps_range):
      for k in range(num_repetitions):
        alg = DeterministicSARSA(alpha=alpha, eps=eps, gamma=gamma, alpha_decay=1., eps_decay=1.,
                                 max_train_iterations=200, max_episode_length=200)
        alg.train()
        alg.test()
        train_results[i, j, k] = np.mean(alg.train_reward)
        test_results[i, j, k] = alg.test_reward
      print('.', end='')
    print()

  
  find_and_param_combinations(alpha_range, eps_range, test_results)
  # TODO: Find and print the top-3 parameter combinations, that perform best during the test phase

  plot_results(alpha_range, eps_range, 'alpha', 'epsilon', train_results, test_results)


def tune_alpha_gamma(eps=0.1):
  # TODO Create suitable parameter ranges (np.arange)
  alpha_range = np.linspace(0.0, 1.0, num=11)
  gamma_range = np.linspace(0.0, 1.0, num=11)

  # TODO: Change `debugging` to `False` after finishing your implementation! Report the results averaged over 5 repetitions!
  debugging = True
  if debugging:
    num_repetitions = 1
  else:
    num_repetitions = 5

  train_results = np.zeros((len(alpha_range), len(gamma_range), num_repetitions))
  test_results = np.zeros((len(alpha_range), len(gamma_range), num_repetitions))

  for i, alpha in enumerate(alpha_range):
    print(f'alpha = {alpha:0.2f} ', end='')
    for j, gamma in enumerate(gamma_range):
      for k in range(num_repetitions):
        alg = DeterministicSARSA(alpha=alpha, eps=eps, gamma=gamma, alpha_decay=1., eps_decay=1.,
                                 max_train_iterations=200, max_episode_length=200)
        alg.train()
        alg.test()
        train_results[i, j, k] = np.mean(alg.train_reward)
        test_results[i, j, k] = alg.test_reward
      print('.', end='')
    print()

  find_and_param_combinations(alpha_range, gamma_range, test_results, False)
  # TODO: Find and print the top-3 parameter combinations, that perform best during the test phase

  plot_results(alpha_range, gamma_range, 'alpha', 'gamma', train_results, test_results)


if __name__ == '__main__':
  tune_alpha_eps(gamma=0.9)
  tune_alpha_gamma(eps=0.1)
