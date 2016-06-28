from __future__ import print_function
from collections import deque
import numpy as np
import gym

env_name = 'CartPole-v0'
env = gym.make(env_name)

def observation_to_action(ob, theta):
  # define policy neural network
  W1 = theta[:-1]
  b1 = theta[-1]
  return int((ob.dot(W1) + b1) < 0)

def theta_rollout(env, theta, num_steps, render = False):
  total_rewards = 0
  observation = env.reset()
  for t in range(num_steps):
    action = observation_to_action(observation, theta)
    observation, reward, done, _ = env.step(action)
    total_rewards += reward
    if render: env.render()
    if done: break
  return total_rewards, t

MAX_EPISODES = 10000
MAX_STEPS    = 200
batch_size   = 25
top_per      = 0.2 # percentage of theta with highest score selected from all the theta
std          = 1 # scale of standard deviation

# initialize
theta_mean = np.zeros(env.observation_space.shape[0] + 1)
theta_std = np.ones_like(theta_mean) * std

episode_history = deque(maxlen=100)
for i_episode in xrange(MAX_EPISODES):
  # maximize function theta_rollout through cross-entropy method
  theta_sample = np.tile(theta_mean, (batch_size, 1)) + np.tile(theta_std, (batch_size, 1)) * np.random.randn(batch_size, theta_mean.size)
  reward_sample = np.array([theta_rollout(env, th, MAX_STEPS)[0] for th in theta_sample])
  top_idx = np.argsort(-reward_sample)[:np.round(batch_size * top_per)]
  top_theta = theta_sample[top_idx]
  theta_mean = top_theta.mean(axis = 0)
  theta_std = top_theta.std(axis = 0)
  total_rewards, t = theta_rollout(env, theta_mean, MAX_STEPS, render = True)

  episode_history.append(total_rewards)
  mean_rewards = np.mean(episode_history)

  print("Episode {}".format(i_episode))
  print("Finished after {} timesteps".format(t+1))
  print("Reward for this episode: {}".format(total_rewards))
  print("Average reward for last 100 episodes: {}".format(mean_rewards))
  if mean_rewards >= 195.0:
    print("Environment {} solved after {} episodes".format(env_name, i_episode+1))
    break
