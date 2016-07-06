from __future__ import print_function
from collections import deque
from rl.tabular_q_learner import QLearner
import numpy as np
import gym

env_name = 'CartPole-v0'
env = gym.make(env_name)

cart_position_bins = np.linspace(-2.4, 2.4, num = 11)[1:-1]
pole_angle_bins = np.linspace(-2, 2, num = 11)[1:-1]
cart_velocity_bins = np.linspace(-1, 1, num = 11)[1:-1]
angle_rate_bins = np.linspace(-3.5, 3.5, num = 11)[1:-1]

def digitalizeState(observation):
  return int("".join([str(o) for o in observation]))

state_dim   = 10 ** env.observation_space.shape[0]
num_actions = env.action_space.n

q_learner = QLearner(state_dim, num_actions)

MAX_EPISODES = 10000
MAX_STEPS    = 200

episode_history = deque(maxlen=100)
for i_episode in xrange(MAX_EPISODES):

  # initialize
  observation = env.reset()
  cart_position, pole_angle, cart_velocity, angle_rate_of_change = observation
  state = digitalizeState([np.digitize(cart_position, cart_position_bins),
                            np.digitize(pole_angle, pole_angle_bins),
                            np.digitize(cart_velocity, cart_velocity_bins),
                            np.digitize(angle_rate_of_change, angle_rate_bins)])
  action = q_learner.initializeState(state)
  total_rewards = 0

  for t in xrange(MAX_STEPS):
    env.render()
    observation, reward, done, _ = env.step(action)
    cart_position, pole_angle, cart_velocity, angle_rate_of_change = observation

    state = digitalizeState([np.digitize(cart_position, cart_position_bins),
                              np.digitize(pole_angle, pole_angle_bins),
                              np.digitize(cart_velocity, cart_velocity_bins),
                              np.digitize(angle_rate_of_change, angle_rate_bins)])
    

    total_rewards += reward
    if done: reward = -200   # normalize reward
    action = q_learner.updateModel(state, reward)

    if done: break

  episode_history.append(total_rewards)
  mean_rewards = np.mean(episode_history)

  print("Episode {}".format(i_episode))
  print("Finished after {} timesteps".format(t+1))
  print("Reward for this episode: {}".format(total_rewards))
  print("Average reward for last 100 episodes: {}".format(mean_rewards))
  if mean_rewards >= 195.0:
    print("Environment {} solved after {} episodes".format(env_name, i_episode+1))
    break
