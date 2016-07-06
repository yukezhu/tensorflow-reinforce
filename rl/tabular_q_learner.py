import random
import numpy as np

# tabular Q-learning where states and actions
# are discrete and stored in a table
class QLearner(object):

  def __init__(self, state_dim,
                     num_actions,
                     init_exp=0.5,     # initial exploration prob
                     final_exp=0.0,    # final exploration prob
                     anneal_steps=500, # N steps for annealing exploration 
                     alpha = 0.2,
                     discount_factor=0.9): # discount future rewards

    # Q learning parameters
    self.state_dim       = state_dim
    self.num_actions     = num_actions
    self.exploration     = init_exp
    self.init_exp        = init_exp
    self.final_exp       = final_exp
    self.anneal_steps    = anneal_steps
    self.discount_factor = discount_factor
    self.alpha           = alpha

    # counters
    self.train_iteration = 0

    # table of q values
    self.qtable = np.random.uniform(low=-1, high=1, size=(state_dim, num_actions))

  def initializeState(self, state):
    self.state = state
    self.action = self.qtable[state].argsort()[-1]
    return self.action

  # select action based on epsilon-greedy strategy
  def eGreedyAction(self, state):
    if self.exploration > random.random():
      action = random.randint(0, self.num_actions-1)
    else:
      action = self.qtable[state].argsort()[-1]
    return action

  # do one value iteration update
  def updateModel(self, state, reward):
    action = self.eGreedyAction(state)

    self.train_iteration += 1
    self.annealExploration()
    self.qtable[self.state, self.action] = (1 - self.alpha) * self.qtable[self.state, self.action] + self.alpha * (reward + self.discount_factor * self.qtable[state, action])

    self.state = state
    self.action = action

    return self.action

  # anneal learning rate
  def annealExploration(self, stategy='linear'):
    ratio = max((self.anneal_steps - self.train_iteration)/float(self.anneal_steps), 0)
    self.exploration = (self.init_exp - self.final_exp) * ratio + self.final_exp
