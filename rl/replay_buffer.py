from collections import deque
import random

class ReplayBuffer(object):

  def __init__(self, buffer_size):

    self.buffer_size = buffer_size
    self.num_experiences = 0
    self.buffer = deque()

  def getBatch(self, batch_size):
    # random draw N
    return random.sample(self.buffer, batch_size)

  def size(self):
    return self.buffer_size

  def add(self, state, action, reward, next_action, done):
    new_experience = (state, action, reward, next_action, done)
    if self.num_experiences < self.buffer_size:
      self.buffer.append(new_experience)
      self.num_experiences += 1
    else:
      self.buffer.popleft()
      self.buffer.append(new_experience)

  def count(self):
    # if buffer is full, return buffer size
    # otherwise, return experience counter
    return self.num_experiences

  def erase(self):
    self.buffer = deque()
    self.num_experiences = 0