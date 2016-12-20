import random
import numpy as np
import tensorflow as tf
from replay_buffer import ReplayBuffer

class DeepDeterministicPolicyGradient(object):

  def __init__(self, session,
                     optimizer,
                     actor_network,
                     critic_network,
                     state_dim,
                     action_dim,
                     batch_size=32,
                     replay_buffer_size=1000000, # size of replay buffer
                     store_replay_every=1,       # how frequent to store experience
                     discount_factor=0.99,       # discount future rewards
                     target_update_rate=0.01,
                     reg_param=0.01,             # regularization constants
                     max_gradient=5,             # max gradient norms
                     noise_sigma=0.20,
                     noise_theta=0.15,
                     summary_writer=None,
                     summary_every=100):

    # tensorflow machinery
    self.session        = session
    self.optimizer      = optimizer
    self.summary_writer = summary_writer

    # model components
    self.actor_network  = actor_network
    self.critic_network = critic_network
    self.replay_buffer  = ReplayBuffer(buffer_size=replay_buffer_size)

    # training parameters
    self.batch_size         = batch_size
    self.state_dim          = state_dim
    self.action_dim         = action_dim
    self.discount_factor    = discount_factor
    self.target_update_rate = target_update_rate
    self.max_gradient       = max_gradient
    self.reg_param          = reg_param

    # Ornstein-Uhlenbeck noise for exploration
    self.noise_var = tf.Variable(tf.zeros([1, action_dim]))
    noise_random = tf.random_normal([1, action_dim], stddev=noise_sigma)
    self.noise = self.noise_var.assign_sub((noise_theta) * self.noise_var - noise_random)

    # counters
    self.store_replay_every   = store_replay_every
    self.store_experience_cnt = 0
    self.train_iteration      = 0

    # create and initialize variables
    self.create_variables()
    var_lists = tf.get_collection(tf.GraphKeys.VARIABLES)
    self.session.run(tf.initialize_variables(var_lists))

    # make sure all variables are initialized
    self.session.run(tf.assert_variables_initialized())

    if self.summary_writer is not None:
      # graph was not available when journalist was created
      self.summary_writer.add_graph(self.session.graph)
      self.summary_every = summary_every

  def create_variables(self):
    
    with tf.name_scope("model_inputs"):
      # raw state representation
      self.states = tf.placeholder(tf.float32, (None, self.state_dim), name="states")
      # action input used by critic network
      self.action = tf.placeholder(tf.float32, (None, self.action_dim), name="action")

    # define outputs from the actor and the critic
    with tf.name_scope("predict_actions"):
      # initialize actor-critic network
      with tf.variable_scope("actor_network"):
        self.policy_outputs = self.actor_network(self.states)
      with tf.variable_scope("critic_network"):
        self.value_outputs    = self.critic_network(self.states, self.action)
        self.action_gradients = tf.gradients(self.value_outputs, self.action)[0]

      # predict actions from policy network
      self.predicted_actions = tf.identity(self.policy_outputs, name="predicted_actions")
      tf.histogram_summary("predicted_actions", self.predicted_actions)
      tf.histogram_summary("action_scores", self.value_outputs)

    # get variable list
    actor_network_variables  = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="actor_network")
    critic_network_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="critic_network")

    # estimate rewards using the next state: r + argmax_a Q'(s_{t+1}, u'(a))
    with tf.name_scope("estimate_future_rewards"):
      self.next_states = tf.placeholder(tf.float32, (None, self.state_dim), name="next_states")
      self.next_state_mask = tf.placeholder(tf.float32, (None,), name="next_state_masks")
      self.rewards = tf.placeholder(tf.float32, (None,), name="rewards")

      # initialize target network
      with tf.variable_scope("target_actor_network"):
        self.target_actor_outputs = self.actor_network(self.next_states)
      with tf.variable_scope("target_critic_network"):
        self.target_critic_outputs = self.critic_network(self.next_states, self.target_actor_outputs)

      # compute future rewards
      self.next_action_scores = tf.stop_gradient(self.target_critic_outputs)[:,0] * self.next_state_mask
      tf.histogram_summary("next_action_scores", self.next_action_scores)
      self.future_rewards = self.rewards + self.discount_factor * self.next_action_scores

    # compute loss and gradients
    with tf.name_scope("compute_pg_gradients"):

      # compute gradients for critic network
      self.temp_diff        = self.value_outputs[:,0] - self.future_rewards
      self.mean_square_loss = tf.reduce_mean(tf.square(self.temp_diff))
      self.critic_reg_loss  = tf.reduce_sum([tf.reduce_sum(tf.square(x)) for x in critic_network_variables])
      self.critic_loss      = self.mean_square_loss + self.reg_param * self.critic_reg_loss
      self.critic_gradients = self.optimizer.compute_gradients(self.critic_loss, critic_network_variables)

      # compute actor gradients (we don't do weight decay for actor network)
      self.q_action_grad = tf.placeholder(tf.float32, (None, self.action_dim), name="q_action_grad")
      actor_policy_gradients = tf.gradients(self.policy_outputs, actor_network_variables, -self.q_action_grad)
      self.actor_gradients = zip(actor_policy_gradients, actor_network_variables)

      # collect all gradients
      self.gradients = self.actor_gradients + self.critic_gradients

      # clip gradients
      for i, (grad, var) in enumerate(self.gradients):
        # clip gradients by norm
        if grad is not None:
          self.gradients[i] = (tf.clip_by_norm(grad, self.max_gradient), var)

      # summarize gradients
      for grad, var in self.gradients:
        tf.histogram_summary(var.name, var)
        if grad is not None:
          tf.histogram_summary(var.name + '/gradients', grad)

      # emit summaries
      tf.scalar_summary("critic_loss", self.critic_loss)
      tf.scalar_summary("critic_td_loss", self.mean_square_loss)
      tf.scalar_summary("critic_reg_loss", self.critic_reg_loss)

      # apply gradients to update actor network
      self.train_op = self.optimizer.apply_gradients(self.gradients)

    # update target network with Q network
    with tf.name_scope("update_target_network"):
      self.target_network_update = []

      # slowly update target network parameters with the actor network parameters
      actor_network_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="actor_network")
      target_actor_network_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="target_actor_network")
      for v_source, v_target in zip(actor_network_variables, target_actor_network_variables):
        # this is equivalent to target = (1-alpha) * target + alpha * source
        update_op = v_target.assign_sub(self.target_update_rate * (v_target - v_source))
        self.target_network_update.append(update_op)

      # same for the critic network
      critic_network_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="critic_network")
      target_critic_network_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="target_critic_network")
      for v_source, v_target in zip(critic_network_variables, target_critic_network_variables):
        # this is equivalent to target = (1-alpha) * target + alpha * source
        update_op = v_target.assign_sub(self.target_update_rate * (v_target - v_source))
        self.target_network_update.append(update_op)

      # group all assignment operations together
      self.target_network_update = tf.group(*self.target_network_update)

    self.summarize = tf.merge_all_summaries()
    self.no_op = tf.no_op()

  def sampleAction(self, states, exploration=True):
    policy_outs, ou_noise = self.session.run([
      self.policy_outputs,
      self.noise
    ], {
      self.states: states
    })
    # add OU noise for exploration
    policy_outs = policy_outs + ou_noise if exploration else policy_outs
    return policy_outs

  def updateModel(self):

    # not enough experiences yet
    if self.replay_buffer.count() < self.batch_size:
      return

    batch           = self.replay_buffer.getBatch(self.batch_size)
    states          = np.zeros((self.batch_size, self.state_dim))
    rewards         = np.zeros((self.batch_size,))
    actions         = np.zeros((self.batch_size, self.action_dim))
    next_states     = np.zeros((self.batch_size, self.state_dim))
    next_state_mask = np.zeros((self.batch_size,))

    for k, (s0, a, r, s1, done) in enumerate(batch):
      states[k]  = s0
      rewards[k] = r
      actions[k] = a
      if not done:
        next_states[k] = s1
        next_state_mask[k] = 1

    # whether to calculate summaries
    calculate_summaries = self.train_iteration % self.summary_every == 0 and self.summary_writer is not None

    # compute a = u(s)
    policy_outs = self.session.run(self.policy_outputs, {
      self.states: states
    })

    # compute d_a Q(s,a) where s=s_i, a=u(s)
    action_grads = self.session.run(self.action_gradients, {
      self.states: states,
      self.action: policy_outs
    })

    critic_loss, _, summary_str = self.session.run([
      self.critic_loss,
      self.train_op,
      self.summarize if calculate_summaries else self.no_op
    ], {
      self.states:          states,
      self.next_states:     next_states,
      self.next_state_mask: next_state_mask,
      self.action:          actions,
      self.rewards:         rewards,
      self.q_action_grad:   action_grads
    })

    # update target network using Q-network
    self.session.run(self.target_network_update)

    # emit summaries
    if calculate_summaries:
      self.summary_writer.add_summary(summary_str, self.train_iteration)

    self.train_iteration += 1

  def storeExperience(self, state, action, reward, next_state, done):
    # always store end states
    if self.store_experience_cnt % self.store_replay_every == 0 or done:
      self.replay_buffer.add(state, action, reward, next_state, done)
    self.store_experience_cnt += 1
