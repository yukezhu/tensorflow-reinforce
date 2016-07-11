import random
import numpy as np
import tensorflow as tf

class PolicyGradient(object):

  def __init__(self, session,
                     optimizer,
                     policy_network,
                     state_dim,
                     num_actions,
                     batch_size=32,
                     discount_factor=0.99, # discount future rewards
                     reg_param=0.0001,     # regularization constants
                     max_gradient=5,       # max gradient norms
                     summary_writer=None,
                     summary_every=100):

    # tensorflow machinery
    self.session        = session
    self.optimizer      = optimizer
    self.summary_writer = summary_writer

    # model components
    self.policy_network = policy_network

    # training parameters
    self.batch_size      = batch_size
    self.state_dim       = state_dim
    self.num_actions     = num_actions
    self.discount_factor = discount_factor
    self.max_gradient    = max_gradient
    self.reg_param       = reg_param

    self.exploration     = 0.1      # final exploration prob

    # counters
    self.train_iteration = 0

    # rollout buffer
    self.state_buffer  = []
    self.reward_buffer = []
    self.action_buffer = []
    self.gradient_map  = dict()

    self.all_rewards_mean = []
    self.all_rewards_std  = []

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
    # rollout action based on current policy
    with tf.name_scope("predict_actions"):
      # raw state representation
      self.states = tf.placeholder(tf.float32, (None, self.state_dim), name="states")
      # initialize policy network
      with tf.variable_scope("policy_network"):
        self.policy_outputs = self.policy_network(self.states)
      # predict actions from policy network
      self.action_scores = tf.identity(self.policy_outputs, name="action_scores")
      # Note 1: tf.multinomial is not good enough to use yet
      # so we don't use self.predicted_actions for now
      self.predicted_actions = tf.multinomial(self.action_scores, 1)

    # compute loss and gradients
    with tf.name_scope("compute_pg_gradients"):
      # gradients for selecting action from policy network
      self.rollout_states   = tf.placeholder(tf.float32, (None, self.state_dim), name="rollout_states")
      self.rollout_actions  = tf.placeholder(tf.int32,   (None,), name="rollout_actions")
      self.discount_rewards = tf.placeholder(tf.float32, (None,), name="discount_rewards")
      with tf.variable_scope("policy_network", reuse=True):
        self.rollout_logprobs = self.policy_network(self.rollout_states)
      # gradients that encourage the network to take the actions that were taken
      # self.action_targets     = tf.one_hot(self.rollout_actions, self.num_actions, 1.0, 0.0)
      # self.cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(self.rollout_logprobs, self.action_targets)
      self.cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(self.rollout_logprobs, self.rollout_actions)
      self.pg_loss            = tf.reduce_mean(self.cross_entropy_loss * self.discount_rewards)

      # regularization loss
      policy_network_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="policy_network")
      self.reg_loss = self.reg_param * tf.reduce_sum([tf.reduce_sum(tf.square(x)) for x in policy_network_variables])

      # compute total loss and gradients
      self.loss      = self.pg_loss + self.reg_loss
      self.gradients = self.optimizer.compute_gradients(self.loss)

      self.grad_placeholder = []
      for i, (grad, var) in enumerate(self.gradients):
        # clip gradients by norm
        if grad is not None:
          self.gradients[i] = (tf.clip_by_norm(grad, self.max_gradient), var)
        # create gradient placeholder
        self.grad_placeholder.append((tf.placeholder("float", shape=var.get_shape()), var))

      for grad, var in self.gradients:
        tf.histogram_summary(var.name, var)
        if grad is not None:
          tf.histogram_summary(var.name + '/gradients', grad)

      tf.scalar_summary("pg_loss", self.pg_loss)
      tf.scalar_summary("reg_loss", self.reg_loss)
      tf.scalar_summary("total_loss", self.loss)

    # training update
    with tf.name_scope("train_policy_network"):
      # apply gradients to update policy network
      self.train_op = self.optimizer.apply_gradients(self.grad_placeholder)

    self.summarize = tf.merge_all_summaries()
    self.no_op = tf.no_op()

  def sampleAction(self, states):
    # TODO: use this code piece when tf.multinomial gets better
    # sample action from current policy
    # actions = self.session.run(self.predicted_actions, {self.states: states})[0]
    # return actions[0]

    # temporary workaround
    def softmax(y):
      """ simple helper function here that takes unnormalized logprobs """
      maxy = np.amax(y)
      e = np.exp(y - maxy)
      return e / np.sum(e)

    # epsilon-greedy exploration strategy
    if random.random() < self.exploration:
      return random.randint(0, self.num_actions-1)
    else:
      action_scores = self.session.run(self.action_scores, {self.states: states})[0]
      action_probs  = softmax(action_scores) - 1e-5
      action = np.argmax(np.random.multinomial(1, action_probs))
      return action

  def storeRollout(self, state, action, reward):
    self.action_buffer.append(action)
    self.reward_buffer.append(reward)
    self.state_buffer.append(state)

  def updateModel(self):

    states   = np.zeros((self.batch_size, self.state_dim))
    actions  = np.zeros((self.batch_size,))
    rewards  = np.zeros((self.batch_size,))

    # only update model when we see a reward
    if np.std(self.reward_buffer) < np.finfo(np.float).eps:
      self.cleanUp()
      return

    N = len(self.reward_buffer)
    r = 0 # use discounted reward to approximate Q value

    # compute discounted future rewards
    discount_rewards = np.zeros(N)
    for t in reversed(xrange(N)):
      # future discounted reward from now on
      r = self.reward_buffer[t] + self.discount_factor * r
      discount_rewards[t] = r

    self.all_rewards_mean.append(np.mean(discount_rewards))
    self.all_rewards_std.append(np.std(discount_rewards))
    discount_rewards -= np.mean(self.all_rewards_mean)
    discount_rewards /= np.mean(self.all_rewards_std)

    print('update model: mean = ', np.mean(self.all_rewards_mean))
    print('update model: std  = ', np.mean(self.all_rewards_std))

    # # normalize rewards for robust gradients
    # discount_rewards -= np.mean(discount_rewards)
    # discount_rewards /= np.std(discount_rewards)

    # update policy network with the rollout in batches
    k = 0 # batch counter
    num_batches = 0
    for t in xrange(N):

      # future discounted reward from now on
      states[k]  = self.state_buffer[t]
      actions[k] = self.action_buffer[t]
      rewards[k] = discount_rewards[t]
      k += 1

      # once we have a full batch (or the last batch)
      if k >= self.batch_size or t == N - 1:
        # handle last batch
        states  = states[:k]
        actions = actions[:k]
        rewards = rewards[:k]

        # whether to calculate summaries
        calculate_summaries = self.train_iteration % self.summary_every == 0 and self.summary_writer is not None

        # evaluate gradients
        grad_evals = [grad for grad, var in self.gradients]

        # perform one update of training
        feed_dict = {
          self.rollout_states:   states,
          self.rollout_actions:  actions,
          self.discount_rewards: rewards
        }

        gradients = self.session.run(grad_evals, feed_dict)

        # # pg_loss = self.session.run(self.pg_loss, feed_dict)
        # from IPython import embed; embed()
        
        summary_str = self.session.run(self.summarize if calculate_summaries else self.no_op, feed_dict)

        # accumulate gradients in a chain of rollout
        for i, (_, var) in enumerate(self.gradients):
          if var.name not in self.gradient_map:
            self.gradient_map[var.name] = 0
          self.gradient_map[var.name] += gradients[i]

        # emit summaries
        if calculate_summaries:
          self.summary_writer.add_summary(summary_str, self.train_iteration)

        k = 0
        num_batches += 1
        self.train_iteration += 1

    # take one training update
    feed_dict = {}
    for i, (grad, var) in enumerate(self.grad_placeholder):
      feed_dict[grad] = self.gradient_map[var.name] / num_batches
    self.session.run(self.train_op, feed_dict)

    # clean up
    self.cleanUp()

  def cleanUp(self):
    self.state_buffer  = []
    self.reward_buffer = []
    self.action_buffer = []
    self.gradient_map  = dict()