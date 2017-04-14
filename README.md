# Tensorflow-Reinforce
A collection of [Tensorflow](https://www.tensorflow.org) implementations of reinforcement learning models. Models are evaluated in [OpenAI Gym](https://gym.openai.com) environments. Any contribution/feedback is more than welcome. **Disclaimer:** These implementations are used for educational purposes only (i.e., to learn deep RL myself). There is no guarantee that the exact models will work on any of your particular RL problems without changes.

Environments
------------
This codebase works in both Python 2.7 and 3.5. The models are implemented in Tensorflow 1.0.

Models
------
| Model          | Code           | References  |
|:-------------  |:-------------- |:------------|
| Cross-Entropy Method | [run_cem_cartpole](https://github.com/yukezhu/tensorflow-reinforce/blob/master/run_cem_cartpole.py) | [Cross-entropy method](https://en.wikipedia.org/wiki/Cross-entropy_method) |
| Tabular Q Learning | [rl/tabular_q_learner](https://github.com/yukezhu/tensorflow-reinforce/blob/master/rl/tabular_q_learner.py) | [Sutton and Barto, Chapter 8](http://people.inf.elte.hu/lorincz/Files/RL_2006/SuttonBook.pdf) |
| Deep Q Network | [rl/neural_q_learner](https://github.com/yukezhu/tensorflow-reinforce/blob/master/rl/neural_q_learner.py) | [Mnih et al.](http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html) |
| Double Deep Q Network | [rl/neural_q_learner](https://github.com/yukezhu/tensorflow-reinforce/blob/master/rl/neural_q_learner.py) | [van Hasselt et al.](http://arxiv.org/abs/1509.06461) |
| REINFORCE Policy Gradient | [rl/pg_reinforce](https://github.com/yukezhu/tensorflow-reinforce/blob/master/rl/pg_reinforce.py) | [Sutton et al.](https://webdocs.cs.ualberta.ca/~sutton/papers/SMSM-NIPS99.pdf) |
| Actor-critic Policy Gradient | [rl/pg_actor_critic](https://github.com/yukezhu/tensorflow-reinforce/blob/master/rl/pg_actor_critic.py) | [Minh et al.](https://arxiv.org/abs/1602.01783) |
| Deep Deterministic Policy Gradient | [rl/pg_ddpg](https://github.com/yukezhu/tensorflow-reinforce/blob/master/rl/pg_ddpg.py) | [Lillicrap et al.](https://arxiv.org/abs/1509.02971) |

License
-------
MIT
