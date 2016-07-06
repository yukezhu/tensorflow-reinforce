# Tensorflow-Reinforce
A collection of [Tensorflow](https://www.tensorflow.org) implementations of reinforcement learning models. Models are evaluated in [OpenAI Gym](https://gym.openai.com) environments.

Models
======
| Model          | Code           | References  |
|:-------------  |:-------------- |:------------|
| Cross-Entropy Method | [run_cem_cartpole](https://github.com/yukezhu/tensorflow-reinforce/blob/master/run_cem_cartpole.py) | [Cross-entropy method](https://en.wikipedia.org/wiki/Cross-entropy_method) |
| Tabular Q Learning | [rl/tabular_q_learner](https://github.com/yukezhu/tensorflow-reinforce/blob/master/rl/tabular_q_learner.py) | [Sutton and Barto, Chapter 8](http://people.inf.elte.hu/lorincz/Files/RL_2006/SuttonBook.pdf) |
| Deep Q Network | [rl/neural_q_learner](https://github.com/yukezhu/tensorflow-reinforce/blob/master/rl/neural_q_learner.py) | [Mnih et al.](http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html) |
| Double Deep Q Network | [rl/neural_q_learner](https://github.com/yukezhu/tensorflow-reinforce/blob/master/rl/neural_q_learner.py) | [van Hasselt et al.](http://arxiv.org/abs/1509.06461) |
