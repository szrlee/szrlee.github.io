---
title: 'HyperDQN: A Randomized Exploration Method for Deep Reinforcement Learning'
authors:
- Ziniu Li
- Yingru Li
- Yushun Zhang
- Tong Zhang
- Zhi-Quan Luo
date: '2022-01-01'
publishDate: '2024-02-21T16:25:24.164184Z'
publication_types:
- paper-conference
publication: '*International Conference on Learning Representations (ICLR)*'
links:
- name: ICLR2022
  url: https://openreview.net/pdf?id=X0nrKAXu7g-
tags:
  - efficiency
  - reinforcement-earning
  - algorithm
  - deep-learning
  - agent

abstract: Randomized least-square value iteration (RLSVI) is a provably efficient exploration method. However, it is limited to the case where (1) a good feature is known in advance and (2) this feature is fixed during the training. If otherwise, RLSVI suffers an unbearable computational burden to obtain the posterior samples. In this work, we present a practical algorithm named HyperDQN to address the above issues under deep RL. In addition to a non-linear neural network (i.e., base model) that predicts Q-values, our method employs a probabilistic hypermodel (i.e., meta model), which outputs the parameter of the base model. When both models are jointly optimized under a specifically designed objective, three purposes can be achieved. First, the hypermodel can generate approximate posterior samples regarding the parameter of the Q-value function. As a result, diverse Q-value functions are sampled to select exploratory action sequences. This retains the punchline of RLSVI for efficient exploration. Second, a good feature is learned to approximate Q-value functions. This addresses limitation (1). Third, the posterior samples of the Q-value function can be obtained in a more efficient way than the existing methods, and the changing feature does not affect the efficiency. This deals with limitation (2). On the Atari suite, HyperDQN with 20M frames outperforms DQN with 200M frames in terms of the maximum human-normalized score. For SuperMarioBros, HyperDQN outperforms several exploration bonus and randomized exploration methods on 5 out of 9 games.

# Summary. An optional shortened abstract.
summary: We design a practical randomized exploration method to address the sample efficiency issue in online reinforcement learning.

projects:
  - Efficient RL
---
