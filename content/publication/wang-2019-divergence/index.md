---
title: Divergence-augmented policy optimization
authors:
- Qing Wang
- Yingru Li
- Jiechao Xiong
- Tong Zhang
date: '2019-01-01'
publishDate: '2023-12-07T12:23:02.291196Z'
publication_types:
- paper-conference
publication: '*Advances in Neural Information Processing Systems*'

abstract: In deep reinforcement learning, policy optimization methods need to deal with issues such as function approximation and the reuse of off-policy data. Standard policy gradient methods do not handle off-policy data well, leading to premature convergence and instability. This paper introduces a method to stabilize policy optimization when off-policy data are reused. The idea is to include a Bregman divergence between the behavior policy that generates the data and the current policy to ensure small and safe policy updates with off-policy data. The Bregman divergence is calculated between the state distributions of two policies, instead of only on the action probabilities, leading to a divergence augmentation formulation. Empirical experiments on Atari games show that in the data-scarce scenario where the reuse of off-policy data becomes necessary, our method can achieve better performance than other state-of-the-art deep reinforcement learning algorithms.

summary: Stabilizing policy optimization when off-policy data are reused, addressing the data efficiency issue in RL for real-world problems.
---
