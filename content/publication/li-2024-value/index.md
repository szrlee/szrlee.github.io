---
title: 'A value-targeted analysis of posterior sampling reinforcement learning with
  linear function approximation '
authors:
- Yingru Li
- Zhi-Quan Luo
date: '2024-01-01'
publishDate: '2024-02-21T16:25:24.158149Z'
publication_types:
- paper-conference
publication: '*The 27th International Conference on Artificial Intelligence and Statistics
  (AISTATS)*'

tags:
- Reinforcement Learning
- Regret
- Probability
- Efficiency

abstract: We study model-based reinforcement learning (RL) with function approximation and randomized exploration. We focus on finitehorizon episodic RL where the transition probability kernel is a linear mixture model (Ayoub et al, 2020). Under this setting, we establish an $\tilde{\mathcal{O}}\left(d \sqrt{H^3 T \log T}\right)$ Bayesian regret bound for posterior sampling reinforcement learning (PSRL), where $d$ is the ambient dimension of the transition kernel, $H$ is the planning horizon, and $T$ is the number of total interactions ( \# episodes $\times \#$ horizon ). We provide the first prior-dependent Bayesian regret bound for RL with function approximation. We also provide a priorfree Bayesian regret bound. Specifically, our upper bound improves a $\mathcal{O}(\sqrt{\log T})$ factor over previous best known Bayesian regret bound for PSRL (Osband and Van Roy, 2014:) when specified to linear mixture MDPs. To highlight, our analysis is built upon a value-targeted perspective of model learning. To formalise the Bayesian regret bound, we develop a decoupling argument and a variance reduction argument instead of any construction of confidence sets and concentration inequalities.

summary: New theoretical framework for analyzing Bayesian regret in RL with function approximation, providing prior-dependent and prior-free results.
---
