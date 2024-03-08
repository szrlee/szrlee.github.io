---
title: 'Prior-dependent analysis of posterior sampling reinforcement learning with function approximation'
authors:
- Yingru Li
- Zhi-Quan Luo
date: '2024-01-01'
publishDate: '2024-02-21T16:25:24.158149Z'
publication_types:
- paper-conference
publication: '*The 27th International Conference on Artificial Intelligence and Statistics (AISTATS)*'

tags:
- Reinforcement Learning
- Regret
- Probability
- Efficiency

abstract: This work advances randomized exploration in reinforcement learning (RL) with function approximation modeled by linear mixture MDPs. We establish the first prior-dependent Bayesian regret bound for RL with function approximation; and refine the Bayesian regret analysis for posterior sampling reinforcement learning (PSRL), presenting an upper bound of $\tilde{\mathcal{O}}(d\sqrt{H^3 T \log T})$, where $d$ represents the dimensionality of the transition kernel, $H$ the planning horizon, and $T$ the total number of interactions. This signifies a methodological enhancement by optimizing the $\mathcal{O}(\sqrt{\log T})$ factor over the previous benchmark (Osband and Van Roy, 2014) specified to linear mixture MDPs. Our approach, leveraging a value-targeted model learning perspective, introduces a decoupling argument and a variance reduction technique, moving beyond traditional analyses reliant on confidence sets and concentration inequalities to formalize Bayesian regret bounds more effectively.

summary: This paper delves into the Bayesian regret of posterior sampling reinforcement learning (PSRL) and presents a novel prior-dependent regret bound within the linear mixture model. The bound hinges on the variance of the underlying MDP in the prior distribution, offering a distinctive perspective in the realm of randomized exploration.
---
