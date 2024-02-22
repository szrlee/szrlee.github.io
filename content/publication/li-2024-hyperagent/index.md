---
title: 'HyperAgent: A Simple, Scalable, Efficient and Provable Reinforcement Learning
  Framework for Complex Environments'
authors:
- Yingru Li
- Jiawei Xu
- Lei Han
- Zhiquan Luo
date: '2024-01-01'
publishDate: '2024-02-21T16:25:24.139610Z'
publication_types:
- paper-conference
publication: '*The 41st International Conference on Machine Learning (ICML)* (Submitted)'
links:
- name: arXiv
  url: https://arxiv.org/abs/2402.10228

featured: true
abstract: To solve complex tasks under resource constraints, reinforcement learning (RL) agents need to be simple, efficient, and scalable with (1) large state space and (2) increasingly accumulated data of interactions. We propose the HyperAgent, a RL framework with hypermodel, index sampling schemes and incremental update mechanism, enabling computation-efficient sequential posterior approximation and data-efficient action selection under general value function approximation beyond conjugacy. The implementation of HyperAgent is simple as it only adds one module and one line of code additional to DDQN. Practically, HyperAgent demonstrates its robust performance in large-scale deep RL benchmarks with significant efficiency gain in terms of both data and computation. Theoretically, among the practically scalable algorithms, HyperAgent is the first method to achieve provably scalable per-step computational complexity as well as sublinear regret under tabular RL. The core of our theoretical analysis is the sequential posterior approximation argument, made possible by the first analytical tool for sequential random projection, a non-trivial martingale extension of the Johnson-Lindenstrauss lemma. This work bridges the theoretical and practical realms of RL, establishing a new benchmark for RL algorithm design.

# Summary. An optional shortened abstract.
summary: Addressing data and computation efficiency challenge in RL for real-world problems with theoretical advancements and practical algorithm designs.(1) Theoretically, we demonstrate exponential improvements in per-step computation complexity; (2) Empirically, HyperAgent has significant practical efficiency gains, particularly in deep RL benchmarks.

projects:
  - Efficient RL
---

### Summary of contributions:

1. New probability tools in high-dimensional probability and statistics
- a non-trivial martingale extension of Johnson-Lindenstrauss (JL) for adaptively sampled data due to the sequential nature of RL;
- a unified and simple analysis for JL via high-dimension extension of Hanson-Wright.

2. Methodology for sequential-decision making
- hypermodel for efficient incremental approximation of the posterior (uncertainty estimation) over complex models without leveraging conjugacy as encountering more data;
- index sampling for approximate posterior sampling for data-efficient sequential decision-making.

3. Results for sequential-decision making
- Practically, our developed HyperAgent demonstrates its robust performance in large-scale deep RL benchmarks with significant efficiency gain in terms of both data and computation;
- Theoretically, first method to achieve logarithmic per-step computation and sublinear under episodic tabular RL among practically scalable algorithms.
