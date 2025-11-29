---
title: 'Scalable Exploration via Ensemble++'
subtitle: 'Efficient Thompson Sampling for Deep Reinforcement Learning'
summary: 'Ensemble++ achieves Thompson Sampling-level exploration with only O(d log T) ensemble directions, enabling scalable uncertainty quantification for neural bandits and beyond.'
authors:
  - admin
tags:
  - Reinforcement Learning
  - Exploration
  - Thompson Sampling
  - Deep Learning
  - Bandits
categories:
  - Research
date: '2025-11-29T00:00:00Z'
lastmod: '2025-11-29T00:00:00Z'
featured: true
draft: false
math: true
toc: true

# Featured image
image:
  caption: 'Ensemble++ Poster'
  focal_point: ''
  preview_only: false

# Projects (optional)
projects: []
---

## The Exploration Challenge in Deep RL

Thompson Sampling is one of the most elegant algorithms for balancing exploration and exploitation in sequential decision-making. The idea is simple: maintain a posterior distribution over reward functions, sample from it, and act optimally with respect to the sample. This approach naturally trades off exploring uncertain regions against exploiting known good actions.

But there's a catch: **exact Thompson Sampling is computationally intractable** for modern deep learning systems. Neural networks don't admit conjugate posteriors, and maintaining full uncertainty quantification over millions of parameters is simply infeasible.

---

## The Ensemble Approximation Gap

Prior work attempted to approximate Thompson Sampling using ensembles—train multiple neural networks and use their disagreement as a proxy for uncertainty. The problem? **The required ensemble size scales prohibitively**:

- Theoretical guarantees require $\Omega(T \cdot |\mathcal{X}|)$ ensemble members
- For practical horizons and action spaces, this means thousands to millions of networks
- Training and inference costs become unmanageable

This creates a fundamental tension: we want the exploration benefits of Thompson Sampling but cannot afford its computational cost.

---

## Ensemble++: The Key Innovation

**Ensemble++** resolves this tension through a clever insight: instead of maintaining independent ensemble members, maintain a **shared ensemble matrix factor** that updates incrementally using random linear combinations.

### Core Mechanism

The algorithm maintains matrices that incrementally approximate the posterior covariance. At each step:

1. **Sample**: Draw a reference vector to form action-selection hypotheses via linear combination
2. **Observe**: Collect reward feedback from the environment
3. **Update**: Modify the ensemble factor using perturbation vectors

This enables approximate posterior sampling **without full matrix decomposition or expensive per-step retraining**.

### The Symmetrized Ridge-Regression Perspective

A unified formulation expresses both linear and nonlinear variants through a two-sided perturbation loss:

$$\mathcal{L}(\theta) = \sum_{t=1}^{n} \left( y_t - f_\theta(x_t) \right)^2 + \lambda \|\theta\|^2 + \text{perturbation terms}$$

For linear features, this admits closed-form updates. For neural networks, we optimize via SGD while preserving the incremental update mechanics.

---

## Theoretical Guarantees

The theoretical contribution is substantial. For linear bandits, Ensemble++ achieves:

| Setting | Regret Bound |
|---------|-------------|
| Compact action sets | $\mathcal{O}(d^{3/2}\sqrt{T}(\log T)^{3/2})$ |
| Finite action sets | $\mathcal{O}(d\sqrt{T \log|\mathcal{X}|} \log T)$ |

These bounds are **comparable to exact Thompson Sampling** while requiring only:

- **Ensemble size**: $\Theta(d \log T)$ directions (vs. $\Omega(T \cdot |\mathcal{X}|)$ for prior methods)
- **Per-step complexity**: $\mathcal{O}(d^2)$ for linear case
- **Storage**: $\Theta(d \log T)$ ensemble directions

The approach works **uniformly across compact and finite action sets** without algorithmic reconfiguration—a notable improvement over prior methods that required problem-specific tuning.

---

## Extension to Neural Networks

The same principle extends to nonlinear settings by replacing fixed linear features with **learnable neural representations**:

1. Extract features from the last layer of a neural network
2. Apply the Ensemble++ update rule to these features
3. Propagate gradients through the full network via SGD

This bridges the gap between principled Bayesian exploration and practical deep RL systems.

---

## Empirical Results

Comprehensive experiments validate the theoretical claims across multiple settings:

### Linear Bandits
Ensemble++ matches Thompson Sampling performance with a fraction of the ensemble size, while baselines like Ensemble+ and EpiNet require significantly larger ensembles for comparable performance.

### Quadratic Bandits
Even in non-linear settings, Ensemble++ maintains strong performance, demonstrating the robustness of the incremental update mechanism.

### Neural Bandits
With neural network function approximation, Ensemble++ achieves superior regret-versus-computation trade-offs compared to existing ensemble methods.

### GPT-Based Bandits
Scaling to large language models, Ensemble++ enables efficient exploration in high-dimensional representation spaces where traditional methods become prohibitively expensive.

---

## Why This Matters

The exploration-exploitation trade-off is fundamental to sequential decision-making. Ensemble++ makes principled exploration **practical at scale** by:

1. **Reducing computational burden**: From exponential to logarithmic ensemble scaling
2. **Preserving theoretical guarantees**: Near-optimal regret bounds
3. **Enabling modularity**: Plug-and-play integration with existing neural architectures
4. **Unifying settings**: Same algorithm works across action space types

For practitioners building agents that must learn efficiently in uncertain environments—from recommendation systems to robotic control to LLM-based agents—Ensemble++ offers a principled yet practical path forward.

---

## Poster

**[TBD]**

---

## Paper Details

**Title**: Scalable Exploration via Ensemble++

**Authors**: Yingru Li, Jiawei Xu, Baoxiang Wang, Zhi-Quan Luo

**Venue**: NeurIPS 2025

**Links**:
- [arXiv](https://arxiv.org/abs/2407.13195)

---

## Citation

```bibtex
@inproceedings{li2025scalable,
  title     = {Scalable Exploration via Ensemble++},
  author    = {Li, Yingru and Xu, Jiawei and Wang, Baoxiang and Luo, Zhi-Quan},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2025}
}
```

---

*Last updated: November 30, 2025*
