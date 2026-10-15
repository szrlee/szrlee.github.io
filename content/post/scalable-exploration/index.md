---
title: "Thompson Sampling for the Price of an SGD Step"
subtitle: "Exploration at Scale, Part 1: a random index, a cloud of particles, and posterior samples from incremental updates"
summary: "Thompson sampling explores where plausible models disagree, but needs a fresh posterior sample every round. Build the randomness into the model instead: a random index as a second input, and a cloud of particles that moves with every data point. A sample becomes one forward pass, and the update is ordinary incremental learning."
authors:
  - admin
tags:
  - Reinforcement Learning
  - Exploration
  - Thompson Sampling
  - Bandits
  - Uncertainty
categories:
  - Research
date: '2026-10-13T00:20:00Z'
lastmod: '2026-10-13T00:20:00Z'
featured: true
draft: false
math: true
series:
  - Exploration at Scale
series_order: 1
---

{{% callout note %}}
**Exploration at Scale · Part 1 of 2.** [Part 2](/post/exploration-d-log-t/) shows why *d* log *T* particles are enough and what carries over to neural networks. Based on [Ensemble++ (NeurIPS 2025)](https://arxiv.org/abs/2407.13195) and [HyperAgent (ICML 2024)](https://arxiv.org/abs/2402.10228).

- The goal is **compute-scalable exploration for data-efficient learning**: explore well enough to need less data, at a cost per step that stays fixed as the data grows.
- Thompson sampling sends the learner where plausible models **disagree**, and stops exploring on its own once they agree.
- Its price is a **fresh posterior sample every round**, which needs a matrix square root even in a linear model and has no closed form in a network.
- Build the randomness into the model: a **random index** as a second input, so that a sample is one forward pass.
- In a linear model the index weights **M stored particles**, and one incremental step per data point keeps the cloud on the posterior. Memory is the particles, nothing else.
{{% /callout %}}

The figures step through like slides. Use **→** or **▶**, and step back with **←**.

## 1. Thompson sampling turns uncertainty into exploration

**Sample one plausible model, act on it, observe, update: the agent goes where the models disagree, as often as they say it might pay off.**

{{< stepfig "ts" >}}

A greedy learner plays the best option of the average model and stays at the peak it already knows. Undirected noise, such as ε-greedy, a sampling temperature or an entropy bonus, spends trials everywhere, including on options already known to be bad, and keeps paying for them for as long as it runs. Thompson sampling needed no rule to stop exploring: as the posterior narrowed, the samples stopped disagreeing. Over 200 runs it earns 1.15 per round, against 0.79 for ε-greedy and 0.68 for greedy.

## 2. With a model, the posterior sample becomes the bottleneck

**Every round changes the posterior, so every round needs a new sample, and that sample is the expensive step.**

| Step | Linear model | Neural network |
|---|---|---|
| 1 · Sample one plausible model | Closed form, but each sample needs a matrix square root, cubic in the dimension | No closed-form posterior to sample from |
| 2 · Act on it | Best option under the sample | Best option under the sample |
| 3 · Observe | A noisy response | A noisy response |
| 4 · Update | One recursive step per data point | One SGD step per data point |

The update is cheap in both cases; the sample is not. What I want is **a posterior sample for the price of one update step**.

## 3. The idea: a model that maps simple noise to posterior samples

**Give the model a second input, a random index ξ, and train it so that simple noise is pushed forward onto the posterior.**

{{< stepfig "index" >}}

The model is {{< math >}}$f_\nu(x, \xi)${{< /math >}}, with one set of weights {{< math >}}$\nu${{< /math >}} for every index. Sampling then costs one forward pass with a fresh index, and one update step moves every sample at once. In a linear model, where the exact posterior is known and every step can be checked, each sample is a point in parameter space:

{{< stepfig "params" >}}

{{< math >}}
$$
\tilde\theta(\xi) = \mu + \mathbf{A}\,\xi, \qquad \xi \sim N(0, I_M), \qquad f_\nu(x,\xi) = \big\langle \phi(x),\ \tilde\theta(\xi)\big\rangle, \qquad \nu = (\mu, \mathbf{A}).
$$
{{< /math >}}

The weights are a mean and M stored directions. The rest of this post builds them.

## 4. A cloud of particles moves with every data point

**Store M particles; one cheap step per data point, with noise we inject ourselves, keeps the cloud on the exact posterior.**

{{< stepfig "particles" >}}

Two moves make it work. **Sampling is a random combination:** a Gaussian index weighting M stored arrows is exactly Gaussian, so the only thing to get right is the covariance of the cloud. **Updating is a perturbed fit:** each particle regresses on its own noisy copy of every response, anchored at its own prior draw:

{{< math >}}
$$
\tilde\theta_{t,m} = \arg\min_{\theta}\ \sum_{s\le t}\Big(Y_s + Z_{s,m} - \langle\theta, X_s\rangle\Big)^2 + \big\|\theta - \tilde\theta_{0,m}\big\|^2_{\Sigma_0^{-1}},
\qquad Z_{s,m}\sim N(0,1),\quad \tilde\theta_{0,m}\sim N(\mu_0,\Sigma_0).
$$
{{< /math >}}

When a data point arrives, the objective gains one term, so the update is one recursive step. Nothing is reweighted or resampled, which avoids the weight degeneracy of a particle filter (the idea goes back to the ensemble Kalman filter), and no data point is stored.

<details class="spoiler">
<summary>Why each particle is an exact posterior sample when the data are fixed</summary>

The minimizer is linear in the Gaussian perturbations {{< math >}}$Z_{s,m}${{< /math >}} and {{< math >}}$\tilde\theta_{0,m}${{< /math >}}, so it is Gaussian. Its mean is the ridge solution, which is the posterior mean {{< math >}}$\mu_t${{< /math >}}. Its covariance is {{< math >}}$\Sigma_t\big(\Sigma_0^{-1} + \sum_s X_s X_s^{\top}\big)\Sigma_t = \Sigma_t${{< /math >}}. Hence {{< math >}}$\tilde\theta_{t,m} \sim N(\mu_t, \Sigma_t)${{< /math >}}, and the cloud is M posterior draws.

</details>

## 5. At any time, the same recipe gives a fresh sample

**Mean plus stored arrows times a random index: a posterior sample with no square root anywhere.**

{{< stepfig "sample" >}}

{{< math >}}
$$
\tilde\theta(\xi) = \mu_t + \mathbf{A}_t\,\xi = \mu_t + \tfrac{1}{\sqrt M}\sum_{m}\xi_m\big(\tilde\theta_{t,m}-\mu_t\big), \qquad \text{a posterior sample if } \mathbf{A}_t\mathbf{A}_t^{\top} \approx \Sigma_t .
$$
{{< /math >}}

This is Thompson sampling at the cost of one incremental step per data point and one index per action, with memory equal to the M particles. It rests on one property: the cloud keeps the right covariance as data arrive. [Part 2](/post/exploration-d-log-t/) shows when that holds, why it is harder when the agent picks its own data, how few particles it takes (*d* log *T*), and what carries over to a network.

## References

- Y. Li, J. Xu, B. Wang, Z.-Q. Luo. [Scalable Exploration via Ensemble++](https://arxiv.org/abs/2407.13195). NeurIPS 2025.
- Y. Li, J. Xu, L. Han, Z.-Q. Luo. [Q-Star Meets Scalable Posterior Sampling: Bridging Theory and Practice via HyperAgent](https://arxiv.org/abs/2402.10228). ICML 2024.
- Z. Li, Y. Li, Y. Zhang, T. Zhang, Z.-Q. Luo. [HyperDQN: A Randomized Exploration Method for Deep Reinforcement Learning](https://openreview.net/forum?id=X0nrKAXu7g-). ICLR 2022.
- D. Russo, B. Van Roy, A. Kazerouni, I. Osband, Z. Wen. [A Tutorial on Thompson Sampling](https://arxiv.org/abs/1707.02038). 2018.

## Citation

```bibtex
@inproceedings{li2025ensemble,
  title     = {Scalable Exploration via Ensemble++},
  author    = {Li, Yingru and Xu, Jiawei and Wang, Baoxiang and Luo, Zhi-Quan},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2025}
}
```
