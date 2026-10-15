---
title: "Why d log T Particles Are Enough, and What Carries Over to Networks"
subtitle: "Exploration at Scale, Part 2: the cloud as a random projection, a Johnson–Lindenstrauss lemma for adaptive data, and the same update as an SGD step"
summary: "The particles of Ensemble++ are a random projection of the data history, so tracking the posterior is a Johnson–Lindenstrauss statement. When the agent picks its own data the classical lemma fails; a sequential version survives at the price of one log T, so d log T particles suffice. In a network the same update is an SGD step on M linear heads."
authors:
  - admin
tags:
  - Reinforcement Learning
  - Exploration
  - Thompson Sampling
  - Random Projection
  - Deep Learning
categories:
  - Research
  - Theory
date: '2026-10-13T00:30:00Z'
lastmod: '2026-10-13T00:30:00Z'
math: true
draft: false
series:
  - Exploration at Scale
series_order: 2
---

{{% callout note %}}
**Exploration at Scale · Part 2 of 2.** [Part 1](/post/scalable-exploration/) built Thompson sampling from a cloud of M particles and a random index. Based on [Ensemble++ (NeurIPS 2025)](https://arxiv.org/abs/2407.13195), [HyperAgent (ICML 2024)](https://arxiv.org/abs/2402.10228) and [Probability Tools for Sequential Random Projection](https://arxiv.org/abs/2402.14026).

- The particles are a **random projection of the data history**, so tracking the posterior is a Johnson–Lindenstrauss statement.
- When the agent **picks its own data**, the noise reaches the inputs and the classical lemma no longer applies.
- A **sequential Johnson–Lindenstrauss lemma** survives adaptivity at the price of one log T, so **d log T particles** match Thompson sampling, against (actions × T) before.
- In a network the same update is **an SGD step on M linear heads**: architecture and optimizer designed together.
{{% /callout %}}

The figures step through like slides. Use **→** or **▶**, and step back with **←**.

## 1. The particles are a random projection of the history

**Each particle is the whole data history weighted by one random direction, so M particles are M random projections of it.**

{{< stepfig "proj" >}}

Write the history as {{< math >}}$\Phi_t${{< /math >}} (the prior, then one column per input) and the injected noise as {{< math >}}$\Pi_t${{< /math >}} (the prior draws, then one row per data point). The arrows of the cloud are their product, and its covariance is

{{< math >}}
$$
\mathbf{A}_t = \Sigma_t\,\Phi_t\,\Pi_t, \qquad \mathbf{A}_t\mathbf{A}_t^{\top} = \Sigma_t\,\Phi_t\big(\Pi_t\Pi_t^{\top}\big)\Phi_t^{\top}\,\Sigma_t \;=\; \Sigma_t \quad\text{if } \Pi_t \text{ preserves norms on the history.}
$$
{{< /math >}}

A new data point appends one column and one row, so the product changes by a rank-one term and never grows. The injected noise is a Johnson–Lindenstrauss projection of the history from {{< math >}}$d+t${{< /math >}} dimensions down to M, maintained as a stream.

## 2. Once the agent picks its own data, the classical lemma fails

**The noise is drawn once and never redrawn, which keeps the update cheap, but it also means the old noise steers the inputs the agent chooses next.**

{{< stepfig "adaptive" >}}

The classical lemma needs the vectors to be fixed before the projection is drawn. Here each new column of the history is chosen after the agent has seen earlier rows of the noise. That dependence is also why the only earlier analysis ([Qin et al., 2022](https://arxiv.org/abs/2203.01303)) needed an ensemble linear in the horizon times the number of actions. Practitioners had been using 10 to 100 members for a decade without an explanation.

## 3. Sequential Johnson–Lindenstrauss: adaptivity costs one log T

**Norms are preserved for adaptively chosen sequences, uniformly over time, with one extra logarithm of the horizon.**

Let {{< math >}}$\mathbf{z}_i${{< /math >}} be the rows of the injected noise and {{< math >}}$v${{< /math >}} a vector of the history whose entries may depend on all earlier rows. Then, with probability at least {{< math >}}$1-\delta${{< /math >}},

{{< math >}}
$$
M \ge \frac{C}{\varepsilon^2}\Big(\log\frac1\delta + \log T\Big) \;\Longrightarrow\; \forall\, t \le T:\quad (1-\varepsilon)\sum_{i\le t} v_i^2 \;\le\; \Big\|\sum_{i\le t} v_i\,\mathbf{z}_i\Big\|^2 \;\le\; (1+\varepsilon)\sum_{i\le t} v_i^2 .
$$
{{< /math >}}

Independence cannot be used, so the proof runs through a stopped process and self-normalized martingale concentration. For the cloud, the result reads

{{< math >}}
$$
M \ge C\,d\log T \;\Longrightarrow\; \tfrac12\,\Sigma_t \preceq \mathbf{A}_t\mathbf{A}_t^{\top} \preceq \tfrac32\,\Sigma_t \qquad \forall\, t \le T .
$$
{{< /math >}}

<figure class="post-fig">
<img src="/figs/img/min-M-vs-d.png" alt="Minimum number of particles needed to match Thompson sampling against the dimension d, for two sizes of the action set: the points lie on a straight line.">
<figcaption>The minimum M that matches Thompson sampling grows linearly in the dimension d. From the Ensemble++ paper.</figcaption>
</figure>

With M of order *d* log *T*, the regret is comparable to exact Thompson sampling for compact or finite action sets, fixed or time-varying. Memory grows with the logarithm of the lifetime: doubling the horizon adds a number of particles proportional to *d*, however long the agent has already run.

## 4. In a network the same update is an SGD step

**The exact gain of the linear model becomes a learning rate, and each particle becomes a linear head trained on its own noisy targets.**

{{< stepfig "sgd" >}}

The posterior sampler is trained by the same stochastic updates as the model, on the same data stream, at the cost of a few extra linear heads. Seen from the network's side, the model has a second input:

{{< stepfig "hyper" >}}

Neither half works alone. Without the noisy targets, a network learns to ignore the index. Without the index, the noisy targets give one perturbed model, and the next sample needs a full retraining.

| | Architecture | Optimization |
|---|---|---|
| Design | A second input, the random index; output linear in it (a mean head plus M index heads), each with a fixed random prior | Every head's target carries its own injected noise, drawn once per data point; minibatch SGD from a bounded FIFO buffer |
| Cost | Memory: M extra linear heads and a bounded buffer, no covariance matrix | Step: a fixed number of SGD steps per interaction. Sample: one forward pass with a fresh index |

Where the index should enter (the last layer, deeper layers, adapters on a large model) is a design choice that is still largely unexplored.

## 5. Results: less data, fewer parameters

**Directed exploration buys the data efficiency, and a few linear heads keep the compute bounded.**

<figure class="post-fig">
<img src="/figs/img/ensemble-regret-vs-parameters.png" alt="Cumulative regret against number of parameters on a nonlinear bandit: Ensemble++ is lower and to the left of Ensemble+ and EpiNet.">
<figcaption>Nonlinear bandit: regret against the number of parameters, which is the cost of every step. Ensemble++ reaches lower regret with fewer parameters than Ensemble+ and EpiNet.</figcaption>
</figure>

<figure class="post-fig">
<img src="/figs/img/ensemble-size-vs-actions.png" alt="Regret for 8, 32 and 128 particles as the action set grows: more particles, lower regret, nearly flat in the number of actions.">
<figcaption>More particles, lower regret, and the regret barely moves as the action set grows sixfold, as the theory predicts.</figcaption>
</figure>

<figure class="post-fig">
<img src="/figs/img/content-moderation.png" alt="Online content moderation with a GPT-2 backbone: Ensemble++ learns faster and more reliably than greedy or Ensemble+.">
<figcaption>A GPT-2 backbone with an Ensemble++ head, deciding online which posts to send to human review: faster and steadier than greedy, and 5% above Ensemble+.</figcaption>
</figure>

In deep RL, [HyperAgent](https://arxiv.org/abs/2402.10228) adds the same heads to DQN:

<figure class="post-fig">
<img src="/figs/img/hyperagent-data-vs-parameters.png" alt="Atari: interaction data needed for human-level performance against number of parameters; HyperAgent sits in the corner with little data and few parameters.">
<figcaption>Atari: human-level performance from 1.5M interactions, 15% of DDQN's interaction data, with few parameters. From the HyperAgent paper.</figcaption>
</figure>

<figure class="post-fig">
<img src="/figs/img/hyperagent-deepsea.png" alt="DeepSea: episodes needed to solve the task grow linearly with its size for HyperAgent.">
<figcaption>DeepSea, a hard-exploration benchmark: the episodes HyperAgent needs grow linearly with the size of the problem, the optimal rate.</figcaption>
</figure>

## References

- Y. Li, J. Xu, B. Wang, Z.-Q. Luo. [Scalable Exploration via Ensemble++](https://arxiv.org/abs/2407.13195). NeurIPS 2025.
- Y. Li, J. Xu, L. Han, Z.-Q. Luo. [Q-Star Meets Scalable Posterior Sampling: Bridging Theory and Practice via HyperAgent](https://arxiv.org/abs/2402.10228). ICML 2024.
- Y. Li. [Probability Tools for Sequential Random Projection](https://arxiv.org/abs/2402.14026). arXiv, 2024.
- Z. Li, Y. Li, Y. Zhang, T. Zhang, Z.-Q. Luo. [HyperDQN: A Randomized Exploration Method for Deep Reinforcement Learning](https://openreview.net/forum?id=X0nrKAXu7g-). ICLR 2022.
- C. Qin, Z. Wen, X. Lu, B. Van Roy. [An Analysis of Ensemble Sampling](https://arxiv.org/abs/2203.01303). NeurIPS 2022.

## Citation

```bibtex
@inproceedings{li2024hyperagent,
  title     = {Q-Star Meets Scalable Posterior Sampling: Bridging Theory and Practice via HyperAgent},
  author    = {Li, Yingru and Xu, Jiawei and Han, Lei and Luo, Zhi-Quan},
  booktitle = {International Conference on Machine Learning},
  year      = {2024}
}
```
