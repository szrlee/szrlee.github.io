---
title: "Keeping the Off-Policy Gap Fixed as RL Scales"
subtitle: "Off-Policy RL at Scale, Part 2: a batch-size-invariant rule for the step size and the queue, where it stops, and what to measure"
summary: "Asynchronous RL buys throughput with fast kernels, a queue and larger batches, and pays for it in the gap between sampler and trainer. Model the gap as an engine-mismatch floor plus staleness, put everything on a data clock, and two settings keep it fixed as the batch grows: the step size follows the optimizer, and the queue is sized in rollouts."
authors:
  - admin
tags:
  - Reinforcement Learning
  - Off-Policy
  - Scaling
  - Optimization
  - Training Dynamics
categories:
  - Research
  - Theory
date: '2026-10-13T00:10:00Z'
lastmod: '2026-10-13T00:10:00Z'
math: true
draft: false
series:
  - Off-Policy RL at Scale
series_order: 2
---

{{% callout note %}}
**Off-Policy RL at Scale · Part 2 of 2.** [Part 1](/post/offpolicy-around-the-sampler/) showed that the update is a tangent plus a remainder bounded by the sequence KL. This part asks how large that gap is and how to keep it fixed as the run grows. It builds on [When Speed Kills Stability](/rl-collapse) and [Beyond Precision](https://arxiv.org/abs/2602.01826).

- The gap is an engine-mismatch **floor** plus **staleness** that grows with the age of the data. Both coefficients are local.
- Compare runs on a **data clock**, after the same number of rollouts. Batch-size invariance means three quantities stay fixed: progress per rollout, noise per rollout, and the age of every token in rollouts.
- Two settings keep them fixed: the **step size follows the optimizer** (×κ for SGD, ×√κ for noise-dominated Adam), and the **queue is sized in rollouts**.
- Anything counted in updates, such as the momentum window, breaks the rule. The rules give the shapes; the constants must be measured.
{{% /callout %}}

The figures step through like slides. Use **→** or **▶**, and step back with **←**.

## 1. Where we are

Part 1 fixed the form of a good off-policy update. Its quality depends on the size of the gap between {{< math >}}$\ps${{< /math >}} and {{< math >}}$\pt${{< /math >}}: the remainder is bounded by the sequence KL, and a mask threshold has to sit above its floor. This part models that gap and finds the settings that keep it fixed when a run is scaled.

## 2. The gap, and how much staleness it can afford

**The gap is a floor plus staleness, and only the room above the floor can be spent on staleness.**

{{< stepfig "gap" >}}

Every token carries an age {{< math >}}$a${{< /math >}}: the number of trainer updates since the weights that drew it. The token KL between sampler and trainer at age {{< math >}}$a${{< /math >}} is

{{< math >}}
$$
\mathrm{KL}(a) \;\approx\; \underbrace{c_0(\theta)}_{\text{floor}} + \underbrace{c_1(\theta)\,\eta^2 a^{p}}_{\text{staleness}}, \qquad \overline{\mathrm{KL}}(a) \approx \overline{\mathrm{KL}}(0) + \tfrac12\,\eta^2\sigma_u^2\,a \ \text{ for a random walk } (p=1).
$$
{{< /math >}}

Updates with a steady direction give {{< math >}}$p=2${{< /math >}}. Whatever KL the run can tolerate, the headroom above the floor sets the affordable mean age. Because the mean age is {{< math >}}$\bar a = \chi N/B${{< /math >}}, the depth {{< math >}}$N${{< /math >}} of the queue is coupled to both the batch size and the step size:

{{< math >}}
$$
\bar a \le \frac{2\cdot\text{headroom}}{\eta^2\sigma_u^2}, \qquad N \le \frac{B}{\chi}\,\bar a_{\max} \;\propto\; \frac{B}{\eta^2}.
$$
{{< /math >}}

**Both coefficients are local.** In a fully on-policy run, which has no staleness, the KL between the engines is the floor by itself. It sits near zero, then spikes as the policy moves, and the largest spike coincides with an entropy jump and a reward collapse:

<figure class="post-fig">
<img src="/figs/img/mismatch-kl-entropy.png" alt="A fully on-policy run: the KL between the two engines spikes with jumps in entropy, and the reward collapses at the largest spike.">
<figcaption>Blue: entropy (left) and reward (right). Red dashed: the KL between the two engines at the same weights. From <a href="/rl-collapse">When Speed Kills Stability</a>, Fig. 2.</figcaption>
</figure>

The floor also responds to the size of the update ([Beyond Precision](https://arxiv.org/abs/2602.01826)), so it is part of the optimization, not a fixed numerical error.

## 3. The data clock

**Compare runs after the same number of rollouts. Then two settings keep three invariants.**

{{< stepfig "clock" >}}

Grow the batch by {{< math >}}$\kappa${{< /math >}}. With {{< math >}}$m_u${{< /math >}} and {{< math >}}$C_u${{< /math >}} the mean and covariance of one update before the learning rate, the first two invariants are per rollout:

{{< math >}}
$$
\mathcal{A}_B = \frac{\eta\, m_u(B)}{B}\ \ (\text{progress}), \qquad \mathcal{D}_B = \frac{\eta^2 C_u(B)}{B}\ \ (\text{noise}), \qquad \text{drift} \approx \tfrac12\Big(\mathbb{E}[S^2]\,\|\mathcal{A}_B\|_F^2 + \mathbb{E}[S]\,\mathrm{tr}\big(F\mathcal{D}_B\big)\Big).
$$
{{< /math >}}

The drift, which is the staleness term of the gap in KL units, depends only on these two and on the age {{< math >}}$S${{< /math >}} of each token *in rollouts*. That age is the third invariant. Here {{< math >}}$F${{< /math >}} is the Fisher metric, the policy's local sensitivity. A queue configured in updates is {{< math >}}$\kappa${{< /math >}} times more rollouts deep at the larger batch, so it breaks this invariant. A queue configured in rollouts keeps it:

| Setting | Rule when {{< math >}}$B \to \kappa B${{< /math >}} | Keeps |
|---|---|---|
| Learning rate | {{< math >}}$\eta \to \kappa\eta${{< /math >}} (SGD) or {{< math >}}$\sqrt{\kappa}\,\eta${{< /math >}} (noise-dominated Adam) | Progress and noise per rollout |
| Queue | Fixed at {{< math >}}$N${{< /math >}} rollouts; prefetch depth in updates {{< math >}}$\div\,\kappa${{< /math >}} | The age of every token, in rollouts |

The exponent of the learning-rate rule depends on the optimizer, not on the queue:

{{< stepfig "opt" >}}

**Where the rule stops.** Anything else counted in updates changes meaning when the batch grows. The momentum window spans {{< math >}}$B/(1-\beta_1)${{< /math >}} rollouts, so the rule needs {{< math >}}$N \gg \kappa B/(1-\beta_1)${{< /math >}}. Otherwise the updates a token waits through all point the same way and the drift grows like the square of its age. The fix is to keep that window fixed in rollouts as well: {{< math >}}$\beta_1 \to \beta_1^{\kappa}${{< /math >}}. Coordinates dominated by signal rather than noise ask for the linear rule even under Adam, so the invariants hold on average and no better.

## 4. The pull at scale

**A KL pull toward the sampler is a pull toward the trainer's own past, and an age weight can make it equal at every age.**

{{< stepfig "pull" >}}

Without a weight, the pull grows like the square root of the drift, so the oldest and least trustworthy tokens pull hardest. Weighting each token's KL by a power of its age flattens the pull exactly when the power is half the drift exponent. And because the weight counts age in updates, {{< math >}}$\beta${{< /math >}} has to follow the batch:

{{< math >}}
$$
w(a) = a^{-\rho},\quad \rho = p/2; \qquad \beta(\kappa B) = \kappa^{-\rho}\,\beta(B).
$$
{{< /math >}}

With no age weight, {{< math >}}$\rho = 0${{< /math >}} and {{< math >}}$\beta${{< /math >}} stays fixed. The pull is delayed feedback toward past weights. It stays stable while the step size times {{< math >}}$\beta${{< /math >}} times the KL curvature is well below one, and that margin shrinks as the batch, and with it the step size, grows.

## 5. What is derived, and what is measured

The analysis derives the *shapes*: the form of the update, the form of the gap, and how the step size, the queue, the momentum window and {{< math >}}$\beta${{< /math >}} move together. The *constants* (the floor, the staleness coefficient and the exponent {{< math >}}$p${{< /math >}}) are local, so they have to be measured along every run with cheap gauges:

| Gauge | What it tells you |
|---|---|
| Sampler–trainer KL, by token age | The floor, and the staleness slope and exponent |
| Age histogram, in rollouts | Whether the queue really holds the age fixed |
| Update mean and covariance at two batch sizes | Whether the learning-rate rule keeps progress and noise per rollout |
| Gradient on fresh against stale data | Whether staleness has started to bend the update |

The same gauges tell you when a run has left the regime where the rule holds.

## References

- J. Liu, Y. Li, et al. [When Speed Kills Stability: Demystifying RL Collapse from the Training-Inference Mismatch](/rl-collapse). Blog, 2025.
- Y. Zhang, Y. Li, et al. [Beyond Precision: Training-Inference Mismatch is an Optimization Problem and Simple LR Scheduling Fixes It](https://arxiv.org/abs/2602.01826). arXiv, 2026.
- Y. Li, J. Liu, et al. [Trust Region Masking for Long-Horizon LLM Reinforcement Learning](https://arxiv.org/abs/2512.23075). ICML 2026.
- P. Goyal, P. Dollár, R. Girshick, et al. [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677). 2017.
- S. Malladi, K. Lyu, A. Panigrahi, S. Arora. [On the SDEs and Scaling Rules for Adaptive Gradient Algorithms](https://arxiv.org/abs/2205.10287). NeurIPS 2022.

## Citation

```bibtex
@online{li-2026-offpolicy-gap-at-scale,
  title  = {Keeping the Off-Policy Gap Fixed as RL Scales},
  author = {Li, Yingru},
  year   = {2026},
  month  = oct,
  url    = {https://richardli.xyz/post/offpolicy-gap-at-scale/}
}
```
