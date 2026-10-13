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
**Series: Off-Policy RL at Scale**

- [**Part 1**](/post/offpolicy-around-the-sampler/): around the sampler. The gain of an update is a tangent plus a remainder bounded by <i>R</i><sub>max</sub> · KL<sub>seq</sub>. Methods differ only in how hard they pull toward the sampler.
- **Part 2 (this post):** how big the off-policy gap is, and how to keep it fixed as the run grows.

**Earlier public work this builds on:** [When Speed Kills Stability](/rl-collapse), [Beyond Precision](https://arxiv.org/abs/2602.01826), and [Trust Region Masking](https://arxiv.org/abs/2512.23075).
{{% /callout %}}

{{% callout note %}}
**TL;DR**

- **The gap model.** The KL between sampler and trainer, for a token of age <i>a</i>, is a **floor** plus a **staleness** term: KL(<i>a</i>) ≈ <i>c</i><sub>0</sub>(θ) + <i>c</i><sub>1</sub>(θ) η² <i>a</i><sup><i>p</i></sup>.
  - The floor comes from the mismatch between engines and is there even at identical weights.
  - Staleness grows with age.
  - Both coefficients are local: they move as the policy moves.
- **Put everything on a data clock.** Compare runs after the same number of rollouts, not the same number of updates. Batch-size invariance means three quantities stay fixed: **progress per rollout**, **noise per rollout**, and **the age of every token, in rollouts**.
- **Two settings keep the three invariants.**
  - The **step size follows the optimizer**: ×κ for SGD, ×√κ for noise-dominated Adam.
  - The **queue is sized in rollouts**, so the prefetch depth in updates comes down by κ.
- **Where the rule stops.** Anything else counted in updates, like the momentum window, changes meaning when the batch grows.
- **What is derived and what is measured.** The rules give the *shapes*. The constants are local, so they are measured along every run with cheap gauges.
{{% /callout %}}

---

## 1. Why the gap needs a model

[Part 1](/post/offpolicy-around-the-sampler/) took the gap between sampler {{< math >}}$\mu${{< /math >}} and trainer {{< math >}}$\pi${{< /math >}} as given. Its size controls everything there: the remainder is bounded by the sequence KL, a mask threshold has to sit above the floor, and a KL pull is a trade against it. This post asks how big the gap is and how it changes when a run is scaled.

Large asynchronous RL runs buy throughput with three levers, and each has a price:

| Lever | What it buys | Its price |
|---|---|---|
| Fast kernels | Samplers generate in parallel on engines built for speed | Their numbers differ from the trainer's, even at the same weights |
| Asynchrony | A queue of {{< math >}}$N${{< /math >}} rollouts; nobody waits for anybody | Every rollout is older than the weights it trains |
| Larger batch | More hardware per update: {{< math >}}$B${{< /math >}} rollouts, step size {{< math >}}$\eta${{< /math >}} | Fewer updates for the same data, so each update has to do more |

All three raise throughput, and all three are paid for in how much each update extracts. The first two prices are one thing: a gap between what samples and what trains. The third is the scaling question.

## 2. The gap: a floor plus staleness

Every token carries an **age** {{< math >}}$a${{< /math >}}: the number of trainer updates since the weights that drew it. Age comes from prefetching, from waiting in the queue, and from rollouts so long that the trainer updates several times while they are being written. It belongs to the token, so one rollout can contain several ages, and one batch mixes many. I model the token KL between sampler and trainer at age {{< math >}}$a${{< /math >}} as

{{< math >}}
$$
\mathrm{KL}(a) \;\approx\; \underbrace{c_0(\theta)}_{\text{the floor: the gap at age 0}} \;+\; \underbrace{c_1(\theta)\,\eta^2 a^{p}}_{\text{staleness: step size }\eta\text{, age }a} .
$$
{{< /math >}}

**The floor** is the engine mismatch: the same weights, two engines, different kernels and precision, so different log-probabilities. It is new with LLM RL, because classical off-policy RL only ever had different *weights*. Because of it, the KL does not start from zero, and no setting of the queue removes it. It is per token, so over a rollout of thousands of tokens it adds up. Whatever KL a run can tolerate, only the part above the floor is left to spend on staleness.

**Both coefficients are local.** The floor is not a constant of the hardware. In fully on-policy runs with fixed engines, the engine KL is near zero early, then spikes, and the spikes coincide with jumps in entropy and with collapse ([When Speed Kills Stability](/rl-collapse)). The floor also responds to the size of the update: a ten times smaller step keeps it falling where the baseline step lets it climb back ([Beyond Precision](https://arxiv.org/abs/2602.01826)). So the same numerical difference between kernels costs more KL in some regions of policy space than in others, and staleness is seen through the same local sensitivity.

**Staleness grows with age.** From any past version the weights do a random walk. The KL is, to second order, half the squared size of the log-ratio, so

{{< math >}}
$$
\overline{\mathrm{KL}}(a) \;\approx\; \overline{\mathrm{KL}}(0) + \tfrac12\,\eta^2\sigma_u^2\, a ,
$$
{{< /math >}}

with {{< math >}}$\sigma_u^2${{< /math >}} the noise energy of one update in the policy's local metric. In general I write the growing part as {{< math >}}$a^{p}${{< /math >}}:

- Random-walk updates give {{< math >}}$p=1${{< /math >}}.
- Updates with a steady direction, such as those inside one momentum window, give {{< math >}}$p=2${{< /math >}}.

The exponent is a property of the run and has to be measured.

## 3. The staleness budget

The model already gives a budget. Whatever KL the run can tolerate, the **headroom** is the room between that level and the floor. The mean age the run can afford is where the staleness line meets the ceiling:

{{< math >}}
$$
\bar a \;\le\; \bar a_{\max} = \frac{2\cdot\text{headroom}}{\eta^2\sigma_u^2} .
$$
{{< /math >}}

If the headroom is zero, no queue setting helps. A larger step size makes the line steeper, as {{< math >}}$\eta^2${{< /math >}}, so the affordable age shrinks. The mean age is the queue's depth measured in updates, {{< math >}}$\bar a = \chi N/B${{< /math >}}. Here {{< math >}}$N${{< /math >}} counts the rollouts that are sampled or in flight, and {{< math >}}$\chi${{< /math >}} depends on scheduling and has to be measured. So, at fixed update noise,

{{< math >}}
$$
N \;\le\; \frac{B}{\chi}\,\bar a_{\max} \;\propto\; \frac{B}{\eta^2} .
$$
{{< /math >}}

How deep the queue may be depends on the batch size and the step size together. That coupling is what the rest of this post resolves.

## 4. Scaling means choosing what stays invariant

What I want from scaling is **batch-size invariance**: two runs at different batch sizes, compared after the *same number of rollouts*, should be in the same place. Then a larger batch is pure throughput.

When people scale a run they are, knowingly or not, choosing what to keep invariant, so here is exactly what I keep. Grow the batch by a factor {{< math >}}$\kappa${{< /math >}}: {{< math >}}$B \to \kappa B${{< /math >}}. The larger run takes {{< math >}}$\kappa${{< /math >}} times fewer steps to consume the same data. Let {{< math >}}$m_u(B)${{< /math >}} and {{< math >}}$C_u(B)${{< /math >}} be the mean and covariance of one update before the learning rate. The first two invariants are per rollout:

{{< math >}}
$$
\mathcal{A}_B = \frac{\eta\, m_u(B)}{B}\quad\text{(progress per rollout)}, \qquad \mathcal{D}_B = \frac{\eta^2\, C_u(B)}{B}\quad\text{(noise per rollout)} .
$$
{{< /math >}}

Individual steps differ between the two runs; the process per unit of data does not. A group size or a baseline then reduces the noise per rollout, as in the [Optimal Token Baseline](https://arxiv.org/abs/2602.07078).

**Setting 1: the learning rate follows the optimizer.**

- **SGD.** A larger batch leaves the mean update where it was and shrinks its noise: {{< math >}}$m_u${{< /math >}} is unchanged and {{< math >}}$C_u \to C_u/\kappa${{< /math >}}. Then {{< math >}}$\eta \to \kappa\eta${{< /math >}} keeps both invariants. This is the linear rule of [Goyal et al. (2017)](https://arxiv.org/abs/1706.02677).
- **Adam, noise-dominated.** Adam divides the gradient by its noise level, which falls like {{< math >}}$1/\sqrt{B}${{< /math >}}. So the normalized update keeps its noise, and its mean grows like {{< math >}}$\sqrt{B}${{< /math >}}: {{< math >}}$m_u \to \sqrt{\kappa}\,m_u${{< /math >}} with {{< math >}}$C_u${{< /math >}} roughly unchanged. Then {{< math >}}$\eta \to \sqrt{\kappa}\,\eta${{< /math >}} keeps both invariants. This is the square-root rule of [Malladi et al. (2022)](https://arxiv.org/abs/2205.10287).

Either way, the step at the larger batch has {{< math >}}$\kappa${{< /math >}} times the mean and {{< math >}}$\kappa${{< /math >}} times the noise covariance, in {{< math >}}$\kappa${{< /math >}} times fewer steps. Two optimizers give two rules, but the picture on the data clock is the same.

## 5. Invariant 3: the age of every token, in rollouts

Now put the queue on the same clock. Give every token an age {{< math >}}$S${{< /math >}} *in rollouts*: the number of rollouts the trainer has consumed since the token was drawn.

**The usual setting breaks it.** A configuration usually fixes the depth of the queue *in updates*. At {{< math >}}$\kappa${{< /math >}} times the batch, the same number of updates is {{< math >}}$\kappa${{< /math >}} times more rollouts. Every token is {{< math >}}$\kappa${{< /math >}} times older in rollouts, the weights move further while it waits, and the gap grows with the batch, although nothing in the configuration changed except {{< math >}}$B${{< /math >}}. This is one way a run that was stable at one batch size stops being stable at the next.

**Why age in rollouts is the right invariant.** Over {{< math >}}$S${{< /math >}} rollouts the weights move by {{< math >}}$S\,\mathcal{A}_B${{< /math >}} on average, with noise covariance {{< math >}}$S\,\mathcal{D}_B${{< /math >}}. In KL units, with {{< math >}}$F${{< /math >}} the policy's local sensitivity (the Fisher metric), the staleness term is to second order

{{< math >}}
$$
\text{drift} \;\approx\; \tfrac12\Big(\mathbb{E}[S^2]\,\big\|\mathcal{A}_B\big\|_F^2 + \mathbb{E}[S]\,\mathrm{tr}\big(F\,\mathcal{D}_B\big)\Big) .
$$
{{< /math >}}

Only the three invariants appear: progress per rollout, noise per rollout, and the ages in rollouts. The learning-rate rule keeps the first two, and the queue has to keep the third. The optimizer view and the staleness view are tracking one object, the accumulated update over a fixed amount of data, and that is why they share a clock.

**Setting 2: size the queue in rollouts.** Keep the same {{< math >}}$N${{< /math >}} rollouts in the queue at both batch sizes. A token is used once the trainer has consumed the queue ahead of it, so its age when used is {{< math >}}$N${{< /math >}} rollouts in both runs. The same queue is now {{< math >}}$\kappa${{< /math >}} times fewer updates deep, so the prefetch depth, which is configured in updates, comes down by {{< math >}}$\kappa${{< /math >}}:

{{< math >}}
$$
\frac{N}{B}\ \text{updates deep} \;\longrightarrow\; \frac{N}{\kappa B}\ \text{updates deep}, \qquad N\ \text{rollouts in both}.
$$
{{< /math >}}

## 6. The rule

| Setting | Rule when {{< math >}}$B \to \kappa B${{< /math >}} | What it keeps |
|---|---|---|
| Learning rate | {{< math >}}$\eta \to \kappa\eta${{< /math >}} (SGD) or {{< math >}}$\sqrt{\kappa}\,\eta${{< /math >}} (noise-dominated Adam) | Progress and noise per rollout |
| Queue | Fixed at {{< math >}}$N${{< /math >}} rollouts; prefetch depth in updates {{< math >}}$\div\,\kappa${{< /math >}} | The age of every token, in rollouts |

With the same age and the same dynamics per rollout, the weights move by the same amount while a token waits. So the staleness term of the gap is the same at every batch size, and whatever was tuned to hold the trainer near the sampler carries over: a mask threshold, or a KL coefficient.

Two remarks:

- **The queue does not choose the exponent of the step size; the optimizer does.** The two settings cannot be tuned one at a time, because they are tied to one clock.
- **The floor is not the queue's to move.** It belongs to the kernels and, as [Beyond Precision](https://arxiv.org/abs/2602.01826) showed, to the size of the update.

## 7. Where the rule stops

A rule is only useful with its boundary, and the boundary comes from the same idea: **anything counted in updates changes meaning when the batch changes.**

**Momentum.** The drift formula treated the updates across the queue as independent. Momentum ties each update to the ones before it, over a window of {{< math >}}$1/(1-\beta_1)${{< /math >}} updates. In rollouts that window is

{{< math >}}
$$
\frac{B}{1-\beta_1} \;\longrightarrow\; \frac{\kappa B}{1-\beta_1}, \qquad \text{and the rule needs} \quad N \gg \frac{\kappa B}{1-\beta_1} .
$$
{{< /math >}}

At the smaller batch the queue is several windows long, so across the queue the updates are close to independent. At the larger batch the queue holds the same number of rollouts but only a few updates, while the momentum window, still counted in updates, covers {{< math >}}$\kappa${{< /math >}} times more data. When the window is longer than the queue, all the updates a token waits through point the same way. The drift then grows like the *square* of the age, {{< math >}}$p=2${{< /math >}}, where the rule assumed it grows like the age, and it is no longer the quantity we kept invariant.

The fix follows the same logic: put the momentum window on the data clock as well, so that it covers the same number of rollouts. That means

{{< math >}}
$$
\beta_1 \;\to\; \beta_1^{\kappa}.
$$
{{< /math >}}

**Signal-dominated coordinates.** The {{< math >}}$\sqrt{\kappa}${{< /math >}} rule assumes coordinates dominated by noise; coordinates dominated by signal move Adam toward the linear rule. Real coordinates sit in between, and one step size has one exponent, so the invariants hold on average and no better.

These are the things I would check first when a rule that worked at one scale stops working at the next.

## 8. The pull at scale: an age weight

Part 1 ended with one coefficient {{< math >}}$\beta${{< /math >}}: how hard the update pulls toward the sampler. In an asynchronous system the samplers ran past versions of the weights, so pulling toward the sampler means pulling the trainer toward *its own past*: one spring to every version that drew data in the batch. Should old and fresh tokens be pulled alike?

The KL is quadratic in the log-ratio, so its gradient grows like the square root of the KL. With a weight {{< math >}}$w(a)${{< /math >}} on a token's KL, the pull at age {{< math >}}$a${{< /math >}} is

{{< math >}}
$$
\text{pull}(a) = \beta\, w(a)\,\sqrt{2\,\text{drift}(a)} .
$$
{{< /math >}}

With no weight, the pull grows with age like {{< math >}}$a^{p/2}${{< /math >}}, and the oldest, least trustworthy tokens dominate. Take the weight to be a power of age:

{{< math >}}
$$
w(a) = a^{-\rho}, \qquad \text{the pull is flat in age exactly when } \rho = p/2 .
$$
{{< /math >}}

The lower bound of Part 1 gives no guidance here. It asks for the same coefficient at every age, and at real horizons it certifies nothing. So this is a design choice, and equal pull at every age is the principle I would choose by.

**How {{< math >}}$\beta${{< /math >}} follows the batch.** The weight counts age in updates, {{< math >}}$a = S/B${{< /math >}}. Under the rule above the queue is fixed in rollouts, so at the larger batch the same token is {{< math >}}$\kappa${{< /math >}} times younger in updates and gets a larger weight:

{{< math >}}
$$
w\Big(\frac{S}{\kappa B}\Big) = \kappa^{\rho}\, w\Big(\frac{S}{B}\Big) \quad\Longrightarrow\quad \beta(\kappa B) = \kappa^{-\rho}\,\beta(B)
$$
{{< /math >}}

keeps the coefficient on every token the same as a function of its age in rollouts. The drift did not change, so the pull should not either. One exponent sets both the age weight and how {{< math >}}$\beta${{< /math >}} follows the batch. With no age weight, {{< math >}}$\rho = 0${{< /math >}} and {{< math >}}$\beta${{< /math >}} stays fixed.

One more property: the pull is delayed feedback toward past weights. It is stable while the step size times {{< math >}}$\beta${{< /math >}} times the curvature of the KL stays well below one, and that margin shrinks as the batch, and with it the step size, grows.

## 9. What is derived, and what is measured

What the analysis derives is the *shapes*:

- the form of the update (Part 1);
- the form of the gap;
- how the step size, the queue, the momentum window and the KL coefficient move together.

What it cannot give is the *constants*: the floor, the staleness coefficient, and the exponent {{< math >}}$p${{< /math >}}. They are local and move as the policy moves, so they are measured along every run. The gauges are cheap to log:

| Gauge | What it tells you |
|---|---|
| Sampler–trainer KL, bucketed by token age | The floor (age 0) and the staleness slope and exponent |
| The age histogram, in rollouts | Whether the queue really holds the age fixed across batch sizes |
| Mean and covariance of the update at two batch sizes | Whether the learning-rate rule keeps progress and noise per rollout |
| The gradient on fresh against stale data | Whether staleness has started to bend the update |

The same gauges tell you when a run has left the regime where the rule holds. That is what I mean by a science of scaling: rules for the shape, and measurements for the constants.

## Summary

| Piece | Statement |
|---|---|
| The gap | {{< math >}}$\mathrm{KL}(a) \approx c_0(\theta) + c_1(\theta)\eta^2 a^p${{< /math >}}: engine-mismatch floor plus staleness, both local |
| Budget | Mean age {{< math >}}$\le 2\cdot\text{headroom}/(\eta^2\sigma_u^2)${{< /math >}}; queue {{< math >}}$\propto B/\eta^2${{< /math >}} |
| Invariants | Progress per rollout, noise per rollout, token age in rollouts |
| Rule | {{< math >}}$\eta\times\kappa${{< /math >}} (SGD) or {{< math >}}$\times\sqrt\kappa${{< /math >}} (Adam); queue fixed in rollouts |
| Boundary | Momentum window: {{< math >}}$N \gg \kappa B/(1-\beta_1)${{< /math >}}, else {{< math >}}$\beta_1 \to \beta_1^\kappa${{< /math >}}; signal-dominated coordinates |
| Pull | Age weight {{< math >}}$a^{-p/2}${{< /math >}} for equal pull; {{< math >}}$\beta(\kappa B) = \kappa^{-\rho}\beta(B)${{< /math >}} |

## References

- J. Liu, Y. Li, et al. [When Speed Kills Stability: Demystifying RL Collapse from the Training-Inference Mismatch](/rl-collapse). Blog, 2025.
- Y. Zhang, Y. Li, et al. [Beyond Precision: Training-Inference Mismatch is an Optimization Problem and Simple LR Scheduling Fixes It](https://arxiv.org/abs/2602.01826). arXiv, 2026.
- Y. Li, J. Liu, et al. [Trust Region Masking for Long-Horizon LLM Reinforcement Learning](https://arxiv.org/abs/2512.23075). ICML 2026.
- Y. Li, J. Xu, et al. [The Optimal Token Baseline: Variance Reduction for Long-Horizon LLM-RL](https://arxiv.org/abs/2602.07078). ICML 2026.
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
