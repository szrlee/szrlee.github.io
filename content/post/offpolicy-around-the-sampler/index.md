---
title: "Around the Sampler: Tangent, Remainder and KL"
subtitle: "Off-Policy RL at Scale, Part 1: an exact first-order view of the off-policy update, with a sequence-level bound and no horizon factor"
summary: "Measured from the sampler, the gain of any policy update is a tangent plus a remainder. The tangent is a covariance loss whose gradient needs only the trainer; the remainder is bounded by R_max times the sequence KL, with no horizon factor. Score centering, per-token importance sampling, KL penalties and sequence masking all fall out of this one picture."
authors:
  - admin
tags:
  - Reinforcement Learning
  - Off-Policy
  - Trust Region
  - Language Models
  - Training Dynamics
categories:
  - Research
  - Theory
date: '2026-10-13T00:00:00Z'
lastmod: '2026-10-13T00:00:00Z'
math: true
draft: false
series:
  - Off-Policy RL at Scale
series_order: 1
---

{{% callout note %}}
**Series: Off-Policy RL at Scale**

- **Part 1 (this post):** around the sampler. The gain of an update is a tangent plus a remainder, and the remainder is controlled by the sequence KL.
- [**Part 2**](/post/offpolicy-gap-at-scale/): keeping the off-policy gap fixed as RL scales. A batch-size-invariant rule for the step size and the queue.

**Earlier public work this builds on:** [When Speed Kills Stability](/rl-collapse) and its three-part analysis ([1](/post/rl-collapse-part1/), [2](/post/rl-collapse-part2/), [3](/post/rl-collapse-part3/)), [Trust Region Masking](https://arxiv.org/abs/2512.23075), [the Optimal Token Baseline](https://arxiv.org/abs/2602.07078), and [Beyond Precision](https://arxiv.org/abs/2602.01826).
{{% /callout %}}

{{% callout note %}}
**TL;DR**

- Measure the gain of an update from the policy that actually drew the data. Then the gain is exactly a **tangent plus a remainder**.
- The tangent is a **covariance loss**. Its value needs the importance ratio, but its gradient needs only the trainer.
- The remainder is bounded by <i>R</i><sub>max</sub> · KL<sub>seq</sub>. The bound holds at any distance from the sampler, is stated for the whole sequence, and has **no horizon factor**.
- The lower edge of that band has a **closed-form maximizer** that is never worse than the sampler.
- Everything else falls out of the same picture:
  - **Score centering** is the gradient of the tangent.
  - **Per-token importance sampling** has the same tangent but more variance.
  - A **KL pull** and a **sequence mask** are the two ways to stay inside the band.
- Every method differs only in **how hard it pulls** toward the sampler: one coefficient, <i>β</i>.
{{% /callout %}}

---

## 1. One off-policy gap

Large RL runs on language models are built for throughput. Samplers generate rollouts on inference engines built for speed, finished rollouts wait in a queue, and a trainer takes a batch, steps, and sends new weights back. Every token therefore carries two distributions:

- the **sampler** {{< math >}}$\mu${{< /math >}}: the distribution that actually drew the token. It runs older weights, on a fast kernel, with whatever precision and logit processing that kernel uses;
- the **trainer** {{< math >}}$\pi${{< /math >}}: the current weights on the training kernel. The gradient is taken through it.

The gap between them has two sources.

- **Engine mismatch.** Two engines run the same weights through different kernels, so they assign different probabilities to the same token. This puts a floor under the gap: a run that is set up to be on-policy is off-policy anyway ([When Speed Kills Stability](/rl-collapse)).
- **Staleness.** Asynchrony means the sampler's weights are a few updates old.

In this post I treat the gap as one object, whatever its source. [Part 2](/post/offpolicy-gap-at-scale/) models how big it is and how it scales.

What we want to raise is {{< math >}}$J(\pi) = \mathbb{E}_{y\sim\pi}[R(y)]${{< /math >}}. What we compute is an average over {{< math >}}$B${{< /math >}} rollouts written by the sampler:

{{< math >}}
$$
\hat g = \frac{1}{B}\sum_{i=1}^{B} R(y^i)\,\nabla_\theta \log \pi(y^i), \qquad y^i \sim \mu .
$$
{{< /math >}}

Its two defects are the bias and the variance:

{{< math >}}
$$
\text{bias} = \mathbb{E}[\hat g] - \nabla_\theta J(\pi), \qquad \text{variance} = \mathbb{E}\big\|\hat g - \mathbb{E}[\hat g]\big\|^2 \propto \frac{1}{B}.
$$
{{< /math >}}

For one step of size {{< math >}}$\eta${{< /math >}} on an {{< math >}}$L${{< /math >}}-smooth objective, the two combine as

{{< math >}}
$$
\begin{aligned}
\mathbb{E}\big[J(\theta_{k+1})\big] - J(\theta_k) \;\ge\;& \underbrace{\eta\Big(1-\tfrac{L\eta}{2}\Big)\|\nabla J\|^2}_{\text{true progress}} \;+\; \underbrace{\eta(1-L\eta)\,\langle \nabla J,\ \text{bias}\rangle}_{\text{set by the gap}} \\
&-\; \underbrace{\tfrac{L\eta^2}{2}\Big[\text{variance} + \|\text{bias}\|^2\Big]}_{\text{set by the batch and the gap}} .
\end{aligned}
$$
{{< /math >}}

Read it as a statement about the system: the gap sets the bias, the batch sets the variance, and the step size multiplies both. The rest of this post is about the bias term: what the update should be when the data came from {{< math >}}$\mu${{< /math >}}.

## 2. The gain, measured from the sampler

Define the **gain** of moving from the sampler to the trainer, and the **log-ratio** of a rollout, one term per token:

{{< math >}}
$$
\text{gain} = J(\pi) - J(\mu), \qquad \ell = \sum_t \ell_t, \qquad \ell_t = \log\frac{\pi(y_t \mid x, y_{\lt t})}{\mu(y_t \mid x, y_{\lt t})} .
$$
{{< /math >}}

Reweighting each rollout by its ratio gives the gain exactly:

{{< math >}}
$$
\text{gain} = \mathbb{E}_{\mu}\big[R\,(e^{\ell}-1)\big].
$$
{{< /math >}}

This is exact but useless as an estimator: {{< math >}}$e^{\ell}${{< /math >}} is a product of thousands of token ratios. From here on, every expectation and covariance is over rollouts drawn by the sampler {{< math >}}$\mu${{< /math >}}.

## 3. The tangent: a covariance loss

Expand the log-ratio token by token. At each prefix, the mean of {{< math >}}$\ell_t${{< /math >}} under the sampler is minus a KL:

{{< math >}}
$$
\bar\ell_t := \mathbb{E}_{y_t\sim\mu}[\ell_t] = -\mathrm{KL}_t, \qquad \mathrm{KL}_t = \mathrm{KL}\big(\mu(\cdot\mid x,y_{\lt t})\,\big\|\,\pi(\cdot\mid x,y_{\lt t})\big).
$$
{{< /math >}}

So the log-ratio of a rollout splits into a centered sum and a sum of token KLs:

{{< math >}}
$$
\ell = \underbrace{\sum_t \big(\ell_t - \bar\ell_t\big)}_{\text{first order}} \;-\; \underbrace{\sum_t \mathrm{KL}_t}_{\text{second order}} .
$$
{{< /math >}}

The centered sum is first order in how far the trainer is from the sampler; the KLs are second order. Keeping the first-order part against the reward gives the **tangent** to the gain at the sampler:

{{< math >}}
$$
\text{tangent} = \mathrm{Cov}\Big(R,\ \sum_t \big(\ell_t-\bar\ell_t\big)\Big) = \sum_t \mathrm{Cov}\Big(R,\ \log\pi(y_t) - \mathbb{E}_{\mu}\big[\log\pi\big]\Big) + C .
$$
{{< /math >}}

The second form holds because the sampler's own log-probability does not depend on the weights we train, so it only contributes the constant {{< math >}}$C${{< /math >}}. The tangent is linear in the change of logits, and its slope at the sampler is the policy gradient there. No ratio appears in it. I call it the **covariance loss**: its *value* needs the ratio, its *gradient* needs only the trainer.

## 4. The remainder, exactly

Whatever the tangent misses is the remainder, and nothing is dropped:

{{< math >}}
$$
\text{gain} = \text{tangent} + \text{remainder}, \qquad \text{remainder} = \mathbb{E}\big[R\,(e^{\ell}-1-\ell)\big] - \mathbb{E}\Big[R\sum_t \mathrm{KL}_t\Big].
$$
{{< /math >}}

This remainder *is* the bias from Section 1, now as an exact quantity: an update that follows the tangent is wrong by exactly the remainder.

**The bound.** The remainder is a difference of two expectations, and three facts control it:

1. Both weights are non-negative: {{< math >}}$e^{x}-1-x \ge 0${{< /math >}}, and every {{< math >}}$\mathrm{KL}_t \ge 0${{< /math >}}.
2. Both weights have the same mean, the sequence KL:
{{< math >}}
$$
\mathbb{E}\big[e^{\ell}-1-\ell\big] = \mathbb{E}\Big[\sum_t \mathrm{KL}_t\Big] = \mathrm{KL}_{\text{seq}} := \mathrm{KL}\big(\mu(y)\,\|\,\pi(y)\big).
$$
{{< /math >}}
   The first equality holds because {{< math >}}$\mathbb{E}_\mu[e^{\ell}]=1${{< /math >}} and {{< math >}}$-\mathbb{E}_\mu[\ell] = \mathrm{KL}_{\text{seq}}${{< /math >}}. The second is the chain rule for KL.
3. The reward lies in {{< math >}}$[0, R_{\max}]${{< /math >}}.

So each of the two expectations is a number between {{< math >}}$0${{< /math >}} and {{< math >}}$R_{\max}\,\mathrm{KL}_{\text{seq}}${{< /math >}}, and so is their difference in absolute value:

{{< math >}}
$$
\boxed{\ \big|\text{remainder}\big| \;\le\; R_{\max}\,\mathrm{KL}_{\text{seq}}\ }
$$
{{< /math >}}

That is the whole proof. What makes the bound useful:

- **It holds at any distance** from the sampler. It is not a Taylor statement.
- **It is a statement about the whole sequence.** There is no worst token and no factor of the horizon.

**Compared with the per-position route.** Classical trust-region bounds go position by position and are governed by the worst token. Take a 4,096-token rollout with worst-token KL {{< math >}}$10^{-4}${{< /math >}}, sequence KL {{< math >}}$0.01${{< /math >}}, and reward scale 1:

| Bound on the surrogate error | Form | Value |
|---|---|---|
| Classical (TRPO) | {{< math >}}$T(T-1)\,D_{\mathrm{KL}}^{\text{tok,max}}${{< /math >}} | 1677 |
| Pinsker–marginal ([TRM](https://arxiv.org/abs/2512.23075)) | {{< math >}}$\tfrac{4}{3}T^{3/2}\,D_{\mathrm{KL}}^{\text{tok,max}}${{< /math >}} | 35 |
| Mixed ([TRM](https://arxiv.org/abs/2512.23075)) | {{< math >}}$2T\sqrt{D_{\mathrm{KL}}^{\text{tok,max}}\cdot \mathrm{KL}_{\text{seq}}}${{< /math >}} | 8.2 |
| Sequence-level (this post) | {{< math >}}$R_{\max}\,\mathrm{KL}_{\text{seq}}${{< /math >}} | 0.01 |

**A caveat I would rather state than be asked about.** The sequence KL adds up over tokens. For very long rollouts it can be far above one, and then no bound that ignores the reward certifies anything. I use the bound for the *shape* of a good step, not as a certificate.

**To second order** the remainder is a reward-weighted quadratic in the log-ratio:

{{< math >}}
$$
\text{remainder} \approx \tfrac12\,\mathrm{Cov}\Big(R,\ \big(\textstyle\sum_t c_t\big)^2 - \sum_t V_t\Big), \qquad c_t = \ell_t - \mathbb{E}_{\mu}[\ell_t], \quad V_t = \mathrm{Var}_{\mu}(c_t).
$$
{{< /math >}}

The cross terms of the {{< math >}}$c_t${{< /math >}}, which are martingale differences, vanish in expectation. That is why the quadratic has mean zero and appears as a covariance. Its sign is not fixed, and its scale is the sequence KL, since {{< math >}}$\mathrm{KL}_{\text{seq}} \approx \tfrac12\sum_t \mathbb{E}[V_t]${{< /math >}}. The boxed bound takes the worst case of this term and holds at any divergence.

## 5. The best step inside the band

The lower edge of the band is a lower bound on the gain:

{{< math >}}
$$
\text{gain} \;\ge\; \text{tangent} - R_{\max}\,\mathrm{KL}_{\text{seq}} .
$$
{{< /math >}}

It is concave in the trainer's logits, being a linear term minus convex KL terms, and it is zero at the sampler. Its top has a closed form: at every prefix, reweight the sampler by one plus the advantage over {{< math >}}$R_{\max}${{< /math >}},

{{< math >}}
$$
\pi^\star = \mu\cdot\Big(1 + \frac{A}{R_{\max}}\Big),
$$
{{< /math >}}

where {{< math >}}$A${{< /math >}} is the advantage of a token under the sampler.

To see it in the one-token case, the bound is {{< math >}}$\sum_y \mu(y)\big(A(y)+R_{\max}\big)\log\pi(y)${{< /math >}} plus a constant. It is maximized by {{< math >}}$\pi \propto \mu\,(A+R_{\max})${{< /math >}}, which already sums to one because {{< math >}}$\sum_y \mu A = 0${{< /math >}}.

Three consequences:

- **It is never worse than the sampler.** The bound is zero at {{< math >}}$\mu${{< /math >}}, so its maximizer has non-negative gain.
- **It points to the natural gradient.** Among the changes a network can actually make, the closest to this target is the natural-gradient step.
- **A KL budget changes the size, not the kind.** A constraint {{< math >}}$\mathrm{KL}_{\text{seq}} \le \delta${{< /math >}} with multiplier {{< math >}}$\lambda${{< /math >}} replaces {{< math >}}$R_{\max}${{< /math >}} by {{< math >}}$R_{\max}+\lambda${{< /math >}}: a smaller step in the same direction.

## 6. What the update computes: score centering

The update we actually compute is the gradient of the tangent. Differentiating the covariance loss, the constant drops and a centered score is left:

{{< math >}}
$$
\nabla_\theta\,\text{tangent} = \sum_t \mathbb{E}\Big[R\,\big(s_{y_t} - \bar s_t\big)\Big], \qquad s_{y_t} = \nabla_\theta \log\pi(y_t), \qquad \bar s_t = \mathbb{E}_{v\sim\mu}\big[s_v\big].
$$
{{< /math >}}

The two terms are the score of the token that was drawn, and the mean score at that prefix under the *sampler*. At the logits the difference is simple. The score of token {{< math >}}$v${{< /math >}} with respect to the logits is {{< math >}}$e_v - \pi${{< /math >}}. Its mean under the sampler is {{< math >}}$\mu - \pi${{< /math >}}. Subtracting, the trainer's probabilities cancel:

{{< math >}}
$$
s_{y_t} - \bar s_t \;\longrightarrow\; e_{y_t} - \mu .
$$
{{< /math >}}

Raise the token that was drawn; lower every token in proportion to how often the *sampler* draws it. Averaged over what the sampler draws, the raises and lowerings cancel exactly. So a constant reward moves nothing, and only the covariance between reward and token moves the weights.

Without the centering this fails. The plain score averages to zero under the *trainer*, but the data comes from the sampler, so even a constant reward pushes the policy. That is the off-policy bias in its simplest form, and the subtraction removes it. Score centering has an independent public treatment in [Marek & Ryabinin (2026)](https://arxiv.org/abs/2609.20807). Here it falls out of the expansion as the gradient of the tangent.

## 7. Per-token importance sampling: same tangent, more variance

The other first-order surrogate keeps, for each token, only that token's ratio {{< math >}}$r_t = e^{\ell_t}${{< /math >}}. It is the surrogate inside PPO and GRPO:

{{< math >}}
$$
L_{\text{IS}} = \mathbb{E}\Big[R\sum_t (r_t - 1)\Big], \qquad \nabla_\theta L_{\text{IS}} = \mathbb{E}\Big[R\sum_t r_t\, s_{y_t}\Big].
$$
{{< /math >}}

It leaves the sampler in the same direction as the covariance loss: **the same tangent**, so to first order it is the same step. The difference is in the gradient. Both gradients are built from the same scores, and both repair the same defect, a score that does not average to zero under the sampler:

- Importance sampling **multiplies** each score by its ratio, since {{< math >}}$\mathbb{E}_\mu[r_t\, s_{y_t}] = \mathbb{E}_\pi[s] = 0${{< /math >}}.
- The covariance loss **subtracts** the mean score.

Multiplication by a random number costs variance. The ratio averages to one, but a token that the sampler rarely draws and the trainer has come to prefer gets a large ratio, and the estimate jumps whenever it appears:

{{< math >}}
$$
\mathbb{E}_\mu\big[r_t^2\big] = 1 + \chi^2\big(\pi \,\|\, \mu\big),
$$
{{< /math >}}

The {{< math >}}$\chi^2${{< /math >}} divergence is dominated by exactly these rare tokens and can be far above the KL, while the KL is all that the bound controls. Clipping the ratio tames the jumps, but gives up first-order correctness on exactly those tokens. The covariance loss has no ratio, so every draw has weight one. Its price is that it needs the mean score {{< math >}}$\bar s_t${{< /math >}}. Both need the trainer to stay near the sampler.

## 8. Two ways to stay near the sampler

**Pull: a KL term in the loss.**

{{< math >}}
$$
\text{loss} = -\,\text{tangent} + \beta\,\mathrm{KL}_{\text{seq}}, \qquad -\nabla_\theta \mathrm{KL}_{\text{seq}} = \sum_t \mathbb{E}\big[\bar s_t\big].
$$
{{< /math >}}

The sequence KL compares the two policies over whole rollouts. So it is a divergence between the distributions over *states* that they induce, where the state is the prefix. That is more than a KL between next-token probabilities at fixed prefixes. It is exactly the formulation of [Divergence-Augmented Policy Optimization (NeurIPS 2019)](https://arxiv.org/abs/2501.15034), which used a divergence between state distributions to keep updates safe when off-policy data are reused.

The pull keeps every rollout. Its gradient is the mean score: it points toward the sampler and ignores the reward. With {{< math >}}$\beta = R_{\max}${{< /math >}} the loss is exactly minus the lower bound of Section 5.

**Reject: a sequence mask.**

{{< math >}}
$$
M(y) = \mathbf{1}\Big[\sum_t \mathrm{KL}_t \le \tau\Big].
$$
{{< /math >}}

This is the same quantity, the sequence KL, taken rollout by rollout. Drop a rollout when its sum is too large. That is [Trust Region Masking](https://arxiv.org/abs/2512.23075), which open stacks such as veRL ship.

- Inside the mask the token ratios are unclipped, so on the rollouts you keep, the estimator is still correct to first order.
- Other sequence-level divergences fit the same mask: the average token KL, the worst token KL (which the TRM theory uses), or the geometric mean of the ratios.
- The mask carries no gradient.
- Its threshold has to sit above the engine-mismatch floor, or it rejects everything.

Pull and reject both act on the sequence KL, which is the whole gap whatever its source, and they compose.

## 9. One number: how hard a method pulls

Write any update as the tangent plus a multiple of the pull:

{{< math >}}
$$
\text{update} = \underbrace{\mathbb{E}_{\mu}\Big[R\sum_t \big(s_{y_t}-\bar s_t\big)\Big]}_{\text{tangent}} \;+\; \beta\cdot \underbrace{\big(-\nabla_\theta \mathrm{KL}_{\text{seq}}\big)}_{\text{pull toward the sampler}} .
$$
{{< /math >}}

Then methods differ by one coefficient:

| Method | Coefficient {{< math >}}$\beta${{< /math >}} | What it does |
|---|---|---|
| Plain policy gradient | {{< math >}}$R-b${{< /math >}}, per rollout | An uncontrolled pull: toward the sampler on good rollouts, away from it on bad ones |
| Covariance loss, or unclipped ratios | {{< math >}}$0${{< /math >}} | The tangent alone |
| Certified lower bound | {{< math >}}$R_{\max}${{< /math >}} | As large as the true correction can be, and blind to the reward |
| With a KL budget | {{< math >}}$R_{\max}+\lambda${{< /math >}} | A smaller step in the same direction |

For plain policy gradient, expanding reward times score gives the tangent plus exactly the pull, with coefficient {{< math >}}$R-b${{< /math >}} rollout by rollout. This uncontrolled term is what is usually called drift: a trust-region coefficient with a random sign.

So the question of which method to use becomes the question of one number. I think of that number as a **stability–plasticity knob**: how much the learner trusts where its data came from. How it should be set, and how it should change with the age of the data and with the batch size, is the scaling question of [Part 2](/post/offpolicy-gap-at-scale/).

## Summary

| Piece | What it is |
|---|---|
| Gain | {{< math >}}$\mathbb{E}_\mu[R(e^{\ell}-1)]${{< /math >}}, exact |
| Tangent | Covariance loss; gradient needs only the trainer |
| Remainder | The bias, exactly; {{< math >}}$\lvert\cdot\rvert \le R_{\max}\mathrm{KL}_{\text{seq}}${{< /math >}}, no horizon factor |
| Best step | {{< math >}}$\pi^\star = \mu(1+A/R_{\max})${{< /math >}}, never worse than the sampler |
| Update | Score centering: {{< math >}}$e_{y}-\mu${{< /math >}} at the logits |
| Per-token IS | Same tangent; variance {{< math >}}$1+\chi^2${{< /math >}} |
| Stay close | Pull ({{< math >}}$\beta\,\mathrm{KL}_{\text{seq}}${{< /math >}}) or reject (sequence mask) |

## References

- J. Liu, Y. Li, et al. [When Speed Kills Stability: Demystifying RL Collapse from the Training-Inference Mismatch](/rl-collapse). Blog, 2025.
- Y. Li, J. Liu, et al. [Trust Region Masking for Long-Horizon LLM Reinforcement Learning](https://arxiv.org/abs/2512.23075). ICML 2026.
- Q. Wang, Y. Li, J. Xiong, T. Zhang. [Divergence-Augmented Policy Optimization](https://arxiv.org/abs/2501.15034). NeurIPS 2019.
- M. Marek, M. Ryabinin. [Score Centering Stabilizes Off-policy Reinforcement Learning](https://arxiv.org/abs/2609.20807). arXiv, 2026.
- J. Schulman, S. Levine, P. Abbeel, M. Jordan, P. Moritz. [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477). ICML 2015.

## Citation

```bibtex
@online{li-2026-around-the-sampler,
  title  = {Around the Sampler: Tangent, Remainder and KL},
  author = {Li, Yingru},
  year   = {2026},
  month  = oct,
  url    = {https://richardli.xyz/post/offpolicy-around-the-sampler/}
}
```
