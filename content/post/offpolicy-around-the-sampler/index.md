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
**Off-Policy RL at Scale · Part 1 of 2.** [Part 2](/post/offpolicy-gap-at-scale/) keeps the gap fixed as the run grows. This builds on [When Speed Kills Stability](/rl-collapse), [Trust Region Masking](https://arxiv.org/abs/2512.23075), the [Optimal Token Baseline](https://arxiv.org/abs/2602.07078) and [Beyond Precision](https://arxiv.org/abs/2602.01826).

- Measured from the sampler, the gain of any update is exactly a **tangent plus a remainder**.
- The tangent is a **covariance loss**: its gradient needs only the trainer.
- The remainder is at most <i>R</i><sub>max</sub> · KL<sub>seq</sub>. The bound is for the whole sequence, with **no horizon factor**.
- Score centering, per-token importance sampling, a KL pull and a sequence mask all fall out of this picture, and methods differ by **one coefficient**, <i>β</i>.
{{% /callout %}}

The figures step through like slides. Use **→** or **▶**, and step back with **←**.

## 1. One off-policy gap

**Every token is drawn by one policy and trained through another.**

{{< stepfig "system" >}}

The levers that buy throughput (fast kernels, asynchrony, larger batches) open a gap between {{< math >}}$\ps${{< /math >}}, which drew the token, and {{< math >}}$\pt${{< /math >}}, which takes the gradient. The gap has two sources: engine mismatch, which is present even at identical weights, and staleness. [Part 2](/post/offpolicy-gap-at-scale/) models both. The gap sets the bias of the update, the batch sets its variance, and the step size multiplies both:

{{< stepfig "lemma" >}}

{{< math >}}
$$
\begin{aligned}
\mathbb{E}\big[J(\theta_{k+1})\big] - J(\theta_k) \;\ge\;& \underbrace{\eta\big(1-\tfrac{L\eta}{2}\big)\|\nabla J\|^2}_{\text{true progress}} + \underbrace{\eta(1-L\eta)\langle \nabla J, \text{bias}\rangle}_{\text{set by the gap}} \\
&- \underbrace{\tfrac{L\eta^2}{2}\big[\text{variance} + \|\text{bias}\|^2\big]}_{\text{set by the batch and the gap}} .
\end{aligned}
$$
{{< /math >}}

This post is about the bias: what the update should be when the data came from {{< math >}}$\ps${{< /math >}}. From here on, {{< math >}}$\mathbb{E}${{< /math >}} and {{< math >}}$\mathrm{Cov}${{< /math >}} are over rollouts drawn by the sampler.

## 2. Gain = tangent + remainder

**Measured from the sampler, the gain of an update is exactly a tangent plus a remainder.**

{{< stepfig "gain" >}}

With {{< math >}}$\ell_t = \log\frac{\pt(y_t)}{\ps(y_t)}${{< /math >}} and {{< math >}}$\ell = \sum_t \ell_t${{< /math >}}, reweighting whole rollouts gives the gain exactly:

{{< math >}}
$$
\text{gain} = J(\pt) - J(\ps) = \mathbb{E}\big[R\,(e^{\ell}-1)\big].
$$
{{< /math >}}

At each prefix, the mean of {{< math >}}$\ell_t${{< /math >}} under the sampler is {{< math >}}$-\mathrm{KL}_t${{< /math >}}. So {{< math >}}$\ell${{< /math >}} splits into a centered first-order sum minus a second-order sum of token KLs. The first-order part gives the **tangent**:

{{< math >}}
$$
\text{tangent} = \mathrm{Cov}\Big(R,\ \sum_t \big(\ell_t-\bar\ell_t\big)\Big) = \sum_t \mathrm{Cov}\Big(R,\ \log\pt(y_t) - \mathbb{E}_{\ps}\big[\log\pt\big]\Big) + C .
$$
{{< /math >}}

This is the **covariance loss**. Computing its value needs the ratio, but its gradient needs only the trainer. Whatever the tangent misses is the remainder, and that remainder *is* the bias:

{{< math >}}
$$
\text{remainder} = \mathbb{E}\big[R\,(e^{\ell}-1-\ell)\big] - \mathbb{E}\Big[R\sum_t \mathrm{KL}_t\Big].
$$
{{< /math >}}

**The bound, in three facts.**

1. Both weights are non-negative.
2. Both have the same mean, {{< math >}}$\mathbb{E}[e^{\ell}-1-\ell] = \mathbb{E}[\sum_t \mathrm{KL}_t] = \mathrm{KL}_{\text{seq}}${{< /math >}}.
3. The reward lies in {{< math >}}$[0, R_{\max}]${{< /math >}}.

So each expectation lies in {{< math >}}$[0, R_{\max}\mathrm{KL}_{\text{seq}}]${{< /math >}}, and therefore

{{< math >}}
$$
\boxed{\ \big|\text{remainder}\big| \le R_{\max}\,\mathrm{KL}_{\text{seq}}\ }
$$
{{< /math >}}

The bound holds at any distance from the sampler and has no horizon factor. Per-position trust-region bounds are governed by the worst token instead. For a 4,096-token rollout with worst-token KL {{< math >}}$10^{-4}${{< /math >}}, sequence KL 0.01 and reward scale 1:

| Bound on the surrogate error | Form | Value |
|---|---|---|
| Classical (TRPO) | {{< math >}}$T(T-1)\,D_{\mathrm{KL}}^{\text{tok,max}}${{< /math >}} | 1677 |
| Mixed ([TRM](https://arxiv.org/abs/2512.23075)) | {{< math >}}$2T\sqrt{D_{\mathrm{KL}}^{\text{tok,max}}\,\mathrm{KL}_{\text{seq}}}${{< /math >}} | 8.2 |
| Sequence-level (this post) | {{< math >}}$R_{\max}\,\mathrm{KL}_{\text{seq}}${{< /math >}} | 0.01 |

For very long rollouts the sequence KL can exceed one, and then no bound that ignores the reward certifies anything. I use the bound for the shape of a good step, not as a certificate.

<details class="spoiler">
<summary>To second order, the remainder is a reward-weighted quadratic</summary>

{{< math >}}
$$
\text{remainder} \approx \tfrac12\,\mathrm{Cov}\Big(R,\ \big(\textstyle\sum_t c_t\big)^2 - \sum_t V_t\Big), \qquad c_t = \ell_t - \mathbb{E}_{\ps}[\ell_t], \quad V_t = \mathrm{Var}_{\ps}(c_t).
$$
{{< /math >}}

The {{< math >}}$c_t${{< /math >}} are martingale differences, so their cross terms vanish in expectation. That is why the quadratic has mean zero and appears as a covariance. Its sign is not fixed, and its scale is the sequence KL, since {{< math >}}$\mathrm{KL}_{\text{seq}} \approx \tfrac12\sum_t \mathbb{E}[V_t]${{< /math >}}.

</details>

## 3. The best step

**The lower edge of the band has a closed-form top, and it is never worse than the sampler.**

The last frame of the figure above shows the lower bound {{< math >}}$\text{gain} \ge \text{tangent} - R_{\max}\mathrm{KL}_{\text{seq}}${{< /math >}}. It is concave in the trainer's logits and zero at the sampler. Its maximizer reweights the sampler at every prefix by one plus the advantage {{< math >}}$A${{< /math >}} over {{< math >}}$R_{\max}${{< /math >}}:

{{< math >}}
$$
\pi^\star = \ps\cdot\Big(1 + \frac{A}{R_{\max}}\Big).
$$
{{< /math >}}

Since the bound is zero at the sampler, this step is never worse than the sampler. Among the changes a network can make, the natural-gradient step comes closest to it. A KL budget with multiplier {{< math >}}$\lambda${{< /math >}} replaces {{< math >}}$R_{\max}${{< /math >}} by {{< math >}}$R_{\max}+\lambda${{< /math >}}: a smaller step in the same direction.

<details class="spoiler">
<summary>Why, in the one-token case</summary>

The bound equals {{< math >}}$\sum_y \ps(y)\big(A(y)+R_{\max}\big)\log\pt(y)${{< /math >}} plus a constant. It is maximized by {{< math >}}$\pt \propto \ps\,(A+R_{\max})${{< /math >}}, which already sums to one because {{< math >}}$\sum_y \ps A = 0${{< /math >}}.

</details>

## 4. Score centering

**The gradient of the tangent raises the drawn token and lowers every token by how often the sampler draws it.**

{{< stepfig "centering" >}}

{{< math >}}
$$
\nabla_\theta\,\text{tangent} = \sum_t \mathbb{E}\Big[R\,\big(s_{y_t} - \bar s_t\big)\Big], \qquad \bar s_t = \mathbb{E}_{v\sim\ps}\big[s_v\big], \qquad s_{y_t} - \bar s_t \;\to\; e_{y_t} - \ps \ \text{ at the logits}.
$$
{{< /math >}}

Averaged over what the sampler draws, the raises and lowerings cancel, so a constant reward moves nothing. The plain score averages to zero under the trainer, not under the sampler, so without centering even a constant reward pushes the policy. That is the off-policy bias in its simplest form. Score centering has an independent public treatment in [Marek & Ryabinin (2026)](https://arxiv.org/abs/2609.20807); here it is simply the gradient of the tangent.

## 5. Per-token importance sampling: same tangent, more variance

**PPO and GRPO multiply by a ratio where the covariance loss subtracts a mean, and multiplication costs variance.**

{{< stepfig "isvar" >}}

{{< math >}}
$$
L_{\text{IS}} = \mathbb{E}\Big[R\sum_t (r_t - 1)\Big], \qquad \nabla_\theta L_{\text{IS}} = \mathbb{E}\Big[R\sum_t r_t\, s_{y_t}\Big], \qquad \mathbb{E}\big[r_t^2\big] = 1 + \chi^2 .
$$
{{< /math >}}

At the sampler it has the same tangent, so to first order it takes the same step. But a token that the sampler rarely draws and the trainer now prefers gets a large ratio, and the estimate jumps whenever that token appears. The {{< math >}}$\chi^2${{< /math >}} divergence, which is dominated by such tokens, can be far above the KL, and the KL is all the bound controls. Clipping tames the jumps, but gives up first-order correctness on exactly those tokens.

## 6. One knob: how hard to pull toward the sampler

**Every method is the tangent plus some multiple {{< math >}}$\beta${{< /math >}} of a pull toward the sampler.**

- **Pull.** Add {{< math >}}$\beta\,\mathrm{KL}_{\text{seq}}${{< /math >}} to the loss. Its gradient is the mean score, which points toward the sampler and ignores the reward. The sequence KL is a divergence between the *state* distributions of the two policies (the state is the prefix), as in [Divergence-Augmented Policy Optimization (NeurIPS 2019)](https://arxiv.org/abs/2501.15034).
- **Reject.** Drop a rollout whose sequence KL is too large, using {{< math >}}$M(y) = \mathbf{1}[\sum_t \mathrm{KL}_t \le \tau]${{< /math >}}. This is [Trust Region Masking](https://arxiv.org/abs/2512.23075), the principled form of masked importance sampling (MIS), which ships in veRL. Inside the mask the estimator is still first-order correct. The threshold must sit above the engine-mismatch floor.

<figure class="post-fig">
<img src="/figs/img/trm-vs-ppo-clipping.png" alt="Training curves: with either sequence mask the score keeps rising while PPO clipping degrades; the mismatch stays bounded under the masks.">
<figcaption>Right: score with the worst-token mask (orange) and the average-token mask (blue) against PPO clipping (green). Left: the mismatch stays bounded under the masks. From the TRM paper.</figcaption>
</figure>

| Method | Coefficient {{< math >}}$\beta${{< /math >}} | What it does |
|---|---|---|
| Plain policy gradient | {{< math >}}$R-b${{< /math >}}, per rollout | An uncontrolled pull: toward the sampler on good rollouts, away from it on bad ones |
| Covariance loss, or unclipped ratios | {{< math >}}$0${{< /math >}} | The tangent alone |
| Certified lower bound | {{< math >}}$R_{\max}${{< /math >}} | As large as the true correction can be, and blind to the reward |
| With a KL budget | {{< math >}}$R_{\max}+\lambda${{< /math >}} | A smaller step in the same direction |

{{< math >}}$\beta${{< /math >}} is a stability–plasticity knob: it sets how much the learner trusts where its data came from. How {{< math >}}$\beta${{< /math >}}, the step size and the queue should move as the run grows is the subject of [Part 2](/post/offpolicy-gap-at-scale/).

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
