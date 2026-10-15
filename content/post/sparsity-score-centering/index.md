---
title: "Sparsity Comes from Score Centering"
subtitle: "Why Post-Training Is Sparse, Part 2: the condition is a score centered under the sampler, not on-policy data, and the sampler's Fisher decides where the update can go"
summary: "With a known sampler, sparsity requires the score to be centered under it; whether the data is on-policy matters only through that. A centered update is capped by the sampler's Fisher, which is concentrated: in the output layer a row can move only if the sampler emits its token, and in hidden layers only the directions that tell likely candidates apart remain. Without centering, a mean shift adds up along the sequence and escapes the cap; under training–inference mismatch it is numerical noise. SFT cannot be centered, since its sampler is unknown: its gradient is the shift itself."
authors:
  - admin
tags:
  - Reinforcement Learning
  - Policy Gradient
  - Fisher Information
  - Score Centering
  - Sparsity
  - Off-Policy
  - Language Models
categories:
  - Research
  - Theory
date: '2026-10-15T00:10:00Z'
lastmod: '2026-10-15T00:10:00Z'
math: true
draft: false
series:
  - Why Post-Training Is Sparse
series_order: 2
---

{{% callout note %}}
**Why Post-Training Is Sparse · Part 2 of 3.** Builds on [Part 1](/post/gradient-is-covariance/), where a centered score makes the expected update a covariance that the Fisher caps. [Part 3](/post/what-policy-gradient-can-reach/) asks what policy gradient can still reach.

- **Centering is the condition, not on-policy data.** The Fisher caps the update only if the score has zero mean under the sampler. On-policy that comes free; off-policy, centering or per-token importance weights provide it.
- **Necessary, not sufficient.** The cap makes the update sparse only if the Fisher is concentrated on a few coordinates. The reward then picks which of those move.
- **The Fisher is concentrated.** In the output layer the formula is exact: a row's Fisher is large only for tokens the sampler emits. In hidden layers only a few directions remain; coordinate sparsity there is measured, not derived.
- **Without centering**, a mean shift adds up along the sequence and escapes the cap. Under training–inference mismatch it is numerical noise.
- **SFT is a different case.** Its sampler is unknown, so it cannot be centered: its whole gradient is the mean shift, and it stays sparse only when the data is close to the model.
{{% /callout %}}

All of this is about the expected update. Whether a stored weight ends up changing also depends on the optimizer and on precision. Adam normalizes per coordinate, so a small but consistently signed gradient still moves a parameter by about one learning rate ([Keeping the Off-Policy Gap Fixed](/post/offpolicy-gap-at-scale/)). [Mukherjee et al. (2025)](https://arxiv.org/abs/2505.11711) count a bf16 weight as unchanged when it moves by at most {{< math >}}$10^{-5}${{< /math >}}, and note that at unlimited precision the updates would be dense. The bound says which coordinates get a small expected update; the optimizer and the precision decide whether that leaves the weight unchanged.

## 1. Output layer: a row can move only if the sampler emits its token

**In the output layer the Fisher is an exact coin flip: row {{< math >}}$c${{< /math >}} can move only if the sampler emits token {{< math >}}$c${{< /math >}}, and hardly at all once it is sure of it.**

{{< stepfig "coinflip" >}}

{{< math >}}
$$
z_a=\sum_k W_{a,k}\,h_k\ \Rightarrow\ \kappa_{(c,k),a}=\mathbf 1[a=c]\,h_k,
$$
$$
\bar\kappa^{\mu}_{(c,k)}=\smu(c)\,h_k,\qquad f_{(c,k)}=h_k^2\,\smu(c)\big(1-\smu(c)\big),
$$
$$
\sum_{c,k}f_{(c,k)}=\|h\|^2\Big(1-\sum_c\smu(c)^2\Big),
$$
$$
F^{\mu}_{\text{row }c}\le\max_s\|h(s)\|^2\cdot\mathbb{E}_{\smu}\big[\#\{t:\ y_t=c\}\big].
$$
{{< /math >}}

The weight {{< math >}}$W_{c,k}${{< /math >}} touches only token {{< math >}}$c${{< /math >}}'s logit, so for it the next token is a coin flip, {{< math >}}$c${{< /math >}} or not {{< math >}}$c${{< /math >}}. Its Fisher is the coin's variance times {{< math >}}$h_k^2${{< /math >}}: near zero when the sampler almost never considers {{< math >}}$c${{< /math >}}, and near zero when it is almost sure of it. With tea at 0.5, coffee at 0.4 and 0.1 spread over a hundred thousand other tokens, the total is about {{< math >}}$1-0.25-0.16=0.59${{< /math >}}, and the tea and coffee rows hold {{< math >}}$0.25+0.24=0.49${{< /math >}} of it: 83%. The other 17% is spread over a hundred thousand rows, about a millionth each.

The two factors have different sources. {{< math >}}$\smu(c)${{< /math >}} comes from the sampler and decides which rows matter: without centering the second moment would be {{< math >}}$\smu(c)\,h_k^2${{< /math >}}, concentrated on the same rows. Centering contributes the factor {{< math >}}$1-\smu(c)${{< /math >}}, which silences a token the sampler is sure of. Its main job is elsewhere: only a centered update is capped by this Fisher at all ([section 3](#3-without-centering-the-mean-shift)).

Along a response, {{< math >}}$\mathbb{E}[\smu(c\mid s_t)]=\Pr(y_t=c)${{< /math >}}, so a row's Fisher is bounded by how often the sampler emits its token. A task emits a few thousand of the vocabulary's hundred thousand tokens; the other rows barely move. The total is also the expected squared logit update of [Logit Dynamics](https://arxiv.org/abs/2506.12912): averaged over the drawn token, its {{< math >}}$1-2P_c+C(P)${{< /math >}} becomes {{< math >}}$1-C(P)${{< /math >}}, with {{< math >}}$C(P)=\sum_aP_a^2${{< /math >}}.

{{< livefig "s-coinflip" "Drag the sliders. As the sampler grows sure of one token the total Fisher collapses, and what is left sits in the rows it still hesitates over." 470 >}}

<details class="spoiler">
<summary>The whole output block, and a bound on concentration</summary>

{{< math >}}
$$
f_W=\big(\operatorname{diag}(\smu)-\smu\,\smu^\top\big)\otimes h\,h^\top .
$$
{{< /math >}}

The left factor is the covariance of a categorical distribution. Its off-diagonal entries are negative: raising one token crowds out another. Since {{< math >}}$\smu(c)(1-\smu(c))\le\smu(c)${{< /math >}}, for any set of tokens {{< math >}}$S${{< /math >}}:

{{< math >}}
$$
\frac{\sum_{c\notin S}\sum_k f_{(c,k)}}{\sum_c\sum_k f_{(c,k)}}\le\frac{\sum_{c\notin S}\smu(c)}{1-\sum_c\smu(c)^2}.
$$
{{< /math >}}

</details>

## 2. Hidden layers: few directions, not few coordinates

**In hidden layers the Fisher sees only the few directions that tell the likely candidates apart.**

{{< stepfig "subspace" >}}

{{< math >}}
$$
z_a=w_a^\top h,\quad u_i=\frac{\partial h}{\partial\theta_i}\ \Rightarrow\ \kappa_{i,a}=w_a^\top u_i,
$$
$$
\mathbb{E}_{\smu}\big[\kappa_{i,a}^2\big]=u_i^\top\big(C_\mu+\bar w_\mu\bar w_\mu^\top\big)u_i,
$$
$$
f_i(s)=\operatorname{Var}_{a\sim\smu}\big[\kappa_{i,a}\big]=u_i^\top C_\mu\,u_i,
$$
$$
C_\mu=\sum_a\smu(a)\,(w_a-\bar w_\mu)(w_a-\bar w_\mu)^\top,
$$
$$
\operatorname{rank}C_\mu\le n-1.
$$
{{< /math >}}

Centering deletes the rank-one term {{< math >}}$\bar w_\mu\bar w_\mu^\top${{< /math >}}: what a parameter does to the average likely logit, {{< math >}}$\bar\kappa^{\mu}_i=\bar w_\mu^\top u_i${{< /math >}}. Raising every likely logit alike moves no probability, so that part counts for nothing. What remains is the covariance of the candidates. If the sampler puts mass on {{< math >}}$n${{< /math >}} tokens, their deviations from the mean sum to zero, so they span at most {{< math >}}$n-1${{< /math >}} directions. In the extreme where {{< math >}}$\smu${{< /math >}} sits on one token, every parameter's Fisher is zero after centering, while the uncentered second moment is not. Short of that extreme, concentration alone already confines the second moment to the {{< math >}}$n${{< /math >}} directions {{< math >}}$w_a${{< /math >}}; centering removes the common part on top, which may carry most of the energy. How much of the Fisher's smallness that accounts for is open ([section 6](#6-scope-and-open-questions)).

This is low dimension in function space, not coordinate sparsity. That would need most {{< math >}}$u_i${{< /math >}} to be nearly orthogonal to these directions, which depends on the architecture and on pretraining, and the directions change from prefix to prefix. The evidence is empirical. In RL on Qwen3-1.7B-Base, [Zhang, Mukherjee & Peng (2026)](https://arxiv.org/abs/2609.36262) find that training only the top 0.1% of parameters by initial Fisher recovers 85.21% of the full-training gain on MATH-500, and 5% recovers 98.82%. The token Fisher under the model's own probabilities is 0.16 of its value under a uniform distribution at the start, and 0.04 after 1,000 steps. These numbers show that the Fisher picks out the coordinates that matter. The authors also report that likely next tokens tend to have similar sensitivities, which is the common part centering removes; but the model-versus-uniform ratio changes concentration and token assignment together, so the two mechanisms are not yet separated ([section 6](#6-scope-and-open-questions)). Sparse is also not low rank: the measured updates touch few coordinates yet are nearly full rank in every matrix ([Mukherjee et al., 2025](https://arxiv.org/abs/2505.11711)).

## 3. Without centering: the mean shift

**Without centering, a mean shift adds up along the sequence and can push gradient into directions the Fisher cannot see.**

{{< stepfig "meanshift" >}}

{{< math >}}
$$
m_i(s)=\bar\kappa^{\mu}_i(s)-\bar\kappa^{p}_i(s),
$$
$$
\mathbb{E}_{a\sim\smu}\big[(\kappa_{i,a}-\bar\kappa^{p}_i)^2\big]=\operatorname{Var}_{a\sim\smu}\big[\kappa_{i,a}\big]+m_i^2,
$$
$$
\mathbb{E}_{\smu}\big[\mathrm{score}_i^{(u)}\,\mathrm{score}_i^{(t)}\big]
$$
$$
=\mathbb{E}_{\smu}\big[\mathrm{score}_i^{(u)}\,m_i(s_t)\big]\ne0\quad(u\lt t).
$$
{{< /math >}}

Draw tokens from {{< math >}}$\smu${{< /math >}} but subtract only the model's mean, as the plain softmax score does, and each position averages to {{< math >}}$m_i${{< /math >}} instead of zero. When the shift keeps its sign, the positions add up coherently: roughly {{< math >}}$T\sigma^2+T^2m^2${{< /math >}} against {{< math >}}$T\sigma^2${{< /math >}} with centering, an order-of-magnitude estimate rather than a theorem.

SFT is the extreme case, and a different one. Its sampler, the data's next-token distribution, is unknown: each prefix shows a single token, so the score cannot be centered under it. Prefixes come from the data, and by the tower property the gradient is exactly the accumulated shift:

{{< math >}}
$$
\frac{\partial}{\partial\theta_i}\,\mathbb{E}_{\text{data}}\big[\log\pi_\theta(y\mid x)\big]=\mathbb{E}_{\text{data}}\Big[\sum_t m_i(s_t)\Big].
$$
{{< /math >}}

Even with a known sampler, centering would delete the whole signal: with {{< math >}}$A\equiv1${{< /math >}} the centered update is {{< math >}}$\operatorname{Cov}_{\smu}(1,c_i)=0${{< /math >}}. Imitation is the mean shift. So how sparse SFT is can only depend on how small the shift is, that is, on how far the data is from the model. The measurements agree: SFT on rejection-sampled, in-distribution data leaves 91% of parameters unchanged, and SFT on out-of-distribution data leaves 7% ([Mukherjee et al., 2025](https://arxiv.org/abs/2505.11711)). In the output layer the demonstrated token's row receives {{< math >}}$(1-\pmodel(y^\star))\,h${{< /math >}}, nearly full size when the model gives {{< math >}}$y^\star${{< /math >}} almost no probability, while the on-policy Fisher of that row is proportional to {{< math >}}$\pmodel(y^\star)${{< /math >}}. Data that leaves the model's distribution touches parameters the model itself never touches.

<details class="spoiler">
<summary>The same cost, booked as a χ² signal</summary>

Let {{< math >}}$D${{< /math >}} be the data distribution and {{< math >}}$w=D/\pi_\theta${{< /math >}}. Because the score has zero mean under {{< math >}}$\pi_\theta${{< /math >}}, any constant {{< math >}}$b${{< /math >}} can be subtracted:

{{< math >}}
$$
G^{\text{SFT}}_i=\mathbb{E}_{\pi}\big[w\cdot\mathrm{score}_i\big]=\mathbb{E}_{\pi}\big[(w-b)\,\mathrm{score}_i\big],
$$
$$
|G^{\text{SFT}}_i|\le\sqrt{\mathbb{E}_{\pi}\big[(w-b)^2\big]}\,\sqrt{F_i},
$$
$$
\mathbb{E}_{\pi}\big[(w-b)^2\big]=\chi^2(D\,\|\,\pi_\theta)+(1-b)^2
$$
$$
\ge\ \chi^2(D\,\|\,\pi_\theta).
$$
{{< /math >}}

The weight is a product {{< math >}}$w=\prod_t\rho_t${{< /math >}} with {{< math >}}$\rho_t=q(y_t\mid s_t)/\pmodel(y_t\mid s_t)${{< /math >}}, where {{< math >}}$q${{< /math >}} is the data's next-token distribution, and {{< math >}}$1+\chi^2(D\|\pi_\theta)=\mathbb{E}_D[w]${{< /math >}}. Conditioning on the last position gives {{< math >}}$\mathbb{E}_D[\rho_T\mid s_T]=1+\chi^2\big(q(\cdot\mid s_T)\,\|\,\pmodel(\cdot\mid s_T)\big)${{< /math >}}. If every position has {{< math >}}$\chi^2(q\|p)\ge c${{< /math >}}, backward induction over positions, with every ratio positive, gives

{{< math >}}
$$
1+\chi^2(D\,\|\,\pi_\theta)\ \ge\ (1+c)^T .
$$
{{< /math >}}

The bound still holds, but its signal scale grows at least exponentially with distance from the model. It is the same {{< math >}}$(1+\chi^2)^T${{< /math >}} form as the variance of sequence-level importance sampling in [rl-collapse Part 2](/post/rl-collapse-part2/), and the {{< math >}}$\chi^2${{< /math >}} that makes per-token importance sampling noisy in [Around the Sampler](/post/offpolicy-around-the-sampler/#5-per-token-importance-sampling-same-tangent-more-variance). Booked directly under the data, the cost moves into the empirical Fisher {{< math >}}$\mathbb{E}_D[\mathrm{score}_i^2]${{< /math >}}, which a small {{< math >}}$F_i${{< /math >}} does not bound.

</details>

## 4. Four cases side by side

**The bound is a property of the estimator, not of RL.**

| Case | Samples from | Score centered under | Fisher bound |
|---|---|---|---|
| On-policy RL, on-policy distillation | the model | the model, for free | holds, with the model's Fisher |
| Off-policy with score centering | a known sampler | the sampler | holds, with the sampler's Fisher |
| SFT | the data, an unknown sampler | the model only; centering is impossible | holds formally; the signal scale is {{< math >}}$\chi^2${{< /math >}} |
| Value-based TD learning | a replay buffer | nothing | does not apply |

TD weights a raw logit gradient, {{< math >}}$\mathbb{E}[\delta\cdot\kappa_{i,a}]${{< /math >}}, which keeps the common component: for a Q-value a uniform shift is the value level, not a redundancy. So the bound does not apply, which is not evidence that TD updates are dense. Two observations fit the table. DPO on a fixed preference set has an unknown sampler too, and it behaves like SFT: sparse on in-distribution data, dense on out-of-distribution data ([Mukherjee et al., 2025](https://arxiv.org/abs/2505.11711)). What separates the cases is the shift, not RL versus SFT. And SFT on expert traces made near-policy by MCMC sampling generalizes and forgets like on-policy methods ([Karan, Chen & Du, 2026](https://arxiv.org/abs/2610.02140)), as a small {{< math >}}$\chi^2(D\,\|\,\pi_\theta)${{< /math >}} would suggest; that paper does not measure sparsity.

## 5. Sparsity is not a defect

**Under training–inference mismatch, centering is what keeps numerical noise out of the update.**

When the sampler is an inference engine, {{< math >}}$\smu\approx\pmodel${{< /math >}}, and the shift {{< math >}}$m_i${{< /math >}} comes only from numerical differences between two implementations. It carries no reward information, yet it injects gradient into directions with near-zero on-policy Fisher and accumulates along the sequence. In the output layer the shift for row {{< math >}}$c${{< /math >}} is {{< math >}}$(\smu(c)-\pmodel(c))\,h${{< /math >}}. Where the engine draws a token more often than the model would, {{< math >}}$\smu(c)>\pmodel(c)${{< /math >}}, that row is pushed although its on-policy Fisher is near zero. This is consistent with the mismatch doing most of its damage on rare tokens ([When Speed Kills Stability](/rl-collapse)).

Centering removes the shift exactly and keeps the update in the directions spanned by {{< math >}}$C_\mu\approx C_p${{< /math >}}. The engine's full distribution is rarely available, but per-token importance weights do the same job with only the drawn token's two probabilities: with {{< math >}}$\rho_t=\pmodel(y_t\mid s_t)/\smu(y_t\mid s_t)${{< /math >}}, {{< math >}}$\mathbb{E}_{\smu}\big[\rho_t(\kappa_{i,y_t}-\bar\kappa^{p}_i)\big]=\sum_a\pmodel(a)\big(\kappa_{i,a}-\bar\kappa^{p}_i\big)=0${{< /math >}} at every prefix. Clipping a token's ratio breaks this. [Trust Region Masking](/post/trust-region-masking/) never clips a token, so it leaves the weights that do the centering intact; it only drops whole sequences that left the trust region. Sparsity here means not moving the parameters that only noise would push.

## 6. Scope and open questions

- **What centering needs.** The sampler's full next-token distribution at every prefix: engine log-probabilities or a stale model. Human demonstrations have no such distribution, which is why SFT cannot be centered. Per-token importance weights need only the drawn token's probability under both (section 5). With top-k or top-p, the sampler is the truncated, renormalized distribution. Estimating {{< math >}}$\bar\kappa^{\mu}${{< /math >}} from a batch makes the cancellation approximate.
- **What it changes.** Off-policy, centering drops the pull toward the sampler; the coefficient {{< math >}}$\beta${{< /math >}} in [Around the Sampler](/post/offpolicy-around-the-sampler/#6-one-knob-how-hard-to-pull-toward-the-sampler) adds it back on purpose.
- **Two mechanisms, not yet separated.** A concentrated sampler and candidates that react alike both lower the Fisher. Two clean tests: permute which tokens get which probabilities at fixed entropy, and measure per coordinate the share centering removes, {{< math >}}$\rho_i=(\bar\kappa^{p}_i)^2/\sum_a\pmodel(a)\,\kappa_{i,a}^2${{< /math >}}. If low-Fisher coordinates have {{< math >}}$\rho_i\approx1${{< /math >}}, the sparsity comes mostly from the common component, as an untested heuristic suggests: likely tokens have large logits, so {{< math >}}$h${{< /math >}} may lean along {{< math >}}$\bar w_\mu${{< /math >}}.
- **One step.** The Fisher itself changes during training, which is where [Part 3](/post/what-policy-gradient-can-reach/) starts.

**Next.** [Part 3](/post/what-policy-gradient-can-reach/) turns the bound around: how far can one update move a response the model almost never produces?

## References

- Y. Zhang, S. Mukherjee, H. Peng. [Understanding LLM Parameter Update Sparsity through the Lens of Fisher](https://arxiv.org/abs/2609.36262). arXiv, 2026.
- S. Mukherjee, L. Yuan, D. Hakkani-Tür, H. Peng. [Reinforcement Learning Finetunes Small Subnetworks in Large Language Models](https://arxiv.org/abs/2505.11711). NeurIPS 2025.
- A. Karan, S. Chen, Y. Du. [Finetuning with Sampling: SFT Learns Better Than You Think](https://arxiv.org/abs/2610.02140). arXiv, 2026.
- Y. Li. [Logit Dynamics in Softmax Policy Gradient Methods](https://arxiv.org/abs/2506.12912). arXiv, 2025.
- J. Liu, Y. Li, et al. [When Speed Kills Stability: Demystifying RL Collapse from the Training-Inference Mismatch](/rl-collapse). Blog, 2025.
- Y. Li, J. Liu, et al. [Trust Region Masking for Long-Horizon LLM Reinforcement Learning](https://arxiv.org/abs/2512.23075). ICML 2026.
