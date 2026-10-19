---
title: "Sparsity Requires Score Centering"
subtitle: "Why Post-Training Is Sparse, Part 2: where the Fisher concentrates, and what escapes the cap without centering"
summary: "Sparsity needs two ingredients. A score centered under the sampler makes the Fisher a cap on each parameter's update, and the Fisher is concentrated when the sampler puts its mass on a few likely tokens that react alike. Without centering, a mean shift escapes the cap. Supervised fine-tuning is all mean shift, because its signal is constant, so centering would leave nothing to learn."
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
**Why Post-Training Is Sparse · Part 2 of 3.** Starts from the cap derived in [Part 1](/post/gradient-is-covariance/); [Part 3](/post/what-policy-gradient-can-reach/) asks what policy gradient can reach.

- **Two ingredients.** Centering under the sampler makes the Fisher a cap on each parameter's expected update, and a concentrated Fisher leaves room under that cap for only a few parameters. The reward picks which of those move.
- **Centering can be done on either side.** On-policy sampling centers the score for free. Off-policy, subtract the score's average under the sampler, weight each token by its importance ratio, or subtract a per-prefix value baseline from the signal; the baseline needs no sampler probabilities.
- **A confident sampler concentrates the Fisher.** In the output layer the Fisher sits exactly on the rows of the tokens the sampler emits. In hidden layers it sits on the few directions that tell those tokens apart, and shrinks further when they react alike; experiments also find it on few coordinates. A diffuse sampler, such as high-temperature sampling, spreads it.
- **On-policy policy gradient has both ingredients in practice.** Its score is centered by construction, and its sampler, the model, is confident, more so as reinforcement learning (RL) sharpens it.
- **Without centering**, a mean shift adds up along the sequence and escapes the cap. Under training–inference mismatch it is numerical noise. In supervised fine-tuning (SFT) it is the whole gradient, because the signal is constant, so centering would erase it whether or not the sampler is known. SFT stays sparse when the data is close to the model; weighting each token by the model's probability also makes it sparse, but changes what it learns.
{{% /callout %}}

**Starting point.** Write {{< math >}}$\smu${{< /math >}} for the sampler, {{< math >}}$\pmodel${{< /math >}} for the model's next-token distribution, {{< math >}}$A${{< /math >}} for the learning signal (a reward or advantage), and {{< math >}}$\kappa_{i,a}=\partial z_a/\partial\theta_i${{< /math >}} for how strongly parameter {{< math >}}$i${{< /math >}} moves the logit of token {{< math >}}$a${{< /math >}}. For the token {{< math >}}$y_t${{< /math >}} drawn after prefix {{< math >}}$s_t${{< /math >}}, the softmax score subtracts the average under the model, {{< math >}}$\mathrm{score}_i^{(t)}=\kappa_{i,y_t}-\bar\kappa^{p}_i(s_t)${{< /math >}}; centering subtracts the average under the sampler, {{< math >}}$c_i^{(t)}=\kappa_{i,y_t}-\bar\kappa^{\mu}_i(s_t)${{< /math >}}. Over a response of {{< math >}}$T${{< /math >}} tokens, {{< math >}}$c_i=\sum_{t=1}^{T}c_i^{(t)}${{< /math >}}. With centering, the expected update is a covariance and the Fisher caps it ([Part 1](/post/gradient-is-covariance/) has the derivation):

{{< math >}}
$$
G_i=\operatorname{Cov}_{\smu}(A,c_i),\qquad |G_i|\le\operatorname{std}_{\smu}(A)\sqrt{F^{\mu}_i},
$$
$$
F^{\mu}_i=\mathbb{E}_{\smu}\Big[\sum_t f_i(s_t)\Big],
$$
$$
f_i(s)=\operatorname{Var}_{a\sim\smu}\big[\kappa_{i,a}(s)\big].
$$
{{< /math >}}

Sections 1 and 2 show where and why the Fisher concentrates; section 3 shows what goes wrong without centering.

This concerns the expected update; whether a stored weight changes also depends on Adam and on precision. [Mukherjee et al. (2025)](https://arxiv.org/abs/2505.11711) count a bf16 weight as unchanged within {{< math >}}$10^{-5}${{< /math >}}, and note that at unlimited precision the updates would be dense.

## 1. Output layer: an exact coin flip

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

The weight {{< math >}}$W_{c,k}${{< /math >}} touches only token {{< math >}}$c${{< /math >}}'s logit, so for it the next token is a coin flip, {{< math >}}$c${{< /math >}} or not {{< math >}}$c${{< /math >}}, and its Fisher is the coin's variance times {{< math >}}$h_k^2${{< /math >}}: near zero when the sampler almost never considers {{< math >}}$c${{< /math >}}, or is almost sure of it. With tea at 0.5, coffee at 0.4 and 0.1 spread over a hundred thousand other tokens, the tea and coffee rows hold 83% of the total, and every other row about a millionth.

The factor {{< math >}}$\smu(c)${{< /math >}} comes from the sampler and decides which rows matter; even the raw second moment {{< math >}}$\smu(c)\,h_k^2${{< /math >}}, with no mean subtracted, sits on the same rows. A diffuse sampler, such as high-temperature sampling, spreads {{< math >}}$\smu(c)${{< /math >}}, and with it the Fisher, however well the score is centered. Centering contributes {{< math >}}$1-\smu(c)${{< /math >}}, which silences a token the sampler is sure of; its main job is to make the cap hold at all ([section 3](#3-without-centering-the-mean-shift)).

Along a response, {{< math >}}$\mathbb{E}[\smu(c\mid s_t)]=\Pr(y_t=c)${{< /math >}}, so a row's Fisher is bounded by how often the sampler emits its token. A task emits only a small part of the vocabulary, and the rows of the other tokens barely move.

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

The total {{< math >}}$1-\sum_c\smu(c)^2${{< /math >}} is also the expected squared logit update of [Logit Dynamics](https://arxiv.org/abs/2506.12912): for a drawn token {{< math >}}$c${{< /math >}} that update is {{< math >}}$1-2\smu(c)+\sum_a\smu(a)^2${{< /math >}}, and its average over {{< math >}}$c${{< /math >}} is {{< math >}}$1-\sum_a\smu(a)^2${{< /math >}}.

</details>

## 2. Hidden layers: a few directions

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

With {{< math >}}$\bar w_\mu=\sum_a\smu(a)\,w_a${{< /math >}} the average output vector, centering deletes the rank-one term {{< math >}}$\bar w_\mu\bar w_\mu^\top${{< /math >}}: what a parameter does to the average likely logit, {{< math >}}$\bar\kappa^{\mu}_i=\bar w_\mu^\top u_i${{< /math >}}. Raising every likely logit alike moves no probability, so that common part counts for nothing, and what remains is the covariance of the candidates. If the sampler puts mass on {{< math >}}$n${{< /math >}} tokens, their deviations from the mean span at most {{< math >}}$n-1${{< /math >}} directions, one fewer than the {{< math >}}$n${{< /math >}} of the raw second moment.

So the hidden-layer Fisher is small for two reasons: the sampler puts its mass on few tokens, and those tokens react alike, as Zhang et al. find for likely next tokens. Under the model's own probabilities the token Fisher is 0.16 of its value under a uniform distribution at the start of RL and 0.04 after 1,000 steps ([Zhang, Mukherjee & Peng, 2026](https://arxiv.org/abs/2609.36262)): the model is a confident sampler, and RL makes it more so. How much each reason contributes is still open ([section 6](#6-scope-and-open-questions)).

Few directions become few coordinates only if most {{< math >}}$u_i${{< /math >}} are nearly orthogonal to these directions, which depends on the architecture and on pretraining, so here the evidence is empirical. In RL on Qwen3-1.7B-Base, training only the top 0.1% of parameters by initial Fisher recovers 85.21% of the full-training gain on MATH-500, and 5% recovers 98.82% ([Zhang, Mukherjee & Peng, 2026](https://arxiv.org/abs/2609.36262)). Sparsity and low rank are separate properties: the measured updates touch few coordinates and are nearly full rank in every matrix ([Mukherjee et al., 2025](https://arxiv.org/abs/2505.11711)).

## 3. Without centering: the mean shift

**Without centering, a mean shift adds up along the sequence and can push gradient into directions the Fisher cannot see.**

{{< stepfig "meanshift" >}}

{{< math >}}
$$
m_i(s)=\bar\kappa^{\mu}_i(s)-\bar\kappa^{p}_i(s),
$$
$$
\mathbb{E}_{\smu}\big[A\,\mathrm{score}_i^{(t)}\mid s_t\big]
$$
$$
=\operatorname{Cov}_{\smu}\big(A,\kappa_{i,y_t}\mid s_t\big)+\mathbb{E}_{\smu}[A\mid s_t]\,m_i(s_t),
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

Draw tokens from {{< math >}}$\smu${{< /math >}} but subtract only the model's mean, as the plain softmax score does. At each prefix the expected update then splits in two: the covariance of the signal with the sensitivities, which the sampler's Fisher caps, and the average signal times the mean shift, which escapes the cap. Centering removes the second part from either side. Subtracting {{< math >}}$\bar\kappa^{\mu}_i${{< /math >}} from the score needs the sampler's full distribution; subtracting the per-prefix value {{< math >}}$V^{\mu}(s_t)=\mathbb{E}_{\smu}[A\mid s_t]${{< /math >}} from the signal needs only a baseline fitted on the samples, and gives the same expected update. Left in, the mean shift also adds up across positions: with per-position variance {{< math >}}$\sigma^2${{< /math >}} and a shift {{< math >}}$m${{< /math >}} of constant sign, the second moment is roughly {{< math >}}$T\sigma^2+T^2m^2${{< /math >}} against {{< math >}}$T\sigma^2${{< /math >}} with centering.

SFT is the extreme case. Its sampler is the data's next-token distribution {{< math >}}$q${{< /math >}}, unknown since each prefix shows a single token, and its signal is constant, {{< math >}}$A\equiv1${{< /math >}}. The covariance is then zero and the average signal is one, so by the tower property the gradient is exactly the accumulated shift:

{{< math >}}
$$
\frac{\partial}{\partial\theta_i}\,\mathbb{E}_{\text{data}}\Big[\sum_t\log\pmodel(y_t\mid s_t)\Big]=\mathbb{E}_{\text{data}}\Big[\sum_t m_i(s_t)\Big].
$$
{{< /math >}}

Centering would remove all of it, from either side: a constant signal is its own value baseline, and {{< math >}}$\operatorname{Cov}_{\smu}(1,c_i)=0${{< /math >}}. Knowing {{< math >}}$q${{< /math >}} makes no difference, and centering with an estimate of {{< math >}}$q${{< /math >}} only scales the gradient down. Imitation is the mean shift, so how sparse SFT is depends on how far the data is from the model. The measurements agree: SFT on rejection-sampled, in-distribution data leaves 91% of parameters unchanged, SFT on out-of-distribution data 7% ([Mukherjee et al., 2025](https://arxiv.org/abs/2505.11711)). In the output layer a demonstrated token {{< math >}}$y^\star${{< /math >}} that the model almost never produces gets a nearly full-size push, {{< math >}}$(1-\pmodel(y^\star))\,h${{< /math >}}, on a row whose on-policy Fisher is about {{< math >}}$\pmodel(y^\star)${{< /math >}}: data that leaves the model's distribution touches parameters the model itself never touches.

<details class="spoiler">
<summary>The same cost, booked as a χ² signal</summary>

Let {{< math >}}$D${{< /math >}} be the data distribution over responses, {{< math >}}$\pi_\theta${{< /math >}} the model's, and {{< math >}}$F_i${{< /math >}} the model's own Fisher. Written under the model, the SFT gradient weights each response by {{< math >}}$D/\pi_\theta${{< /math >}}. Because the score has zero mean under {{< math >}}$\pi_\theta${{< /math >}}, any constant {{< math >}}$b${{< /math >}} can be subtracted from that weight:

{{< math >}}
$$
G^{\text{SFT}}_i=\mathbb{E}_{\pi_\theta}\big[(D/\pi_\theta)\,\mathrm{score}_i\big]
$$
$$
=\mathbb{E}_{\pi_\theta}\big[(D/\pi_\theta-b)\,\mathrm{score}_i\big],
$$
$$
|G^{\text{SFT}}_i|\le\sqrt{\mathbb{E}_{\pi_\theta}\big[(D/\pi_\theta-b)^2\big]}\,\sqrt{F_i}.
$$
{{< /math >}}

With {{< math >}}$\chi^2(D\,\|\,\pi_\theta)=\mathbb{E}_{\pi_\theta}\big[(D/\pi_\theta-1)^2\big]${{< /math >}}, the expectation under the root is {{< math >}}$\chi^2(D\,\|\,\pi_\theta)+(1-b)^2\ge\chi^2(D\,\|\,\pi_\theta)${{< /math >}}. The weight {{< math >}}$D/\pi_\theta${{< /math >}} is a product over positions of {{< math >}}$q(y_t\mid s_t)/\pmodel(y_t\mid s_t)${{< /math >}}, where {{< math >}}$q${{< /math >}} is the data's next-token distribution, and {{< math >}}$1+\chi^2(D\,\|\,\pi_\theta)=\mathbb{E}_D[D/\pi_\theta]${{< /math >}}. At the last position,

{{< math >}}
$$
\mathbb{E}_D\Big[\frac{q(y_T\mid s_T)}{\pmodel(y_T\mid s_T)}\,\Big|\,s_T\Big]
$$
$$
=1+\chi^2\big(q(\cdot\mid s_T)\,\|\,\pmodel(\cdot\mid s_T)\big).
$$
{{< /math >}}

If every position has {{< math >}}$\chi^2(q\,\|\,\pmodel)\ge c${{< /math >}}, backward induction over positions, with every ratio positive, gives

{{< math >}}
$$
1+\chi^2(D\,\|\,\pi_\theta)\ \ge\ (1+c)^T .
$$
{{< /math >}}

The bound still holds, but its signal scale grows at least exponentially in the number of positions where the data leaves the model's distribution. It is the same {{< math >}}$(1+\chi^2)^T${{< /math >}} form as the variance of sequence-level importance sampling in [rl-collapse Part 2](/post/rl-collapse-part2/), and the {{< math >}}$\chi^2${{< /math >}} that makes per-token importance sampling noisy in [Around the Sampler](/post/offpolicy-around-the-sampler/#5-per-token-importance-sampling-same-tangent-more-variance). Booked directly under the data, the cost moves into the empirical Fisher {{< math >}}$\mathbb{E}_D[\mathrm{score}_i^2]${{< /math >}}, which a small {{< math >}}$F_i${{< /math >}} does not bound.

</details>

<details class="spoiler">
<summary>Centering SFT from data, and what does make it sparse</summary>

Estimating the sampler does not help. Centering with an estimate {{< math >}}$\hat q${{< /math >}} of the data's next-token distribution leaves {{< math >}}$\sum_a\big(q(a)-\hat q(a)\big)\kappa_{i,a}${{< /math >}} at each prefix; for {{< math >}}$\hat q=\lambda q+(1-\lambda)\pmodel${{< /math >}} that is {{< math >}}$1-\lambda${{< /math >}} times the SFT gradient, on the same coordinates.

What does change the picture is moving the data from the sampler into the signal. With {{< math >}}$\mathrm{score}_i(a)=\kappa_{i,a}-\bar\kappa^{p}_i${{< /math >}}, which has zero mean under the model, any per-token weight {{< math >}}$w${{< /math >}} turns SFT into a covariance under the model, with prefixes still taken from the data:

{{< math >}}
$$
\mathbb{E}_{a\sim q}\big[w(a)\,\mathrm{score}_i(a)\big]
$$
$$
=\operatorname{Cov}_{a\sim\pmodel}\big(R,\ \mathrm{score}_i\big),\qquad R=\frac{q\,w}{\pmodel}.
$$
{{< /math >}}

The cap's signal scale is {{< math >}}$\sqrt{\mathbb{E}_{\pmodel}[R^2]}${{< /math >}}. Plain SFT, {{< math >}}$w=1${{< /math >}}, gives {{< math >}}$\mathbb{E}_{\pmodel}[R^2]=1+\chi^2(q\,\|\,\pmodel)${{< /math >}}, the per-position {{< math >}}$\chi^2${{< /math >}} of the note above. It equals the squared length of the mean shift {{< math >}}$q-\pmodel${{< /math >}} measured with the pseudo-inverse of the logit Fisher {{< math >}}$\operatorname{diag}(\pmodel)-\pmodel\pmodel^\top${{< /math >}}, so the {{< math >}}$\chi^2${{< /math >}} and the mean shift are one cost written two ways. Weighting each data token by its model probability, {{< math >}}$w=\pmodel(y_t)^{\alpha}${{< /math >}} held fixed in the gradient, gives

{{< math >}}
$$
\mathbb{E}_{\pmodel}[R^2]=\sum_a q(a)^2\,\pmodel(a)^{2\alpha-1}\le1\qquad\big(\alpha\ge\tfrac12\big),
$$
{{< /math >}}

however far the data is from the model. The update is then capped by the model's Fisher with a signal of order one, as in on-policy policy gradient; {{< math >}}$\alpha=1${{< /math >}} is Dynamic Fine-Tuning ([Wu et al., 2025](https://arxiv.org/abs/2508.05629)).

Two costs come with it. The target moves: with one logit per token the fixed point is {{< math >}}$\pmodel\propto q^{1/(1-\alpha)}${{< /math >}}, the mode of {{< math >}}$q${{< /math >}} at {{< math >}}$\alpha=1${{< /math >}}, and any weight computed from the model that varies across tokens makes {{< math >}}$\pmodel=q${{< /math >}} non-stationary, since there the logit update is {{< math >}}$\pmodel(b)\big(w(b)-\mathbb{E}_{\pmodel}[w]\big)${{< /math >}}. And the push on a token the model almost never produces shrinks like {{< math >}}$\pmodel^{\alpha}${{< /math >}}, the limit [Part 3](/post/what-policy-gradient-can-reach/) derives for policy gradient. On Natural Questions, SFT raises accuracy from 31.24% to 36.62% while Dynamic Fine-Tuning lowers it to 30.14% (Wu et al., 2025, §4.5). Within one expected step, then, only what the model's Fisher can see is learned sparsely; keeping the SFT target and sparsity means changing the data, as rejection sampling does.

</details>

## 4. Four cases side by side

**The bound belongs to the estimator: it holds whenever centering removes the mean shift.**

| Case | Samples from | Score centered under | Fisher bound |
|---|---|---|---|
| On-policy policy gradient, including reverse-KL distillation | the model | the model, for free | holds, with the model's Fisher |
| Off-policy policy gradient with score centering | a known sampler | the sampler | holds, with the sampler's Fisher |
| SFT | the data, an unknown sampler | the model only; centering would leave nothing | holds formally; the signal scale is set by the {{< math >}}$\chi^2${{< /math >}} divergence of the data from the model |
| Value-based temporal-difference (TD) learning | a replay buffer | nothing | does not apply |

On-policy distillation with a reverse-KL loss is policy gradient whose per-token reward is the teacher's log-probability minus the model's, so it shares the first row. With a forward-KL loss, the gradient at each prefix is the shift toward the teacher's next-token distribution, and distillation behaves like SFT. Sharing a row means sharing the cap; how much of it an update uses depends on the signal. In a controlled comparison on Qwen3-8B, 77.2% of weights stay unchanged at bf16 precision under RL with verifiable rewards (RLVR), 51.6% under on-policy distillation and 8.1% under SFT ([Shen et al., 2026](https://arxiv.org/abs/2606.07082)). RLVR and distillation both sample from the model; RLVR gives one coefficient per response, distillation one per token.

TD weights a raw logit gradient by the TD error {{< math >}}$\delta${{< /math >}}, {{< math >}}$\mathbb{E}[\delta\cdot\kappa_{i,a}]${{< /math >}}, which keeps the common component, since for a Q-value a uniform shift carries the value level. The bound does not apply, which leaves open whether TD updates are sparse. DPO (direct preference optimization) on a fixed preference set also has an unknown sampler and behaves like SFT: sparse on in-distribution data, dense on out-of-distribution data ([Mukherjee et al., 2025](https://arxiv.org/abs/2505.11711)). What separates the first three rows is the mean shift: zero when the score is centered, the whole signal in SFT.

## 5. Training–inference mismatch {#5-sparsity-is-not-a-defect}

**Under training–inference mismatch, centering is what keeps numerical noise out of the update.**

When the sampler is an inference engine, {{< math >}}$\smu\approx\pmodel${{< /math >}}, and the shift {{< math >}}$m_i${{< /math >}} comes only from numerical differences between two implementations. It carries no reward information, yet it accumulates along the sequence and pushes directions with near-zero on-policy Fisher. In the output layer the shift for row {{< math >}}$c${{< /math >}} is {{< math >}}$(\smu(c)-\pmodel(c))\,h${{< /math >}}: where the engine draws a token more often than the model would, that row is pushed although its on-policy Fisher is near zero, consistent with the mismatch doing most of its damage on rare tokens ([When Speed Kills Stability](/rl-collapse)).

Centering removes the shift exactly. Without the engine's full distribution, per-token importance weights do the same job with only the drawn token's two probabilities: with {{< math >}}$\rho_t=\pmodel(y_t\mid s_t)/\smu(y_t\mid s_t)${{< /math >}}, {{< math >}}$\mathbb{E}_{\smu}\big[\rho_t(\kappa_{i,y_t}-\bar\kappa^{p}_i)\big]=\sum_a\pmodel(a)\big(\kappa_{i,a}-\bar\kappa^{p}_i\big)=0${{< /math >}} at every prefix. Clipping a token's ratio breaks this; [Trust Region Masking](/post/trust-region-masking/) never clips a token and only drops whole sequences that left the trust region. A per-prefix value baseline removes the shift too, as far as it matches {{< math >}}$V^{\mu}(s_t)${{< /math >}}, which PPO's per-token critic aims for; a baseline shared by the whole response, as in GRPO, leaves the shift weighted by {{< math >}}$V^{\mu}(s_t)${{< /math >}} minus that baseline. Here sparsity means leaving alone the parameters that only noise would push.

## 6. Scope and open questions

- **What centering needs.** On the score side, the sampler's full next-token distribution at every prefix (a stale model, or an engine that returns full log-probabilities; with top-k or top-p, the truncated distribution), or per-token importance weights, which need only the drawn token's two probabilities. On the signal side, a per-prefix value baseline, which works on offline data with an unknown sampler as long as the signal varies. Demonstrations alone provide neither probabilities nor a varying signal. Estimating {{< math >}}$\bar\kappa^{\mu}${{< /math >}} or {{< math >}}$V^{\mu}${{< /math >}} from a batch makes the cancellation approximate.
- **What it changes.** Off-policy, centering drops the mean-shift term, which pulls the model toward the sampler; the coefficient {{< math >}}$\beta${{< /math >}} in [Around the Sampler](/post/offpolicy-around-the-sampler/#6-one-knob-how-hard-to-pull-toward-the-sampler), the average signal of section 3's split, sets that pull on purpose.
- **Two mechanisms, still entangled.** A concentrated sampler and candidates that react alike both lower the Fisher, and the ratio in section 2 changes both at once. Cleaner tests: permute which tokens get which probabilities at fixed entropy, and measure, per coordinate, the share of the second moment that centering removes, {{< math >}}$(\bar\kappa^{p}_i)^2/\sum_a\pmodel(a)\,\kappa_{i,a}^2${{< /math >}}; a share near 1 on low-Fisher coordinates would mean the common part dominates.
- **One step.** The Fisher itself changes during training; [Part 3](/post/what-policy-gradient-can-reach/) works out what a one-step bound implies.

**Next.** [Part 3](/post/what-policy-gradient-can-reach/) turns the bound around: how far can one update move a response the model almost never produces?

## References

- Y. Zhang, S. Mukherjee, H. Peng. [Understanding LLM Parameter Update Sparsity through the Lens of Fisher](https://arxiv.org/abs/2609.36262). arXiv, 2026.
- S. Mukherjee, L. Yuan, D. Hakkani-Tür, H. Peng. [Reinforcement Learning Finetunes Small Subnetworks in Large Language Models](https://arxiv.org/abs/2505.11711). NeurIPS 2025.
- Y. Li. [Logit Dynamics in Softmax Policy Gradient Methods](https://arxiv.org/abs/2506.12912). arXiv, 2025.
- J. Liu, Y. Li, et al. [When Speed Kills Stability: Demystifying RL Collapse from the Training-Inference Mismatch](/rl-collapse). Blog, 2025.
- Y. Li, J. Liu, et al. [Trust Region Masking for Long-Horizon LLM Reinforcement Learning](https://arxiv.org/abs/2512.23075). ICML 2026.
- Z. Shen, Y. Li, Q. Yin, et al. [On the Geometry of On-Policy Distillation](https://arxiv.org/abs/2606.07082). arXiv, 2026.
- Y. Wu, Y. Zhou, Z. Ziheng, et al. [On the Generalization of SFT: A Reinforcement Learning Perspective with Reward Rectification](https://arxiv.org/abs/2508.05629). arXiv, 2025.

## Citation

```bibtex
@online{li-2026-sparsity-score-centering,
  title  = {Sparsity Requires Score Centering},
  author = {Li, Yingru},
  year   = {2026},
  month  = oct,
  url    = {https://richardli.xyz/post/sparsity-score-centering/}
}
```
