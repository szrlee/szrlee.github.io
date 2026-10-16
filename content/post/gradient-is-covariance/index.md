---
title: "The Gradient Is a Covariance"
subtitle: "Why Post-Training Is Sparse, Part 1: the score, the Fisher, and a bound that holds for any reward"
summary: "On the sampler's own samples the score has zero mean, so the expected policy gradient is the covariance between the learning signal and the score, and the Fisher is the score's variance. A parameter can move only where its Fisher is large; whether it does is up to the reward. Off-policy, all of this holds once the score is centered under the sampling distribution."
authors:
  - admin
tags:
  - Reinforcement Learning
  - Policy Gradient
  - Fisher Information
  - Score Centering
  - Sparsity
  - Language Models
categories:
  - Research
  - Theory
date: '2026-10-15T00:00:00Z'
lastmod: '2026-10-15T00:00:00Z'
math: true
draft: false
series:
  - Why Post-Training Is Sparse
series_order: 1
---

{{% callout note %}}
**Why Post-Training Is Sparse · Part 1 of 3.** [Part 2](/post/sparsity-score-centering/) finds where the sparsity comes from. [Part 3](/post/what-policy-gradient-can-reach/) asks what policy gradient can reach.

- RL, on-policy distillation and SFT on near-policy data change only a small fraction of parameters. **Small** and **sparse** are different claims.
- On the sampler's own samples the score has **zero mean**. So the expected gradient is a **covariance**, the Fisher is a **variance**, and the Fisher caps the gradient for any reward.
- At one position the Fisher is the **variance of logit sensitivities**: whatever moves every candidate together counts for nothing.
- None of this needs on-policy sampling. It needs a **score centered under the sampling distribution**.
{{% /callout %}}

The figures step through like slides. Use **→** or **▶**, and step back with **←**.

| Symbol | Meaning |
|---|---|
| {{< math >}}$\smu(a\mid s)${{< /math >}} | the distribution that drew the next token after prefix {{< math >}}$s${{< /math >}}: the sampler |
| {{< math >}}$\pmodel(a\mid s)${{< /math >}} | the model's next-token distribution; {{< math >}}$\smu=\pmodel${{< /math >}} on-policy |
| {{< math >}}$\kappa_{i,a}=\partial z_a/\partial\theta_i${{< /math >}} | how strongly parameter {{< math >}}$i${{< /math >}} moves the logit {{< math >}}$z_a${{< /math >}} of token {{< math >}}$a${{< /math >}} |
| {{< math >}}$A${{< /math >}} | the learning signal: a reward or advantage, possibly times an importance ratio |
| {{< math >}}$G_i,\ F_i${{< /math >}} | the expected update and the diagonal Fisher of parameter {{< math >}}$i${{< /math >}} |

## 1. Small is not sparse

**Reweighting what the model already samples explains why the update is small, not why it lands on a few coordinates.**

RL fine-tuning changes only 5–30% of a model's parameters, across seven algorithms and ten models ([Mukherjee et al., 2025](https://arxiv.org/abs/2505.11711)). On-policy distillation and SFT on near-policy data behave the same way ([Zhang, Mukherjee & Peng, 2026](https://arxiv.org/abs/2609.36262)). Two common explanations are that RL only reweights answers the model can already produce, and that a scalar advantage carries at most {{< math >}}$\log_2 B${{< /math >}} bits per episode ([Information Bandwidth](/post/information-bandwidth-rl/)). The first says the update is small, the second that it carries little. Neither makes it sparse: a hundred million parameters each moving a little is small and dense, and can carry those few bits just as well. The question is which coordinates move, and why the same few can be found before any reward is seen.

## 2. The gradient is a covariance, the Fisher a variance

**On the model's own samples the score averages to zero, so the expected gradient is a covariance and the Fisher is a variance.**

{{< stepfig "cov" >}}

{{< math >}}
$$
\mathrm{score}_i=\frac{\partial\log\pi_\theta(y\mid x)}{\partial\theta_i},
$$
$$
\mathbb{E}_{y\sim\pi_\theta}\big[\mathrm{score}_i\big]=\sum_y\partial_i\pi_\theta(y\mid x)=\partial_i 1=0,
$$
$$
G_i=\mathbb{E}\big[A\cdot\mathrm{score}_i\big]=\operatorname{Cov}(A,\mathrm{score}_i),
$$
$$
F_i=\mathbb{E}\big[\mathrm{score}_i^2\big]=\operatorname{Var}(\mathrm{score}_i),
$$
$$
|G_i|\le\operatorname{std}(A)\sqrt{F_i}.
$$
{{< /math >}}

Probabilities sum to one, so whatever a parameter does, the responses it raises are paid for by others, and weighted by probability the changes cancel. With the score's mean at zero, the policy gradient {{< math >}}$\mathbb{E}[A\cdot\mathrm{score}_i]${{< /math >}} is a covariance and the Fisher is a variance; Cauchy–Schwarz gives the bound. A baseline that depends only on the prompt changes nothing in {{< math >}}$G_i${{< /math >}}, since {{< math >}}$\operatorname{Cov}(A-b,\mathrm{score}_i)=\operatorname{Cov}(A,\mathrm{score}_i)${{< /math >}}. It changes only {{< math >}}$\operatorname{std}(A)${{< /math >}} and the noise of the estimate, and the [Optimal Token Baseline](/optimal-token-baseline) is the choice that minimizes that noise.

A large Fisher is necessary for a large gradient, not sufficient. After "She drank a cup of", the output rows for *tea* and *coffee* have a large Fisher; if the reward does not care which one is chosen, their gradient is zero. Measured before RL training, the top 1% of coordinates by Fisher hold 47.40% of the top 1% by gradient magnitude, against 1% at random ([Zhang, Mukherjee & Peng, 2026](https://arxiv.org/abs/2609.36262)). The Fisher says where learning can happen; the reward decides where it does.

<details class="spoiler">
<summary>Proof, and the four assumptions it uses</summary>

The response space is finite (finite vocabulary, bounded length), so every exchange of a derivative and a sum below is exact. Probabilities are positive, which softmax sampling at temperature 1 without top-k or top-p guarantees. The reward {{< math >}}$r(x,y)${{< /math >}} and the prompt distribution do not depend on {{< math >}}$\theta${{< /math >}}.

{{< math >}}
$$
\mathbb{E}_{y\sim\pi_\theta}[\mathrm{score}_i]=\sum_y\pi_\theta\,\frac{\partial_i\pi_\theta}{\pi_\theta}=\partial_i\sum_y\pi_\theta=0,
$$
$$
G_i=\partial_i\,\mathbb{E}_x\Big[\sum_y\pi_\theta\,r\Big]
$$
$$
=\mathbb{E}_x\Big[\sum_y\pi_\theta\,\mathrm{score}_i\,r\Big]=\mathbb{E}[r\cdot\mathrm{score}_i].
$$
{{< /math >}}

Subtracting {{< math >}}$\mathbb{E}[r]\,\mathbb{E}[\mathrm{score}_i]=0${{< /math >}} turns the product into a covariance, and the same step turns {{< math >}}$\mathbb{E}[\mathrm{score}_i^2]${{< /math >}} into a variance. Both hold per prompt, so {{< math >}}$G_i=\mathbb{E}_x[\operatorname{Cov}(r,\mathrm{score}_i\mid x)]${{< /math >}}, which is why a baseline {{< math >}}$b(x)${{< /math >}} drops out.

The Fisher is also the expected curvature: {{< math >}}$\partial_j\mathrm{score}_i=\partial_i\partial_j\pi_\theta/\pi_\theta-\mathrm{score}_i\,\mathrm{score}_j${{< /math >}}, and the first term has mean {{< math >}}$\sum_y\partial_i\partial_j\pi_\theta=0${{< /math >}}, so {{< math >}}$F_{ij}=-\mathbb{E}[\partial_i\partial_j\log\pi_\theta]${{< /math >}}. It must be estimated on responses the model samples. On data labels the same formula gives the empirical Fisher, a different quantity ([Part 2](/post/sparsity-score-centering/)).

</details>

## 3. One position: the Fisher is a spread of sensitivities

**At one prefix, a parameter's token Fisher is the variance of how strongly it moves each candidate's logit.**

{{< stepfig "tokfisher" >}}

{{< math >}}
$$
\frac{\partial\log\pmodel(a\mid s)}{\partial\theta_i}=\kappa_{i,a}-\bar\kappa^{p}_i,
$$
$$
\bar\kappa^{p}_i=\sum_a\pmodel(a\mid s)\,\kappa_{i,a},
$$
$$
f_i(s)=\operatorname{Var}_{a\sim\pmodel}\big[\kappa_{i,a}\big]
$$
$$
=\tfrac12\sum_{a,b}\pmodel(a)\,\pmodel(b)\,\big(\kappa_{i,a}-\kappa_{i,b}\big)^2 .
$$
{{< /math >}}

The softmax subtracts a mean: the token score is how strongly {{< math >}}$\theta_i${{< /math >}} moves the drawn token's logit, minus how strongly it moves the average candidate. Without that subtraction the second moment would be {{< math >}}$f_i(s)+(\bar\kappa^{p}_i)^2${{< /math >}}. The removed part is the common component, which raises every logit together and changes no probability. The pairwise form says the same thing: only differences between likely candidates count. So a parameter has a small token Fisher exactly when it acts about equally on the candidates the model finds likely, either because the model is confident or because those candidates react alike.

## 4. Along the response: the Fisher adds up

**Across positions the cross terms average to zero, so the Fisher is a sum of token Fishers.**

{{< stepfig "chain" >}}

{{< math >}}
$$
\mathrm{score}_i=\sum_t\mathrm{score}_i^{(t)},
$$
$$
\mathbb{E}\big[\mathrm{score}_i^{(u)}\,\mathrm{score}_i^{(t)}\big]
$$
$$
=\mathbb{E}\Big[\mathrm{score}_i^{(u)}\;\underbrace{\mathbb{E}\big[\mathrm{score}_i^{(t)}\mid s_t\big]}_{=0}\Big]=0\quad(u\lt t),
$$
$$
F_i=\mathbb{E}\Big[\sum_t f_i(s_t)\Big].
$$
{{< /math >}}

Given the prefix {{< math >}}$s_t${{< /math >}}, the next token is drawn from {{< math >}}$\pmodel(\cdot\mid s_t)${{< /math >}}, so its score averages to zero. For {{< math >}}$u\lt t${{< /math >}} the earlier token is part of that prefix: its score is already fixed and comes out of the inner expectation. The cancellation holds in expectation, not sample by sample. It makes the Fisher grow with length but treats every parameter alike, so it sets the scale, not which coordinates are large.

## 5. What this needs: a centered score

**None of this needs on-policy sampling. It needs the score centered under the distribution that drew the samples.**

{{< stepfig "centering" >}}

{{< math >}}
$$
c_i^{(t)}=\kappa_{i,y_t}-\bar\kappa^{\mu}_i(s_t),\qquad \bar\kappa^{\mu}_i=\sum_a\smu(a\mid s)\,\kappa_{i,a},
$$
$$
G_i=\mathbb{E}_{\smu}\big[A\,c_i\big]=\operatorname{Cov}_{\smu}(A,c_i),
$$
$$
|G_i|\le\operatorname{std}_{\smu}(A)\sqrt{F^{\mu}_i},
$$
$$
F^{\mu}_i=\mathbb{E}_{\smu}\Big[\sum_t\operatorname{Var}_{a\sim\smu}\big[\kappa_{i,a}(s_t)\big]\Big].
$$
{{< /math >}}

Off-policy, the token was drawn by an inference engine or a stale model, and under that sampler the softmax score averages to {{< math >}}$\bar\kappa^{\mu}_i-\bar\kappa^{p}_i${{< /math >}}, not to zero. Subtracting it at every prefix is score centering. Sections 2–4 then go through word for word with {{< math >}}$\smu${{< /math >}} in place of {{< math >}}$\pmodel${{< /math >}}, and on-policy the centering comes free. This is the gradient of the covariance loss in [Around the Sampler](/post/offpolicy-around-the-sampler/#4-score-centering): the drawn token is raised and every token is lowered by how often the sampler draws it. It needs the sampler's full next-token distribution at every prefix, which engine log-probabilities and stale models provide and human demonstrations do not.

<details class="spoiler">
<summary>The theorem, proved</summary>

For any sampler {{< math >}}$\smu${{< /math >}} whose full next-token distribution is known at every prefix:

1. Zero conditional mean, by construction: {{< math >}}$\mathbb{E}_{a\sim\smu}\big[\kappa_{i,a}-\bar\kappa^{\mu}_i\big]=0${{< /math >}}.
2. Cross terms vanish: for {{< math >}}$u\lt t${{< /math >}}, {{< math >}}$c_i^{(u)}${{< /math >}} is fixed by {{< math >}}$s_t${{< /math >}}, so {{< math >}}$\mathbb{E}_{\smu}\big[c_i^{(u)}c_i^{(t)}\big]=\mathbb{E}_{\smu}\big[c_i^{(u)}\,\mathbb{E}_{\smu}[c_i^{(t)}\mid s_t]\big]=0${{< /math >}}.
3. The Fisher is a variance: by 2 only diagonal terms survive, and by 1 each is {{< math >}}$\operatorname{Var}_{a\sim\smu}[\kappa_{i,a}(s_t)]${{< /math >}}. By 1 and the tower property {{< math >}}$\mathbb{E}_{\smu}[c_i]=0${{< /math >}}, so the second moment is the variance.
4. The update is a covariance: {{< math >}}$\operatorname{Cov}_{\smu}(A,c_i)=\mathbb{E}_{\smu}[A\,c_i]=G_i${{< /math >}}, and Cauchy–Schwarz gives the bound.

On-policy, {{< math >}}$c_i=\mathrm{score}_i${{< /math >}} and {{< math >}}$G_i${{< /math >}} is the gradient. Off-policy, {{< math >}}$G_i${{< /math >}} is the expected update of the centered estimator: the gradient of the covariance loss, not in general of the expected reward.

</details>

**Next.** [Part 2](/post/sparsity-score-centering/) puts this to work. In the output layer the Fisher is an exact coin flip, concentrated on the tokens the sampler emits; in hidden layers it is low-dimensional; and without centering a mean shift can push gradient where the Fisher cannot see.

## References

- Y. Zhang, S. Mukherjee, H. Peng. [Understanding LLM Parameter Update Sparsity through the Lens of Fisher](https://arxiv.org/abs/2609.36262). arXiv, 2026.
- S. Mukherjee, L. Yuan, D. Hakkani-Tür, H. Peng. [Reinforcement Learning Finetunes Small Subnetworks in Large Language Models](https://arxiv.org/abs/2505.11711). NeurIPS 2025.
- M. Marek, M. Ryabinin. [Score Centering Stabilizes Off-policy Reinforcement Learning](https://arxiv.org/abs/2609.20807). arXiv, 2026.
- Y. Li, J. Xu, et al. [The Optimal Token Baseline: Variance Reduction for Long-Horizon LLM-RL](https://arxiv.org/abs/2602.07078). ICML 2026.
- Y. Li. [Information Bandwidth in Reinforcement Learning](/post/information-bandwidth-rl/). Blog, 2025.

## Citation

```bibtex
@online{li-2026-gradient-is-covariance,
  title  = {The Gradient Is a Covariance},
  author = {Li, Yingru},
  year   = {2026},
  month  = oct,
  url    = {https://richardli.xyz/post/gradient-is-covariance/}
}
```
