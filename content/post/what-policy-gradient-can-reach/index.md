---
title: "What Policy Gradient Can Reach"
subtitle: "Why Post-Training Is Sparse, Part 3: a one-step bound on how far any response can move, and what it does not rule out"
summary: "One policy-gradient update moves the log-odds between two responses by at most std(A) times their score difference measured in the sampler's Fisher: a response moves only as far as it resembles what the sampler produces. With one logit per response, a rare response takes time proportional to one over its probability, while a demonstration needs only logarithmic time. The bound covers one step of one estimator; it does not say RL can only sharpen."
authors:
  - admin
tags:
  - Reinforcement Learning
  - Policy Gradient
  - Fisher Information
  - Exploration
  - Sparsity
  - Language Models
categories:
  - Research
  - Theory
date: '2026-10-15T00:20:00Z'
lastmod: '2026-10-15T00:20:00Z'
math: true
draft: false
series:
  - Why Post-Training Is Sparse
series_order: 3
---

{{% callout note %}}
**Why Post-Training Is Sparse · Part 3 of 3.** Uses the covariance form of [Part 1](/post/gradient-is-covariance/) and the Fisher of [Part 2](/post/sparsity-score-centering/).

- One update moves the log-odds of {{< math >}}$y${{< /math >}} against {{< math >}}$y'${{< /math >}} by at most {{< math >}}$\operatorname{std}_{\smu}(A)\sqrt{v^\top F^{\mu}v}${{< /math >}}, where {{< math >}}$v${{< /math >}} is their score difference.
- {{< math >}}$v^\top F^{\mu}v${{< /math >}} is the mean squared **similarity** between {{< math >}}$v${{< /math >}} and the sampler's own outputs: what the sampler never produces barely moves.
- With one logit per response, growth is **proportional to current probability**. Reaching one half from {{< math >}}$\varepsilon${{< /math >}} takes time about {{< math >}}$1/\varepsilon${{< /math >}}; a demonstration needs only about {{< math >}}$\log(1/\varepsilon)${{< /math >}}.
- The bound covers **one step of one estimator**. Shared structure, a changing Fisher, and samplers with centered estimators all widen what is reachable.
{{% /callout %}}

## 1. A question that can be answered

**Ask how much one update moves a response's log-odds against another.**

{{< math >}}
$$
\frac{d}{d\eta}\log\pi_{\theta+\eta G}(y\mid x)\Big|_{\eta=0}=\mathrm{score}(y)^\top G,
$$
$$
\frac{d}{d\eta}\Big[\log\pi(y\mid x)-\log\pi(y'\mid x)\Big]_{\eta=0}=v^\top G,\quad v=\mathrm{score}(y)-\mathrm{score}(y').
$$
{{< /math >}}

"Can RL learn something new?" is too vague to prove anything about. A version that can be proved: for a response {{< math >}}$y${{< /math >}} that the model rarely produces but the reward likes, how much can one update change its probability? The normalizer moves many responses at once, so the log-odds against a reference {{< math >}}$y'${{< /math >}} isolates learning {{< math >}}$y${{< /math >}} itself. Here {{< math >}}$G=\mathbb{E}_{\smu}[A\cdot c]${{< /math >}} is the expected update of Part 1, and {{< math >}}$\mathrm{score}(y)${{< /math >}} is the model's score of the target. In the running example: after "She drank a cup of", can *matcha* rise if the sampler never says it?

## 2. A response moves as far as it resembles the sampler's outputs

**One update moves a response only as far as its score resembles the scores of what the sampler draws.**

{{< stepfig "reach" >}}

{{< math >}}
$$
\mathrm{score}(y)^\top G=\mathbb{E}_{y''\sim\smu}\big[A(y'')\,K(y,y'')\big],\qquad K(y,y'')=\mathrm{score}(y)^\top c(y''),
$$
$$
\big|v^\top G\big|\le\operatorname{std}_{\smu}(A)\sqrt{v^\top F^{\mu}v},\qquad v^\top F^{\mu}v=\mathbb{E}_{y''\sim\smu}\Big[\big(v^\top c(y'')\big)^2\Big].
$$
{{< /math >}}

The similarity {{< math >}}$K(y,y'')${{< /math >}} says how much raising a sampled {{< math >}}$y''${{< /math >}} raises {{< math >}}$y${{< /math >}} along the way. Matcha is never sampled, but if its score shares directions with tea's, rewarding tea lifts matcha too. A response whose score is orthogonal to everything the sampler draws has {{< math >}}$v^\top F^{\mu}v\approx0${{< /math >}} and does not move to first order, however high its reward. With {{< math >}}$v${{< /math >}} a coordinate vector this is the bound of Part 1, now in any direction.

<details class="spoiler">
<summary>Proof, and the preconditioned version</summary>

By Part 1, {{< math >}}$\mathbb{E}_{\smu}[c]=0${{< /math >}}, so {{< math >}}$\mathbb{E}_{\smu}[v^\top c]=0${{< /math >}} and the mean of {{< math >}}$A${{< /math >}} can be subtracted:

{{< math >}}
$$
v^\top G=\mathbb{E}_{\smu}\big[(A-\mathbb{E}_{\smu}A)\,v^\top c\big]\le\sqrt{\operatorname{Var}_{\smu}(A)}\,\sqrt{\mathbb{E}_{\smu}\big[(v^\top c)^2\big]}=\operatorname{std}_{\smu}(A)\sqrt{v^\top F^{\mu}v}.
$$
{{< /math >}}

If the step is {{< math >}}$\Delta\theta=\eta\,DG${{< /math >}} with a diagonal preconditioner {{< math >}}$D${{< /math >}}, as in one Adam step, replace {{< math >}}$v${{< /math >}} with {{< math >}}$Dv${{< /math >}}: {{< math >}}$|v^\top DG|\le\operatorname{std}_{\smu}(A)\sqrt{v^\top DF^{\mu}D\,v}${{< /math >}}.

</details>

## 3. One logit per response: growth in proportion to probability

**With one logit per response, a response grows at a rate proportional to its own probability.**

{{< stepfig "amplify" >}}

{{< math >}}
$$
\frac{\partial\,\mathbb{E}_\pi[r]}{\partial z_y}=\pi(y)\big(r(y)-\mathbb{E}_\pi[r]\big)=\pi(y)\,\Delta(y),
$$
$$
\frac{d\log\pi(y)}{d\eta}=\pi(y)\,\Delta(y)-\sum_{y''}\pi(y'')^2\,\Delta(y''),
$$
$$
\frac{d}{d\eta}\big[\log\pi(y)-\log\pi(y')\big]=\pi(y)\,\Delta(y)-\pi(y')\,\Delta(y').
$$
{{< /math >}}

The second term comes from the normalizer and is the same for every response: it raises or lowers the whole tail without changing ratios inside it. The log-odds keep only the first term, which is small for any response the model rarely produces. The kernel bound agrees: with {{< math >}}$F=\operatorname{diag}(\pi)-\pi\pi^\top${{< /math >}} and {{< math >}}$v=e_y-e_{y'}${{< /math >}}, {{< math >}}$v^\top Fv=\pi(y)+\pi(y')-(\pi(y)-\pi(y'))^2\le\pi(y)+\pi(y')${{< /math >}}. Ignoring the normalizer, {{< math >}}$d\pi/dt\approx\Delta\,\pi^2${{< /math >}}, so {{< math >}}$1/\pi${{< /math >}} falls at rate {{< math >}}$\Delta${{< /math >}} and getting from {{< math >}}$\varepsilon${{< /math >}} to one half takes about {{< math >}}$(1/\varepsilon-2)/\Delta${{< /math >}}. This matches the theory of softmax policy gradient, whose rate depends on how much probability the optimal action keeps ([Mei et al., 2020](https://arxiv.org/abs/2005.06392)), and which can take exponential time in the worst case ([Li et al., 2021](https://arxiv.org/abs/2102.11270)).

{{< livefig "s-amplify" "Drag the sliders. Each tenfold drop in the starting probability costs policy gradient about ten times more time, and the demonstration a constant." 430 >}}

<details class="spoiler">
<summary>The two-response case, solved exactly</summary>

With two responses, rewards {{< math >}}$\Delta${{< /math >}} and 0, and log-odds {{< math >}}$\ell=\log\frac{\pi}{1-\pi}${{< /math >}} starting at {{< math >}}$\ell_0=\log\frac{\varepsilon}{1-\varepsilon}${{< /math >}}:

{{< math >}}
$$
\text{policy gradient: }\ \frac{d\ell}{dt}=2\Delta\,\pi(1-\pi),\qquad t_{1/2}=\frac{e^{-\ell_0}-2\ell_0-e^{\ell_0}}{2\Delta}\approx\frac{1}{2\varepsilon\Delta},
$$
$$
\text{demonstration: }\ \frac{d\ell}{dt}=2(1-\pi),\qquad t_{1/2}=\frac{1-\ell_0-e^{\ell_0}}{2}\approx\tfrac12\log\frac1\varepsilon .
$$
{{< /math >}}

From {{< math >}}$\varepsilon=0.01${{< /math >}} with {{< math >}}$\Delta=1${{< /math >}} these give 54.1 and 2.8; from {{< math >}}$\varepsilon=0.001${{< /math >}}, 506 and 4.0.

</details>

## 4. A demonstration injects probability directly

**A demonstration pushes a response at a rate that does not shrink with its probability.**

{{< math >}}
$$
G=\mathrm{score}(y^\star)\ \Rightarrow\ \frac{d}{d\eta}\log\pi(y^\star)\Big|_{\eta=0}=\big\|\mathrm{score}(y^\star)\big\|^2,
$$
$$
\big\|e_{y^\star}-\pi\big\|^2=\big(1-\pi(y^\star)\big)^2+\sum_{y\ne y^\star}\pi(y)^2 .
$$
{{< /math >}}

No similarity to sampled outputs enters, and the push stays near one however unlikely {{< math >}}$y^\star${{< /math >}} is. The tabular expression is the squared logit update of [Logit Dynamics](https://arxiv.org/abs/2506.12912) for a chosen action, {{< math >}}$1-2P_c+C(P)${{< /math >}}; in the output layer, row {{< math >}}$y^\star${{< /math >}} receives {{< math >}}$(1-\pmodel(y^\star))\,h${{< /math >}} ([Part 2](/post/sparsity-score-centering/#3-without-centering-the-mean-shift)). SFT can inject mass into a response the sampler never produces. Policy gradient can only amplify what it already reaches through similarity: {{< math >}}$1/\varepsilon${{< /math >}} against {{< math >}}$\log(1/\varepsilon)${{< /math >}} in the figure.

## 5. What the bound does not say

**The bound limits one first-order step of one estimator. It does not say RL can only sharpen.**

- **Shared structure.** In a network the similarity {{< math >}}$K${{< /math >}} is not diagonal. A response never sampled is lifted if it shares structure with rewarded samples, so new solutions recombined from existing pieces are within reach.
- **Many steps.** The Fisher moves. In [Zhang, Mukherjee & Peng (2026)](https://arxiv.org/abs/2609.36262), the initial high-Fisher set keeps only 70–78% of its coordinates through 1,000 steps, and the mean Fisher of the initial low-Fisher set rises by 15–17%. Directions that were invisible can open; a multi-step bound would need control over that drift, and none exists yet.
- **First order.** Large steps and strong nonlinearity add higher-order terms.
- **Which estimator, and which sampler.** With full importance weighting, the expected update is the on-policy gradient, bounded by the model's own Fisher whatever the sampler, so exploring alone does not widen a single step. With the score centered under the sampler, the update is {{< math >}}$\operatorname{Cov}_{\smu}(A,c)${{< /math >}}, bounded by {{< math >}}$F^{\mu}${{< /math >}}: a different sampler (a higher temperature, an exploration bonus, or posterior sampling as in [Exploration at Scale](/post/scalable-exploration/)) changes which directions are visible. Value-based TD learning is not score-weighted at all, so the bound does not apply to it.
- **Sparse is not little.** A new capability can live in a few parameters.

## 6. What to measure instead

**Whether policy gradient can learn a behavior depends on score similarity, not on probability alone.**

{{< math >}}
$$
v^\top F^{\mu}v=\mathbb{E}_{y''\sim\smu}\Big[\big(v^\top c(y'')\big)^2\Big],\qquad v=\mathrm{score}(y)-\mathrm{score}(y').
$$
{{< /math >}}

The debate over whether RL extends a base model's abilities usually looks at probability: at large *k*, the base model's pass@*k* can be higher ([Yue et al., 2025](https://arxiv.org/abs/2504.13837)). The bound points to a different quantity, the overlap between the target behavior's score and what the sampler produces. Among equally unlikely responses, those that share structure with high-reward samples can be amplified and isolated ones cannot. Policy gradient is similarity-driven amplification. Whether it yields new capabilities depends on how new behavior connects, through shared parameters, to what the sampler already generates, and on which sampler is used.

**Back to the start.** [Part 1](/post/gradient-is-covariance/) showed that a score centered under the sampler makes the update a covariance; [Part 2](/post/sparsity-score-centering/) showed what that covariance can touch. Choosing the sampler is the exploration problem, which [Exploration at Scale](/post/scalable-exploration/) treats on its own terms.

## References

- J. Mei, C. Xiao, C. Szepesvári, D. Schuurmans. [On the Global Convergence Rates of Softmax Policy Gradient Methods](https://arxiv.org/abs/2005.06392). ICML 2020.
- G. Li, Y. Wei, Y. Chi, Y. Chen. [Softmax Policy Gradient Methods Can Take Exponential Time to Converge](https://arxiv.org/abs/2102.11270). COLT 2021.
- Y. Yue, Z. Chen, R. Lu, et al. [Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?](https://arxiv.org/abs/2504.13837) 2025.
- Y. Li. [Logit Dynamics in Softmax Policy Gradient Methods](https://arxiv.org/abs/2506.12912). arXiv, 2025.
- Y. Zhang, S. Mukherjee, H. Peng. [Understanding LLM Parameter Update Sparsity through the Lens of Fisher](https://arxiv.org/abs/2609.36262). arXiv, 2026.
- Y. Li, J. Xu, B. Wang, Z.-Q. Luo. [Scalable Exploration via Ensemble++](https://arxiv.org/abs/2407.13195). NeurIPS 2025.
