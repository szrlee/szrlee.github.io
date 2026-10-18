---
title: "What Policy Gradient Can Reach"
subtitle: "Why Post-Training Is Sparse, Part 3: a one-step bound on how far any response can move, and its limits"
summary: "One policy-gradient update changes the log-odds between two responses at a rate of at most std(A) times their score difference measured in the sampler's Fisher: a response moves only as far as it resembles what the sampler produces. With one logit per response, a rare response takes time proportional to one over its probability, while a demonstration needs only logarithmic time. The bound covers one step of one estimator; shared structure, many steps and a different sampler can all widen what RL reaches."
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
**Why Post-Training Is Sparse · Part 3 of 3.** Starts from the cap of [Part 1](/post/gradient-is-covariance/) and the concentrated Fisher of [Part 2](/post/sparsity-score-centering/).

- **Reach follows resemblance.** One update moves the log-odds of a response against another by at most std(*A*) · √(*v*ᵀ*Fv*) per unit step, with *v* their score difference and *F* the sampler's Fisher.
- **Resemblance reaches unsampled responses.** A response never drawn but similar to rewarded samples, like *matcha* after *tea*, moves with them; one that resembles nothing the sampler produces barely moves.
- **Growth is proportional to probability.** On-policy, with one logit per response, getting from probability ε to one half takes time about 1/ε; a demonstration needs only about log(1/ε).
- **One step of one estimator.** Shared structure, a changing Fisher, and a centered estimator with a different sampler all widen what is reachable.
{{% /callout %}}

**Starting point.** For a response {{< math >}}$y${{< /math >}} to a prompt {{< math >}}$x${{< /math >}}, the model assigns probability {{< math >}}$\pi_\theta(y\mid x)${{< /math >}}; write {{< math >}}$\mathrm{score}(y)=\nabla_\theta\log\pi_\theta(y\mid x)${{< /math >}}. Responses are drawn by a sampler {{< math >}}$\smu${{< /math >}}, each with a learning signal {{< math >}}$A${{< /math >}} (a reward or advantage), and each enters the update through its score centered under {{< math >}}$\smu${{< /math >}}, written {{< math >}}$c(y)${{< /math >}}: at every position, the drawn token's logit gradient minus its average under {{< math >}}$\smu${{< /math >}}. The expected update and the sampler's Fisher are

{{< math >}}
$$
G=\mathbb{E}_{y\sim\smu}\big[A\,c(y)\big],\qquad F^{\mu}=\mathbb{E}_{\smu}\big[c\,c^\top\big].
$$
{{< /math >}}

Centering makes {{< math >}}$\mathbb{E}_{\smu}[c]=0${{< /math >}}, so {{< math >}}$G${{< /math >}} is a covariance ([Part 1](/post/gradient-is-covariance/)); on-policy, {{< math >}}$c${{< /math >}} is the plain score and {{< math >}}$G${{< /math >}} the policy gradient. [Part 2](/post/sparsity-score-centering/) shows that {{< math >}}$F^{\mu}${{< /math >}} is concentrated on a few directions; this part asks how far {{< math >}}$G${{< /math >}} can move one response.

## 1. A question that can be answered

**Ask how much one update moves a response's log-odds against another.**

{{< math >}}
$$
\frac{d}{d\eta}\log\pi_{\theta+\eta G}(y\mid x)\Big|_{\eta=0}=\mathrm{score}(y)^\top G,
$$
$$
\frac{d}{d\eta}\log\frac{\pi_{\theta+\eta G}(y\mid x)}{\pi_{\theta+\eta G}(y'\mid x)}\Big|_{\eta=0}=v^\top G,
$$
$$
v=\mathrm{score}(y)-\mathrm{score}(y').
$$
{{< /math >}}

"Can reinforcement learning (RL) learn something new?" is too vague to prove anything about. A version that can be proved: for a response {{< math >}}$y${{< /math >}} that the model rarely produces but the reward likes, how much can one update change its probability? The normalizer moves many responses at once, so the log-odds against a reference {{< math >}}$y'${{< /math >}} isolates learning {{< math >}}$y${{< /math >}} itself. Here {{< math >}}$G${{< /math >}} is the expected update defined above, {{< math >}}$\eta${{< /math >}} the size of a step along it, and {{< math >}}$\mathrm{score}(y)${{< /math >}} the model's score of the target. In the running example: after "She drank a cup of", can *matcha* rise if the sampler never says it?

## 2. A response moves as far as it resembles the sampler's outputs

**One update moves a response only as far as its score resembles the scores of what the sampler draws.**

{{< stepfig "reach" >}}

{{< math >}}
$$
\mathrm{score}(y)^\top G=\mathbb{E}_{y''\sim\smu}\big[A(y'')\,K(y,y'')\big],
$$
$$
K(y,y'')=\mathrm{score}(y)^\top c(y''),
$$
$$
\big|v^\top G\big|\le\operatorname{std}_{\smu}(A)\sqrt{v^\top F^{\mu}v},
$$
$$
v^\top F^{\mu}v=\mathbb{E}_{y''\sim\smu}\Big[\big(v^\top c(y'')\big)^2\Big].
$$
{{< /math >}}

The similarity {{< math >}}$K(y,y'')${{< /math >}} says how much raising a sampled {{< math >}}$y''${{< /math >}} raises {{< math >}}$y${{< /math >}} along the way. Matcha is never sampled, but if its score shares directions with tea's, rewarding tea lifts matcha too. A response whose score is orthogonal to the centered scores of everything the sampler draws gets {{< math >}}$\mathrm{score}(y)^\top G=0${{< /math >}}: to first order its log-probability does not move, however high its reward. With {{< math >}}$v${{< /math >}} a coordinate vector it is the per-parameter cap {{< math >}}$|G_i|\le\operatorname{std}_{\smu}(A)\sqrt{F^{\mu}_{ii}}${{< /math >}}; here it holds in any direction.

<details class="spoiler">
<summary>Proof, and the preconditioned version</summary>

Since {{< math >}}$\mathbb{E}_{\smu}[c]=0${{< /math >}}, also {{< math >}}$\mathbb{E}_{\smu}[v^\top c]=0${{< /math >}}, and the mean of {{< math >}}$A${{< /math >}} can be subtracted:

{{< math >}}
$$
v^\top G=\mathbb{E}_{\smu}\big[(A-\mathbb{E}_{\smu}A)\,v^\top c\big]
$$
$$
\le\sqrt{\operatorname{Var}_{\smu}(A)}\,\sqrt{\mathbb{E}_{\smu}\big[(v^\top c)^2\big]}=\operatorname{std}_{\smu}(A)\sqrt{v^\top F^{\mu}v}.
$$
{{< /math >}}

If the step is {{< math >}}$\Delta\theta=\eta\,DG${{< /math >}} with a diagonal preconditioner {{< math >}}$D${{< /math >}}, as in one Adam step, replace {{< math >}}$v${{< /math >}} with {{< math >}}$Dv${{< /math >}}: {{< math >}}$|v^\top DG|\le\operatorname{std}_{\smu}(A)\sqrt{v^\top DF^{\mu}D\,v}${{< /math >}}.

</details>

## 3. One logit per response: growth in proportion to probability

**On-policy, with one logit per response, a response grows at a rate proportional to its own probability.**

{{< stepfig "amplify" >}}

{{< math >}}
$$
\frac{\partial\,\mathbb{E}_\pi[r]}{\partial z_y}=\pi(y)\big(r(y)-\mathbb{E}_\pi[r]\big)=\pi(y)\,\Delta(y),
$$
$$
\frac{d\log\pi(y)}{d\eta}=\pi(y)\,\Delta(y)-\sum_{y''}\pi(y'')^2\,\Delta(y''),
$$
$$
\frac{d}{d\eta}\big[\log\pi(y)-\log\pi(y')\big]
$$
$$
=\pi(y)\,\Delta(y)-\pi(y')\,\Delta(y').
$$
{{< /math >}}

The second term comes from the normalizer and is the same for every response: it raises or lowers the whole tail without changing ratios inside it. The log-odds keep only the first term, which is small for any response the model rarely produces. The kernel bound agrees: with the on-policy Fisher in logits, {{< math >}}$F^{\mu}=\operatorname{diag}(\pi)-\pi\pi^\top${{< /math >}}, and {{< math >}}$v=e_y-e_{y'}${{< /math >}}, where {{< math >}}$e_y${{< /math >}} is the indicator of {{< math >}}$y${{< /math >}}, we get

{{< math >}}
$$
v^\top F^{\mu}v=\pi(y)+\pi(y')-\big(\pi(y)-\pi(y')\big)^2
$$
$$
\le\pi(y)+\pi(y').
$$
{{< /math >}}

Measuring time {{< math >}}$t${{< /math >}} as step size times number of steps and ignoring the normalizer, {{< math >}}$d\pi/dt\approx\Delta\,\pi^2${{< /math >}}, so {{< math >}}$1/\pi${{< /math >}} falls at rate {{< math >}}$\Delta${{< /math >}} and getting from {{< math >}}$\varepsilon${{< /math >}} to one half takes about {{< math >}}$(1/\varepsilon-2)/\Delta${{< /math >}}. This matches the theory of softmax policy gradient, whose rate depends on how much probability the optimal action keeps ([Mei et al., 2020](https://arxiv.org/abs/2005.06392)), and which can take exponential time in the worst case ([Li et al., 2021](https://arxiv.org/abs/2102.11270)).

{{< livefig "s-amplify" "Drag the sliders. Each tenfold drop in the starting probability costs policy gradient about ten times more time, and the demonstration a constant." 430 >}}

<details class="spoiler">
<summary>The two-response case, solved exactly</summary>

With two responses, rewards {{< math >}}$\Delta${{< /math >}} and 0, and log-odds {{< math >}}$\ell=\log\frac{\pi}{1-\pi}${{< /math >}} starting at {{< math >}}$\ell_0=\log\frac{\varepsilon}{1-\varepsilon}${{< /math >}}:

{{< math >}}
$$
\text{policy gradient: }\ \frac{d\ell}{dt}=2\Delta\,\pi(1-\pi),
$$
$$
t_{1/2}=\frac{e^{-\ell_0}-2\ell_0-e^{\ell_0}}{2\Delta}\approx\frac{1}{2\varepsilon\Delta},
$$
$$
\text{demonstration: }\ \frac{d\ell}{dt}=2(1-\pi),
$$
$$
t_{1/2}=\frac{1-\ell_0-e^{\ell_0}}{2}\approx\tfrac12\log\frac1\varepsilon .
$$
{{< /math >}}

From {{< math >}}$\varepsilon=0.01${{< /math >}} with {{< math >}}$\Delta=1${{< /math >}} these give 54.1 and 2.8; from {{< math >}}$\varepsilon=0.001${{< /math >}}, 506 and 4.0.

</details>

## 4. A demonstration injects probability directly

**A demonstration pushes a response at a rate that does not shrink with its probability.**

Imitating a response {{< math >}}$y^\star${{< /math >}} moves the parameters along its own score:

{{< math >}}
$$
\frac{d}{d\eta}\log\pi_{\theta+\eta\,\mathrm{score}(y^\star)}(y^\star)\Big|_{\eta=0}=\big\|\mathrm{score}(y^\star)\big\|^2,
$$
$$
\big\|e_{y^\star}-\pi\big\|^2=\big(1-\pi(y^\star)\big)^2+\sum_{y\ne y^\star}\pi(y)^2 .
$$
{{< /math >}}

The push depends only on {{< math >}}$y^\star${{< /math >}} itself and stays near one however unlikely it is. The tabular expression, {{< math >}}$1-2\pi(y^\star)+\sum_y\pi(y)^2${{< /math >}}, is the squared logit update of [Logit Dynamics](https://arxiv.org/abs/2506.12912) for a chosen action; in the output layer, the row of {{< math >}}$y^\star${{< /math >}} receives {{< math >}}$1-\pi(y^\star)${{< /math >}} times the last hidden state ([Part 2](/post/sparsity-score-centering/#3-without-centering-the-mean-shift)). Supervised fine-tuning can inject mass into a response the sampler never produces. Policy gradient can only amplify what it already reaches through similarity: {{< math >}}$1/\varepsilon${{< /math >}} against {{< math >}}$\log(1/\varepsilon)${{< /math >}} in the figure.

## 5. Limits of the bound

**The bound covers one first-order step of one estimator, and leaves room for RL to learn new behavior.**

- **Shared structure.** In a network the similarity {{< math >}}$K${{< /math >}} has off-diagonal entries, so a response never sampled moves with the rewarded samples it resembles ([section 2](#2-a-response-moves-as-far-as-it-resembles-the-samplers-outputs)): new solutions recombined from existing pieces are within reach.
- **Many steps.** The Fisher moves. In [Zhang, Mukherjee & Peng (2026)](https://arxiv.org/abs/2609.36262), the initial high-Fisher set keeps only 70–78% of its coordinates through 1,000 steps, and the mean Fisher of the initial low-Fisher set rises by 15–17%. Directions that were invisible can open; a multi-step bound would need control over that drift, and none exists yet.
- **First order.** Large steps and strong nonlinearity add higher-order terms.
- **Which estimator, and which sampler.** With full importance weighting (each sampled response weighted by {{< math >}}$\pi_\theta/\smu${{< /math >}}), the expected update is the on-policy gradient, bounded by the model's own Fisher whatever the sampler, so changing the sampler leaves the reach of a single step unchanged. With the score centered under the sampler, the update is {{< math >}}$\operatorname{Cov}_{\smu}(A,c)${{< /math >}}, bounded by {{< math >}}$F^{\mu}${{< /math >}}: a different sampler (a higher temperature, an exploration bonus, or posterior sampling as in [Exploration at Scale](/post/scalable-exploration/)) changes which directions are visible. Value-based temporal-difference learning has no score weighting, so the bound does not apply to it.
- **Sparse can be substantial.** A new capability can live in a few parameters.

## 6. What to measure

**Whether policy gradient can learn a behavior depends on score similarity as well as on probability.**

{{< math >}}
$$
v^\top F^{\mu}v=\mathbb{E}_{y''\sim\smu}\Big[\big(v^\top c(y'')\big)^2\Big],
$$
$$
v=\mathrm{score}(y)-\mathrm{score}(y').
$$
{{< /math >}}

The debate over whether RL extends a base model's abilities usually looks at probability: at large *k*, the base model's pass@*k*, the chance that one of *k* samples is correct, can be higher ([Yue et al., 2025](https://arxiv.org/abs/2504.13837)). The bound points to a different quantity, the overlap between the target behavior's score and what the sampler produces. Among equally unlikely responses, those that share structure with high-reward samples can be amplified and isolated ones cannot. Policy gradient is similarity-driven amplification. Whether it yields new capabilities depends on how new behavior connects, through shared parameters, to what the sampler already generates, and on which sampler is used.

**The series in one line.** A score centered under the sampler makes the update a covariance capped by the Fisher ([Part 1](/post/gradient-is-covariance/)); that Fisher is concentrated when the sampler favors a few tokens that react alike, so updates are sparse ([Part 2](/post/sparsity-score-centering/)); and the same Fisher bounds how far one step can move any response. Choosing the sampler is the exploration problem, which [Exploration at Scale](/post/scalable-exploration/) treats on its own terms.

## References

- J. Mei, C. Xiao, C. Szepesvári, D. Schuurmans. [On the Global Convergence Rates of Softmax Policy Gradient Methods](https://arxiv.org/abs/2005.06392). ICML 2020.
- G. Li, Y. Wei, Y. Chi, Y. Chen. [Softmax Policy Gradient Methods Can Take Exponential Time to Converge](https://arxiv.org/abs/2102.11270). COLT 2021.
- Y. Yue, Z. Chen, R. Lu, et al. [Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?](https://arxiv.org/abs/2504.13837) 2025.
- Y. Li. [Logit Dynamics in Softmax Policy Gradient Methods](https://arxiv.org/abs/2506.12912). arXiv, 2025.
- Y. Zhang, S. Mukherjee, H. Peng. [Understanding LLM Parameter Update Sparsity through the Lens of Fisher](https://arxiv.org/abs/2609.36262). arXiv, 2026.
- Y. Li, J. Xu, B. Wang, Z.-Q. Luo. [Scalable Exploration via Ensemble++](https://arxiv.org/abs/2407.13195). NeurIPS 2025.

## Citation

```bibtex
@online{li-2026-what-policy-gradient-can-reach,
  title  = {What Policy Gradient Can Reach},
  author = {Li, Yingru},
  year   = {2026},
  month  = oct,
  url    = {https://richardli.xyz/post/what-policy-gradient-can-reach/}
}
```
