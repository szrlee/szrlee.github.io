---
title: 'Information Bandwidth in Reinforcement Learning'
subtitle: 'How much one episode can teach: the reward sets the ceiling, and the form of the gradient can only lower it'
summary: 'Given its history, a learner gains nothing from a response it sampled itself, so every bit enters through the reward. A pass/fail episode teaches at most one bit, and less when the outcome is predictable. With a reward at every token, one scalar advantage keeps at most log₂ B bits; per-token advantages keep the whole reward sequence.'
authors:
  - admin
tags:
  - Reinforcement Learning
  - Information Theory
  - Policy Gradient
  - Language Models
  - Deep Learning
categories:
  - Research
  - Theory
date: '2025-10-01T00:00:00Z'
lastmod: '2026-10-15T00:00:00Z'
featured: true
draft: false
math: true
---

{{% callout note %}}
- A learner **gains nothing from a response it sampled itself**: given its history, the response says nothing about the reward. Every bit enters **through the reward**.
- So an episode teaches at most the **entropy of its reward**. A pass/fail episode teaches **at most 1 bit**, and only *h*(*p*) bits when the learner already expects to pass with probability *p*.
- With a reward at every token, **one scalar advantage** keeps at most log₂ *B* bits, *B* the number of values it takes. **Per-token advantages** keep the whole reward sequence.
- With one outcome reward, REINFORCE, GRPO and PPO with a critic all learn **at most 1 bit per episode**, and *K* episodes put at most *K* bits into the weights. A rank-1 adapter has room to spare.
{{% /callout %}}

The figures step through like slides. Use **→** or **▶**, and step back with **←**.

| Symbol | Meaning |
|---|---|
| {{< math >}}$\xi,\ \pi^\star${{< /math >}} | the unknown reward function, with a prior, and the optimal policy it defines |
| {{< math >}}$\mathcal H${{< /math >}} | the history: every earlier response and its reward |
| {{< math >}}$\tau,\ \mathbf r${{< /math >}} | this episode's response and its rewards {{< math >}}$\mathbf r=(r_1,\dots,r_T)${{< /math >}} |
| {{< math >}}$u${{< /math >}} | the update computed from the episode: one gradient, or a whole PPO step |
| {{< math >}}$\psi_t${{< /math >}} | the score of token {{< math >}}$t${{< /math >}}: {{< math >}}$\nabla_\theta\log\pi_\theta(a_t\mid s_t)${{< /math >}} |

## 1. One bit per episode

**For an outcome reward the one-bit count is exact, and the limit comes from the reward, not from policy gradient.**

[LoRA Without Regret](https://thinkingmachines.ai/blog/lora/) finds that LoRA matches full fine-tuning for policy-gradient RL even at rank 1, and explains it by counting bits. A policy-gradient update depends on the reward only through a scalar advantage, so an episode with one final reward teaches about one bit. Their MATH run, 10,000 problems with 32 samples each, then has at most 320,000 bits to absorb, and a rank-1 adapter on Llama-3.1-8B has almost ten times as many parameters.

This post makes the count exact. It rests on one fact: the learner drew the response itself. From there the chain rule shows what sets the count (the reward), what can lower it (an update that collapses the reward to one number), and what cannot raise it (baselines and critics).

## 2. Every bit enters through the reward

**Given the history, a response the learner sampled says nothing about the reward, so no update can learn more than the reward's entropy.**

For a language model the transitions are known: the state {{< math >}}$s_t${{< /math >}} is the prompt and the tokens so far, the action {{< math >}}$a_t${{< /math >}} is the next token, and taking it appends it. Only the reward is unknown. Put a prior on its parameter {{< math >}}$\xi${{< /math >}}, and let {{< math >}}$\pi^\star${{< /math >}} be the optimal policy for {{< math >}}$R_\xi${{< /math >}}, with ties broken by a fixed rule. The prior is a modeling device, not a claim that algorithms are Bayesian: it makes "what the update tells us about {{< math >}}$\pi^\star${{< /math >}}" a mutual information, {{< math >}}$I(u;\pi^\star\mid\mathcal H)${{< /math >}}. The **information bandwidth** of an algorithm is the most one episode can teach it, {{< math >}}$\sup_{\mathcal H} I(u;\pi^\star\mid\mathcal H)${{< /math >}}.

**Lemma.** Let the update {{< math >}}$u${{< /math >}} be any function of the response {{< math >}}$\tau${{< /math >}}, its rewards {{< math >}}$\mathbf r${{< /math >}}, the history {{< math >}}$\mathcal H${{< /math >}}, and randomness independent of {{< math >}}$\xi${{< /math >}}. Then

{{< math >}}
$$
I(u;\pi^\star\mid\mathcal H)\ \le\ I(\tau,\mathbf r;\pi^\star\mid\mathcal H)
$$
$$
=\ \underbrace{I(\tau;\pi^\star\mid\mathcal H)}_{=\,0}+I(\mathbf r;\pi^\star\mid\tau,\mathcal H)\ \le\ H(\mathbf r\mid\tau,\mathcal H).
$$
{{< /math >}}

The steps are data processing, the chain rule, and {{< math >}}$I\le H${{< /math >}}. The middle term vanishes because the sampler that drew {{< math >}}$\tau${{< /math >}}, on-policy or not, is a function of {{< math >}}$\mathcal H${{< /math >}}, and its sampling noise does not depend on {{< math >}}$\xi${{< /math >}}: given the history, {{< math >}}$\tau${{< /math >}} is independent of {{< math >}}$\xi${{< /math >}}, hence of {{< math >}}$\pi^\star${{< /math >}}.

Any algorithm that trains on its own samples computes its update this way: REINFORCE, GRPO, and PPO with its clipping, critic and several epochs per batch. Nothing in the lemma is specific to policy gradient; it holds for any learner trained on its own responses and their rewards, as long as the environment adds nothing to {{< math >}}$\tau${{< /math >}}. Algorithms differ only in how much of {{< math >}}$H(\mathbf r\mid\tau,\mathcal H)${{< /math >}} their update keeps.

## 3. A pass/fail episode teaches at most one bit, often far less

**With one outcome reward, an episode teaches at most *h*(*p*) ≤ 1 bit, where *p* is the pass rate the learner predicts.**

For a reward {{< math >}}$r\in\{0,1\}${{< /math >}} at the end, conditioning can only lower entropy, so

{{< math >}}
$$
I(u;\pi^\star\mid\mathcal H)\ \le\ H(r\mid\tau,\mathcal H)\ \le\ H(r\mid\mathcal H)=h(\bar p),
$$
$$
h(p)=-p\log_2 p-(1-p)\log_2(1-p)\ \le\ 1,
$$
{{< /math >}}

where {{< math >}}$\bar p=P(r=1\mid\mathcal H)${{< /math >}} is the pass rate predicted from everything seen so far.

{{< stepfig "passrate" >}}

Applying the same bound to each reward in turn, a group of {{< math >}}$n${{< /math >}} samples of one prompt teaches at most {{< math >}}$n\,h(\bar p)${{< /math >}} bits. Prompts the model almost always or almost never solves teach almost nothing. Their groups are mostly all passes or all fails, which a group-mean baseline turns into a zero update and which the dynamic sampling of [Yu et al. (2025)](https://arxiv.org/abs/2503.14476) filters out.

Over a run, the weights {{< math >}}$\theta_K${{< /math >}} after {{< math >}}$K${{< /math >}} episodes are a function of the history and of randomness independent of {{< math >}}$\xi${{< /math >}}, so the chain rule adds up the episodes:

{{< math >}}
$$
I(\theta_K;\pi^\star)\ \le\ I(\mathcal H_K;\pi^\star)
$$
$$
=\ \sum_{k=1}^{K} I(\tau_k,r_k;\pi^\star\mid\mathcal H_{k-1})\ \le\ \sum_{k=1}^{K}\mathbb E\,h(\bar p_k)\ \le\ K.
$$
{{< /math >}}

{{< math >}}$K${{< /math >}} pass/fail episodes put at most {{< math >}}$K${{< /math >}} bits about {{< math >}}$\pi^\star${{< /math >}} into the weights, whatever the model size or compute. The 320,000 bits of the MATH run in section 1 are a bound, not an estimate.

## 4. With dense rewards, the advantage decides what gets through

**One scalar advantage keeps at most log₂ *B* bits; per-token advantages keep the whole reward sequence.**

With a reward {{< math >}}$r_t${{< /math >}} at every token there is more to learn per episode, and the update can now lose some of it. Given {{< math >}}$\tau${{< /math >}} and {{< math >}}$\mathcal H${{< /math >}} the token scores {{< math >}}$\psi_t${{< /math >}} are fixed, and the advantages only weight them:

{{< math >}}
$$
u_{\text{scalar}}=\Big(\sum_t\psi_t\Big)\Big(\sum_t r_t-b\Big),
$$
$$
u_{\text{per-token}}=\sum_t\psi_t\,A_t .
$$
{{< /math >}}

The scalar form depends on {{< math >}}$\mathbf r${{< /math >}} only through {{< math >}}$\mathrm{Adv}=\sum_t r_t-b${{< /math >}}, one number with at most {{< math >}}$B${{< /math >}} values, so the lemma tightens to {{< math >}}$I(u;\pi^\star\mid\mathcal H)\le H(\mathrm{Adv}\mid\tau,\mathcal H)\le\log_2 B${{< /math >}}. Per-token advantages, whether returns-to-go, TD errors or GAE, are an invertible function of {{< math >}}$\mathbf r${{< /math >}} given {{< math >}}$(\tau,\mathcal H)${{< /math >}}. When the token scores are linearly independent, the update keeps everything the rewards say about {{< math >}}$\pi^\star${{< /math >}}.

{{< stepfig "outcomes" >}}

A sum forgets order: (+1, −1), (−1, +1) and (0, 0) all add up to zero. What that costs depends on the reward. For {{< math >}}$T=1000${{< /math >}} tokens and no discount:

| | one pass/fail reward at the end | a reward in {−1, 0, +1} at every token |
|---|---|---|
| one scalar advantage | ≤ 1 bit | ≤ log₂ 2001 ≈ 11 bits |
| per-token advantages | ≤ 1 bit | ≤ 1000 · log₂ 3 ≈ 1585 bits |

The scalar ceiling grows like {{< math >}}$\log T${{< /math >}}, the per-token ceiling like {{< math >}}$T${{< /math >}}. With one outcome reward there is nothing to forget, and the two forms are equal. With {{< math >}}$B_r${{< /math >}} values per reward, the dense ceiling is reached only if every reward is uniform and unpredictable from the earlier ones,

{{< math >}}
$$
H(\mathbf r\mid\tau,\mathcal H)=\sum_{t=1}^{T}H(r_t\mid r_{\lt t},\tau,\mathcal H)\ \le\ T\log_2 B_r ,
$$
{{< /math >}}

so a dense reward whose steps are predictable from one another teaches little more than a single reward.

<details class="spoiler">
<summary>Proof: per-token advantages are invertible, and the scalar needs finitely many values</summary>

Given {{< math >}}$(\tau,\mathcal H)${{< /math >}}, each form of per-token advantage is an invertible map of the rewards:

- returns-to-go {{< math >}}$G_t=\sum_{k\ge t}\gamma^{k-t}r_k${{< /math >}} give {{< math >}}$r_t=G_t-\gamma G_{t+1}${{< /math >}} with {{< math >}}$G_{T+1}=0${{< /math >}}, and subtracting a known baseline changes nothing;
- TD errors {{< math >}}$\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)${{< /math >}} give {{< math >}}$r_t${{< /math >}} back once the critic's values are subtracted, and the critic is a function of {{< math >}}$\mathcal H${{< /math >}};
- GAE, {{< math >}}$A_t=\sum_{l\ge 0}(\gamma\lambda)^l\delta_{t+l}${{< /math >}}, is a triangular mix of the TD errors with unit diagonal, so it is invertible too.

If the {{< math >}}$\psi_t${{< /math >}} are linearly independent, {{< math >}}$u=\sum_t\psi_tA_t${{< /math >}} determines the advantages, hence the rewards, and {{< math >}}$I(u;\pi^\star\mid\mathcal H)=I(\mathbf r;\pi^\star\mid\tau,\mathcal H)${{< /math >}}.

For the scalar form, {{< math >}}$B${{< /math >}} counts the values {{< math >}}$\sum_t\gamma^{t-1}r_t${{< /math >}} can take. With {{< math >}}$\gamma=1${{< /math >}} and rewards in {{< math >}}$\{-1,0,+1\}${{< /math >}} the sum is an integer in {{< math >}}$[-T,T]${{< /math >}}, so {{< math >}}$B\le 2T+1${{< /math >}}. With a discount, distinct sequences generally give distinct sums; at {{< math >}}$\gamma=1/3${{< /math >}} the sum is a balanced-ternary expansion and all {{< math >}}$3^T${{< /math >}} sequences survive. A real number could then carry everything, but only at about {{< math >}}$T\log_2 3${{< /math >}} bits of precision. A float32 takes at most {{< math >}}$2^{32}${{< /math >}} values, so the scalar carries at most 32 bits whatever {{< math >}}$T${{< /math >}} is.

</details>

## 5. What changes the bandwidth, and what does not

**Baselines and critics move bits around without adding any. More bits come only from the reward, the sampler, or someone else's data.**

- **Baselines, critics and per-token credit.** A baseline computed from the history, or from the other rewards in a group, only reshuffles bits the group already has. A critic is trained on earlier episodes, so what it knows is already in {{< math >}}$\mathcal H${{< /math >}}. Neither adds a bit. What they change is the noise that comes with the bits, and so how many episodes an optimizer needs to use them; the [Optimal Token Baseline](/optimal-token-baseline) is the variance-optimal choice.
- **A denser reward.** A process reward that scores each of {{< math >}}$m${{< /math >}} steps raises the ceiling to as much as {{< math >}}$m\log_2 B_r${{< /math >}} bits, but only with per-token advantages, and only as far as its scores are unpredictable from one another.
- **The sampler.** {{< math >}}$H(\mathbf r\mid\tau,\mathcal H)${{< /math >}} depends on which response is drawn, so a sampler that goes where the outcome is uncertain gets more bits per episode. Information-directed sampling ([Russo & Van Roy, 2018](https://arxiv.org/abs/1403.5556)) picks samples by trading this information against regret. Thompson sampling sends the learner where plausible models disagree, which is where outcomes are uncertain, and [Exploration at Scale](/post/scalable-exploration/) makes it affordable for deep networks.
- **Someone else's responses.** The lemma rests on one fact: the learner drew {{< math >}}$\tau${{< /math >}} itself. A demonstration is drawn by someone else, so the response itself carries information, up to {{< math >}}$T\log_2\lvert\mathcal V\rvert${{< /math >}} bits for a vocabulary {{< math >}}$\mathcal V${{< /math >}}. This is why SFT can teach far more per episode, and why it lifts rare responses that policy gradient raises only slowly ([What Policy Gradient Can Reach](/post/what-policy-gradient-can-reach/)).

Few bits explain why an RL update carries little. Which coordinates it moves is a separate question, the subject of [Why Post-Training Is Sparse](/post/gradient-is-covariance/).

## 6. Scope

- The bound is on **capacity**, not rate: what an episode can teach, not what an optimizer extracts from it.
- For a continuous reward, stop one step early: the bound is {{< math >}}$I(\mathbf r;\pi^\star\mid\tau,\mathcal H)${{< /math >}}, which stays finite when the reward is noisy.
- In a multi-turn agent, tool outputs and user replies inside {{< math >}}$\tau${{< /math >}} come from the environment, not from the learner. They can carry information too, and the bound gains a term for what they reveal.
- Nothing here is measured in a real run.

## References

- J. Schulman and Thinking Machines Lab. [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/). Thinking Machines Lab: Connectionism, 2025.
- E. J. Hu, Y. Shen, P. Wallis, et al. [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685). ICLR 2022.
- R. S. Sutton, D. McAllester, S. Singh, Y. Mansour. Policy Gradient Methods for Reinforcement Learning with Function Approximation. NeurIPS 1999.
- V. R. Konda, J. N. Tsitsiklis. Actor-Critic Algorithms. NeurIPS 1999.
- J. Schulman, P. Moritz, S. Levine, M. Jordan, P. Abbeel. [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438). ICLR 2016.
- J. Schulman, F. Wolski, P. Dhariwal, A. Radford, O. Klimov. [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347). arXiv, 2017.
- Z. Shao, P. Wang, Q. Zhu, et al. [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300). arXiv, 2024.
- Q. Yu, Z. Zhang, R. Zhu, et al. [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476). arXiv, 2025.
- D. Russo, B. Van Roy. [Learning to Optimize via Information-Directed Sampling](https://arxiv.org/abs/1403.5556). Operations Research, 2018.
- T. M. Cover, J. A. Thomas. Elements of Information Theory, 2nd ed. Wiley, 2006.

## Acknowledgements

I thank [@mgostIH](https://x.com/mgostIH) for pointing out that REINFORCE with per-token returns {{< math >}}$G_t-b${{< /math >}} keeps the reward sequence, since each {{< math >}}$r_t${{< /math >}} can be read off two consecutive returns. The first version of this post analyzed only the scalar advantage and TD errors.

## Citation

```bibtex
@article{li2025information,
  title   = {Information Bandwidth in Reinforcement Learning},
  author  = {Li, Yingru},
  journal = {Richard Li's Blog},
  year    = {2025},
  url     = {https://richardli.xyz/post/information-bandwidth-rl/}
}
```

*First posted October 2025; revised November 2025 and October 2026.*
