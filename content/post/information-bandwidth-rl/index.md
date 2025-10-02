---
title: 'Information Bandwidth in Reinforcement Learning'
subtitle: 'Understanding Sample Efficiency Through Signal Density'
summary: 'An information-theoretic analysis explaining why policy gradient learns 1 bit per episode and why LoRA works for RL fine-tuning.'
authors:
  - admin
tags:
  - Reinforcement Learning
  - Information Theory
  - Language Models
  - Deep Learning
categories:
  - Research
  - Theory
date: '2025-10-01T00:00:00Z'
lastmod: '2025-10-01T00:00:00Z'
featured: true
draft: false
math: true
toc: true

# Featured image
image:
  caption: '"LoRA without Regret"'
  focal_point: ''
  preview_only: false

# Projects (optional)
#   Associate this post with one or more projects.
projects: []
---

## Understanding Sample Efficiency Through Signal Density

When I first read the "[LoRA Without Regret](https://thinkingmachines.ai/blog/lora/)" blog post, one claim caught my attention: policy gradient algorithms learn roughly **1 bit of information per episode**. This insight elegantly explains why LoRA—with its mere thousands of trainable parameters—works so remarkably well for RL fine-tuning of large language models.

But what does this actually mean? And if policy gradients learn so little per episode, how much do other RL algorithms learn? In this post, I'll work through an information-theoretic framework to answer these questions rigorously.

---

## The Key Insight

**TL;DR**: Policy gradient uses **sparse episode-level signals** (one scalar per episode), while actor-critic uses **dense token-level signals** (one per token). This fundamental difference in signal structure creates an information bandwidth ceiling that explains both why LoRA works and where future improvements lie.

| Algorithm | Signal Type | Bandwidth Upper Bound |
|-----------|-------------|---------------------|
| Policy Gradient | Episode return {{< math >}}$G${{< /math >}} | {{< math >}}$\leq \log_2(B)${{< /math >}} bits/episode |
| Actor-Critic | TD errors {{< math >}}$\{\delta_t\}${{< /math >}} | {{< math >}}$\leq T \log_2(B_\delta)${{< /math >}} bits/episode |

*Note: These are upper bounds on information bandwidth based on signal structure. Actual information depends on signal quality and correlation.*

With binary preferences ({{< math >}}$B = 2${{< /math >}}), policy gradient is bounded by 1 bit per episode. With sequences of {{< math >}}$T \sim 1000${{< /math >}} tokens, actor-critic's theoretical ceiling is ~1000× higher.

---

## Framework: Token-Level MDPs and Information Flow

### The Token Generation Process

When we fine-tune a language model with RL, we're working with a specific type of MDP:

**States**: Token sequences {{< math >}}$s = (x_1, \ldots, x_t)${{< /math >}}
**Actions**: Next token {{< math >}}$a = x_{t+1}${{< /math >}} from vocabulary
**Transitions**: Deterministic concatenation {{< math >}}$s' = s \circ a${{< /math >}}
**Rewards**: Determined by unknown parameter {{< math >}}$\xi${{< /math >}} (human preferences, task objectives, etc.)

The crucial property: **transitions are deterministic and known**. All uncertainty lives in the reward function.

### Bayesian RL Framework

We model uncertainty explicitly:

- **Prior**: {{< math >}}$\xi \sim p(\xi)${{< /math >}} over reward parameters
- **Induced distribution**: {{< math >}}$p(\pi^*)${{< /math >}} over optimal policies
- **Deterministic mapping**: Each {{< math >}}$\xi${{< /math >}} determines an optimal policy {{< math >}}$\pi^*_\xi${{< /math >}}

This makes both the learning signal {{< math >}}$S${{< /math >}} and optimal policy {{< math >}}$\pi^*${{< /math >}} well-defined random variables, allowing rigorous computation of mutual information {{< math >}}$I(S; \pi^*)${{< /math >}}.

### Information Bandwidth

We define the **information bandwidth** as:

{{< math >}}
$$\mathcal{B} = I(S; \pi^*) \leq H(S)$$
{{< /math >}}

This measures how much the learning signal {{< math >}}$S${{< /math >}} reduces uncertainty about the optimal policy {{< math >}}$\pi^*${{< /math >}}.

---

## Minimal Assumptions

Our rigorous results require two assumptions:

### Assumption A1 (Unique Optimum)

**Statement**: Each reward parameter {{< math >}}$\xi${{< /math >}} determines a unique optimal policy {{< math >}}$\pi^*_\xi${{< /math >}}.

**Justification**: Generically true for neural network optimization with large parameter spaces. Floating-point precision breaks exact ties; multiple exact optima are measure-zero events.

### Assumption A2 (Finite Effective Resolution)

**Statement**: Returns {{< math >}}$G${{< /math >}} have effective resolution of {{< math >}}$B${{< /math >}} distinguishable values.

**When it holds**:
- *Exactly*: Binary preferences ({{< math >}}$B = 2${{< /math >}}), Likert scales ({{< math >}}$B = 4${{< /math >}}-{{< math >}}$7${{< /math >}})
- *Approximately*: Continuous rewards with finite precision, noise, or practical distinguishability limits

**What we don't assume**: Policy determinism, low reward variance, specific prior structure, or how informative {{< math >}}$G${{< /math >}} is about {{< math >}}$\xi${{< /math >}}.

---

## Policy Gradient: The Sparse Signal Ceiling

### The Algorithm

1. Sample trajectory {{< math >}}$\tau = (s_0, a_0, \ldots, s_T)${{< /math >}}
2. Observe return {{< math >}}$G = R_\xi(s_T)${{< /math >}}
3. Update: {{< math >}}$\theta \leftarrow \theta + \alpha \nabla_\theta \log p_\theta(\tau) \cdot G${{< /math >}}

**Learning signal**: {{< math >}}$S = G${{< /math >}} (a single scalar)

### Main Result

**Theorem 1 (Policy Gradient Information Ceiling):**

*Given Assumptions A1 and A2:*

{{< math >}}
$$\mathcal{B}_{\text{PG}} = I(G; \pi^*) \leq \log_2(B) \text{ bits per episode}$$
{{< /math >}}

<details>
<summary><strong>Proof (click to expand)</strong></summary>

We prove this in two steps: first showing information flow from {{< math >}}$G${{< /math >}} to {{< math >}}$\pi^*${{< /math >}} must go through {{< math >}}$\xi${{< /math >}}, then bounding the entropy of {{< math >}}$G${{< /math >}}.

**Step 1: Information Flow Bound**

*Claim:* {{< math >}}$I(G; \pi^*) \leq I(G; \xi)${{< /math >}}

*Proof:*

Since {{< math >}}$\pi^* = \pi^*_\xi${{< /math >}} is a deterministic function of {{< math >}}$\xi${{< /math >}} (by Assumption A1), we have:
{{< math >}}
$$H(\pi^* | \xi) = 0$$
{{< /math >}}

By the chain rule for mutual information:
{{< math >}}
$$I(G; \pi^*, \xi) = I(G; \xi) + I(G; \pi^* | \xi)$$
{{< /math >}}

To evaluate {{< math >}}$I(G; \pi^* | \xi)${{< /math >}}:
{{< math >}}
$$I(G; \pi^* | \xi) = H(\pi^* | \xi) - H(\pi^* | G, \xi) = 0 - 0 = 0$$
{{< /math >}}

where both terms are zero because {{< math >}}$\pi^*${{< /math >}} is deterministic given {{< math >}}$\xi${{< /math >}}.

Therefore:
{{< math >}}
$$I(G; \pi^*, \xi) = I(G; \xi)$$
{{< /math >}}

Applying the chain rule in a different order:
{{< math >}}
$$I(G; \pi^*, \xi) = I(G; \pi^*) + I(G; \xi | \pi^*)$$
{{< /math >}}

Since mutual information is non-negative: {{< math >}}$I(G; \xi | \pi^*) \geq 0${{< /math >}}

Combining these:
{{< math >}}
$$I(G; \pi^*) = I(G; \pi^*, \xi) - I(G; \xi | \pi^*) \leq I(G; \pi^*, \xi) = I(G; \xi)$$
{{< /math >}}

**Step 2: Entropy Upper Bound**

*Claim:* {{< math >}}$I(G; \xi) \leq H(G) \leq \log_2(B)${{< /math >}}

*Proof:*

By definition of mutual information:
{{< math >}}
$$I(G; \xi) = H(G) - H(G | \xi)$$
{{< /math >}}

Since {{< math >}}$H(G | \xi) \geq 0${{< /math >}}:
{{< math >}}
$$I(G; \xi) \leq H(G)$$
{{< /math >}}

For the entropy bound, by Assumption A2, {{< math >}}$G${{< /math >}} takes at most {{< math >}}$B${{< /math >}} distinct values. For any discrete random variable with at most {{< math >}}$B${{< /math >}} outcomes:
{{< math >}}
$$H(G) = -\sum_{i=1}^{B'} p_i \log_2(p_i)$$
{{< /math >}}

where {{< math >}}$B' \leq B${{< /math >}} is the number of values with non-zero probability.

This is maximized when all outcomes are equally likely:
{{< math >}}
$$H(G) \leq -\sum_{i=1}^{B'} \frac{1}{B'} \log_2\left(\frac{1}{B'}\right) = \log_2(B') \leq \log_2(B)$$
{{< /math >}}

**Combining both steps:**
{{< math >}}
$$I(G; \pi^*) \leq I(G; \xi) \leq H(G) \leq \log_2(B)$$
{{< /math >}} ∎

</details>

**This is a hard ceiling** regardless of sequence length {{< math >}}$T${{< /math >}}, model complexity, or computational resources.

### Concrete Bounds

- Binary preferences: {{< math >}}$\leq 1${{< /math >}} bit/episode
- 4-level Likert scale: {{< math >}}$\leq 2${{< /math >}} bits/episode
- Continuous (~10 effective levels): {{< math >}}$\leq 3.3${{< /math >}} bits/episode

The return {{< math >}}$G${{< /math >}} compresses {{< math >}}$T \gg 1000${{< /math >}} tokens into one number—this compression creates the fundamental bottleneck.

---

## Actor-Critic: The Dense Signal Potential

### The Algorithm

At each timestep {{< math >}}$t${{< /math >}}:
1. Select action {{< math >}}$a_t \sim \pi_\theta(\cdot | s_t)${{< /math >}}
2. Compute TD error: {{< math >}}$\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)${{< /math >}}
3. Update actor and critic using {{< math >}}$\delta_t${{< /math >}}

**Learning signal**: {{< math >}}$S = \{\delta_t\}_{t=0}^{T-1}${{< /math >}} (one per token)

This is fundamentally different from policy gradient: instead of one scalar per episode, we get feedback at every token.

### Main Result

**Theorem 2 (Actor-Critic Information Ceiling):**

*Given Assumptions A1 and A2:*

{{< math >}}
$$\mathcal{B}_{\text{AC}} = I(\{\delta_t\}; \pi^*) \leq T \log_2(B_\delta) \text{ bits per episode}$$
{{< /math >}}

<details>
<summary><strong>Proof (click to expand)</strong></summary>

We bound the entropy of the sequence of TD errors.

**Step 1: Chain Rule Decomposition**

By the chain rule for entropy:
{{< math >}}
$$H(\{\delta_t\}_{t=0}^{T-1}) = H(\delta_0, \delta_1, \ldots, \delta_{T-1}) = \sum_{t=0}^{T-1} H(\delta_t | \delta_0, \ldots, \delta_{t-1})$$
{{< /math >}}

Using shorthand {{< math >}}$\delta_{<t} = (\delta_0, \ldots, \delta_{t-1})${{< /math >}}:
{{< math >}}
$$H(\{\delta_t\}) = \sum_{t=0}^{T-1} H(\delta_t | \delta_{<t})$$
{{< /math >}}

**Step 2: Bounding Each Conditional Entropy**

Each TD error {{< math >}}$\delta_t${{< /math >}} is a scalar. By Assumption A2 (extended to TD errors), each {{< math >}}$\delta_t${{< /math >}} has effective resolution of {{< math >}}$B_\delta${{< /math >}} distinguishable values.

Even conditioned on past TD errors, the entropy cannot exceed the entropy of a uniform distribution over {{< math >}}$B_\delta${{< /math >}} values:
{{< math >}}
$$H(\delta_t | \delta_{<t}) \leq \log_2(B_\delta)$$
{{< /math >}}

This holds because:
- If {{< math >}}$\delta_t${{< /math >}} is deterministic given {{< math >}}$\delta_{<t}${{< /math >}}: {{< math >}}$H(\delta_t | \delta_{<t}) = 0 \leq \log_2(B_\delta)${{< /math >}}
- If {{< math >}}$\delta_t${{< /math >}} has residual uncertainty: it still takes at most {{< math >}}$B_\delta${{< /math >}} values

**Step 3: Summing Over All Timesteps**

{{< math >}}
$$H(\{\delta_t\}) = \sum_{t=0}^{T-1} H(\delta_t | \delta_{<t}) \leq \sum_{t=0}^{T-1} \log_2(B_\delta) = T \log_2(B_\delta)$$
{{< /math >}}

**Step 4: Applying Information Flow Bound**

By the same argument as in Theorem 1:
{{< math >}}
$$I(\{\delta_t\}; \pi^*) \leq H(\{\delta_t\}) \leq T \log_2(B_\delta)$$
{{< /math >}} ∎

</details>

### The Gap

For {{< math >}}$T = 1000${{< /math >}} tokens and binary feedback:
- Policy gradient: {{< math >}}$\leq 1${{< /math >}} bit/episode
- Actor-critic: {{< math >}}$\leq 1000${{< /math >}} bits/episode

**Theoretical ceiling: 1000× higher.**

**Critical caveat**: Realizing this requires well-trained critics and managing correlation between successive TD errors. Traditional RL achieves 10-100× speedups; LLM-RL lacks stable critic training recipes at scale.

---

## Why LoRA Works: Matching Capacity to Information Ceiling

### The Argument

Typical setup:
- Training episodes: {{< math >}}$N \sim 1000${{< /math >}}
- LoRA rank: {{< math >}}$r = 8${{< /math >}}, dimension: {{< math >}}$d = 4096${{< /math >}}

**LoRA capacity**: {{< math >}}$2rd \approx 65{,}000${{< /math >}} parameters

**Information ceiling**: {{< math >}}$\leq 1000${{< /math >}} bits (binary preferences, 1000 episodes)

**Capacity comparison**: ~65× more parameters than information ceiling bits.

This suggests the parameter bottleneck isn't binding—we have far more capacity than the information ceiling allows us to use.

### Why Full Fine-Tuning is Overkill

A 7B parameter model provides ~7 billion degrees of freedom versus ~1,000 bits of information—a factor of ~7 million. LoRA's modest capacity naturally matches policy gradient's information ceiling.

**Empirical consistency**: LLM-RL typically needs 1,000-10,000 episodes, consistent with accumulating 1,000-10,000 bits at 1-3 bits/episode.

---

## Implications

### Current State

Policy gradient with LoRA dominates because:
- ✓ Stable at scale
- ✓ Parameter-efficient (capacity exceeds ceiling)
- ✗ Sample-inefficient ({{< math >}}$\leq \log_2(B)${{< /math >}} bits/episode)

### The Opportunity

Actor-critic methods have a theoretical ceiling 1000× higher. Traditional RL achieves 10-100× practical speedups. **The bottleneck**: stable critic training for LLMs remains unsolved.

### Research Directions

1. **Stable critic training** at LLM scale
2. **Token-level reward design** for dense signals
3. **Hybrid approaches** combining Monte Carlo and bootstrapping
4. **Low-rank value functions** matching information requirements
5. **Information-directed sampling** for efficient exploration

---

## Conclusion

This information-theoretic framework establishes:

**Policy gradient ceiling**: Compressing {{< math >}}$T \gg 1000${{< /math >}} tokens into scalars creates a {{< math >}}$\leq \log_2(B)${{< /math >}} bits/episode ceiling (typically 1-3 bits). LoRA's capacity naturally matches this ceiling.

**Actor-critic potential**: Token-level signals have a {{< math >}}$\leq T \log_2(B_\delta)${{< /math >}} bits/episode ceiling—orders of magnitude higher. Unlocking this requires solving stable critic training for LLMs.

**The path forward**: Future breakthroughs will come from raising the ceiling through effective value function learning, not from increasing adapter capacity.

The ceiling is structural, not algorithmic—that's why there's room for transformative improvement.

---

## Technical Notes

**What we prove rigorously** (given A1, A2):
- {{< math >}}$I(G; \pi^*) \leq \log_2(B)${{< /math >}} bits/episode (policy gradient)
- {{< math >}}$I(\{\delta_t\}; \pi^*) \leq T \log_2(B_\delta)${{< /math >}} bits/episode (actor-critic)

These are hard upper bounds independent of sequence length, model complexity, or optimization quality.

**What remains conjectural**:
- How close algorithms get to these ceilings in practice
- Correlation structure between successive TD errors
- Sample complexity predictions (optimization dynamics matter)

**Scope**: Stationary reward learning with known dynamics (autoregressive generation). Extensions needed for exploration, partial observability, or unknown environment dynamics.

---

## References

**Core Papers**:
- Ouyang, L., Wu, J., Jiang, X., et al. (2022). "Training language models to follow instructions with human feedback." *NeurIPS*. (InstructGPT)
- Hu, E. J., Shen, Y., Wallis, P., et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." *ICLR*.
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). "Proximal Policy Optimization Algorithms." *arXiv*.

**Information Theory**:
- Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory* (2nd ed.). Wiley-Interscience.
- Russo, D., & Van Roy, B. (2014). "Learning to Optimize via Information-Directed Sampling." *Operations Research*, 66(1), 230-252.

**Foundational RL**:
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.

**Inspiration**: ThinkingMachines.ai (2025). "[LoRA Without Regret](https://thinkingmachines.ai/blog/lora/)."

---

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
