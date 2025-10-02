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

## TL;DR: The Main Results

**The fundamental bottleneck**: Policy gradient compresses 1000+ tokens into one scalar reward, creating a hard ceiling of {{< math >}}$\leq \log_2(B)${{< /math >}} bits per episode. For binary preferences, this is {{< math >}}$\leq 1${{< /math >}} bit/episode. This explains both why LoRA works (excess capacity matches the ceiling) and why training needs thousands of episodes.

**The theoretical upper bound**: Actor-critic methods get one signal per token, with a mathematical upper bound of {{< math >}}$\leq T \log_2(B_\delta)${{< /math >}} bits/episode—potentially **1000-10000× higher** under independence assumptions. For {{< math >}}$T=1000${{< /math >}} tokens and 8-bit TD errors, this bound is {{< math >}}$\leq 8000${{< /math >}} bits/episode.

**The fundamental reality**: Correlation between TD errors (inherent to bootstrap methods) reduces this bound substantially. Empirical speedups are 10-100× over policy gradient. Much of this gap may reflect fundamental barriers in TD learning, not just algorithmic limitations.

**The actionable insight**: Future breakthroughs require solving stable value learning at scale—though fundamental correlation in bootstrap methods may limit achievable gains. LoRA already provides 300-500× excess capacity for current methods.

| Algorithm | Signal Density | Information Upper Bound | Empirical Reality |
|-----------|---------------|---------------------|-------------------|
| Policy Gradient | 1 scalar/episode | {{< math >}}$\leq 1${{< /math >}} bit/episode | ~1 bit/episode |
| Actor-Critic | {{< math >}}$T${{< /math >}} scalars/episode | {{< math >}}$\leq 8000${{< /math >}} bits/episode (assumes independence) | ~10-100 bits/episode |

---

## Part 1: The Mathematical Framework

### Setup: Language Model Fine-Tuning as an MDP

When fine-tuning an LLM with RL, we work with a specific type of MDP:

- **States** {{< math >}}$s${{< /math >}}: Token sequences {{< math >}}$(x_1, \ldots, x_t)${{< /math >}}
- **Actions** {{< math >}}$a${{< /math >}}: Next token {{< math >}}$x_{t+1}${{< /math >}} from vocabulary
- **Transitions**: Deterministic (append token: {{< math >}}$s' = s \circ a${{< /math >}})
- **Rewards** {{< math >}}$R_\xi${{< /math >}}: Determined by unknown parameter {{< math >}}$\xi${{< /math >}} (preferences, objectives)

**Key property**: Transitions are known and deterministic. All uncertainty is in the reward function {{< math >}}$\xi${{< /math >}}.

### Information-Theoretic Lens

To enable rigorous analysis, we use a Bayesian framework as a **mathematical modeling tool**:

1. Put a prior {{< math >}}$p(\xi)${{< /math >}} over reward parameters
2. This induces a distribution {{< math >}}$p(\pi^*)${{< /math >}} over optimal policies
3. Each {{< math >}}$\xi${{< /math >}} determines a unique optimal policy {{< math >}}$\pi^*_\xi${{< /math >}}

This doesn't claim algorithms maintain explicit posteriors—it's an analytical device that makes the learning signal {{< math >}}$S${{< /math >}} and optimal policy {{< math >}}$\pi^*${{< /math >}} well-defined random variables, enabling computation of mutual information {{< math >}}$I(S; \pi^*)${{< /math >}}.

**Definition (Information Bandwidth)**:

{{< math >}}
$$\mathcal{B} = I(S; \pi^*)$$
{{< /math >}}

This measures how many bits of uncertainty about the optimal policy {{< math >}}$\pi^*${{< /math >}} are resolved per episode by the learning signal {{< math >}}$S${{< /math >}}.

### Two Minimal Assumptions

**Assumption A1 (Unique Optimum)**: Each {{< math >}}$\xi${{< /math >}} determines a unique optimal policy {{< math >}}$\pi^*_\xi${{< /math >}}.

*Justification*: Generic for neural networks with many parameters. Floating-point precision breaks ties; exact degeneracy is measure-zero.

**Assumption A2 (Finite Resolution)**: The learning signal has finite effective resolution—it can take at most {{< math >}}$B${{< /math >}} distinguishable values.

*Justification*: Holds exactly for binary preferences ({{< math >}}$B=2${{< /math >}}) or Likert scales ({{< math >}}$B=4${{< /math >}}-{{< math >}}$7${{< /math >}}). Approximately true for continuous signals with noise, finite precision, or practical distinguishability limits.

---

## Part 2: Policy Gradient's 1-Bit Ceiling

### The Algorithm

Policy gradient (REINFORCE, PPO) works as follows:

1. Sample trajectory {{< math >}}$\tau = (s_0, a_0, \ldots, s_T)${{< /math >}}
2. Observe scalar return {{< math >}}$G = R_\xi(s_T)${{< /math >}}
3. Update: {{< math >}}$\theta \leftarrow \theta + \alpha \nabla_\theta \log p_\theta(\tau) \cdot G${{< /math >}}

**Learning signal**: {{< math >}}$S = G${{< /math >}} (one scalar per episode)

### The Information Ceiling

**Theorem 1 (Policy Gradient Information Ceiling):**

*Under assumptions A1 and A2, policy gradient's information bandwidth satisfies:*

{{< math >}}
$$I(G; \pi^*) \leq \log_2(B) \text{ bits per episode}$$
{{< /math >}}

**Intuition**: Information about {{< math >}}$\pi^*${{< /math >}} must flow through {{< math >}}$\xi${{< /math >}} (by data processing inequality), and the scalar {{< math >}}$G${{< /math >}} has entropy bounded by its resolution {{< math >}}$\log_2(B)${{< /math >}}.

<details>
<summary><strong>Detailed Proof (click to expand)</strong></summary>

We prove this in two steps: first showing information flow must go through {{< math >}}$\xi${{< /math >}}, then bounding the entropy.

**Step 1: Information Flow via Data Processing Inequality**

Since {{< math >}}$\pi^* = \pi^*_\xi${{< /math >}} is a deterministic function of {{< math >}}$\xi${{< /math >}} (by Assumption A1), we have the Markov chain:

{{< math >}}
$$G \to \xi \to \pi^*$$
{{< /math >}}

Reasoning: {{< math >}}$G${{< /math >}} depends on the trajectory and reward function {{< math >}}$\xi${{< /math >}}. Given {{< math >}}$\xi${{< /math >}}, the optimal policy {{< math >}}$\pi^*${{< /math >}} is fully determined, so {{< math >}}$\pi^*${{< /math >}} is conditionally independent of {{< math >}}$G${{< /math >}} given {{< math >}}$\xi${{< /math >}}.

By the **data processing inequality**, post-processing cannot increase information:

{{< math >}}
$$I(G; \pi^*) \leq I(G; \xi)$$
{{< /math >}}

**Step 2: Entropy Upper Bound**

By definition of mutual information:
{{< math >}}
$$I(G; \xi) = H(G) - H(G|\xi)$$
{{< /math >}}

Since conditional entropy is non-negative {{< math >}}$H(G|\xi) \geq 0${{< /math >}}:
{{< math >}}
$$I(G; \xi) \leq H(G)$$
{{< /math >}}

By Assumption A2, {{< math >}}$G${{< /math >}} takes at most {{< math >}}$B${{< /math >}} distinct values. For any discrete random variable {{< math >}}$X${{< /math >}} with support size {{< math >}}$|X| \leq B${{< /math >}}:

{{< math >}}
$$H(X) = -\sum_x p(x) \log_2 p(x) \leq \log_2(|X|) \leq \log_2(B)$$
{{< /math >}}

Equality holds when {{< math >}}$X${{< /math >}} is uniform over its support.

**Combining both steps:**
{{< math >}}
$$I(G; \pi^*) \leq I(G; \xi) \leq H(G) \leq \log_2(B)$$
{{< /math >}} ∎

</details>

**This is a hard ceiling** regardless of sequence length {{< math >}}$T${{< /math >}}, model complexity, or computational resources.

### Concrete Examples

- **Binary preferences** ({{< math >}}$B=2${{< /math >}}): {{< math >}}$\leq 1${{< /math >}} bit/episode
- **Likert scale** ({{< math >}}$B=5${{< /math >}}): {{< math >}}$\leq 2.3${{< /math >}} bits/episode
- **8-bit resolution** ({{< math >}}$B=256${{< /math >}}): {{< math >}}$\leq 8${{< /math >}} bits/episode

### Why This Matters

**The compression bottleneck**: A typical LLM generation has {{< math >}}$T \sim 1000${{< /math >}} tokens, each chosen from hundreds of possibilities. Policy gradient compresses all this rich structure—which words worked well, where the response went wrong, which reasoning steps succeeded—into **one number**.

This structural compression is why:
- **Training needs thousands of episodes**: With 1 bit/episode and binary feedback, 1000 episodes gives {{< math >}}$\leq 1000${{< /math >}} bits total
- **LoRA works well**: As we'll see, LoRA provides 300-500× more capacity than this ceiling
- **Adding parameters doesn't help**: The bottleneck is signal sparsity, not model capacity

---

## Part 3: Actor-Critic's Dense Signal Upper Bound

### The Algorithm

Actor-critic methods (A3C, PPO with value function) work differently:

**At each timestep** {{< math >}}$t${{< /math >}}:
1. Observe state {{< math >}}$s_t${{< /math >}}, take action {{< math >}}$a_t${{< /math >}}
2. Compute TD error: {{< math >}}$\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)${{< /math >}}
3. Update both actor and critic using {{< math >}}$\delta_t${{< /math >}}

**Learning signal**: {{< math >}}$S = \{\delta_t\}_{t=0}^{T-1}${{< /math >}} (one signal per token)

Instead of waiting until the end for one scalar, we get feedback at **every step**.

### Extended Assumption

**Assumption A2' (Finite TD Resolution)**: Each TD error {{< math >}}$\delta_t${{< /math >}} has effective resolution {{< math >}}$B_\delta${{< /math >}} distinguishable values.

**Justification**:
- Neural networks use finite precision (float32/16, or quantized)
- SGD with finite samples creates effective discretization
- Empirically, 8-bit resolution ({{< math >}}$B_\delta \approx 256${{< /math >}}) captures practical precision

**Important caveat**: This is stronger than A2 (applies to derived quantities, not just observations). The resulting bound is an upper limit on *potential* information.

### The Information Ceiling

**Theorem 2 (Actor-Critic Information Upper Bound):**

*Under assumptions A1 and A2', actor-critic's information bandwidth satisfies:*

{{< math >}}
$$I(\{\delta_t\}; \pi^*) \leq T \log_2(B_\delta) \text{ bits per episode}$$
{{< /math >}}

**For {{< math >}}$T=1000${{< /math >}} and {{< math >}}$B_\delta=256${{< /math >}}**: This mathematical upper bound is {{< math >}}$8000${{< /math >}} bits/episode—**8000× higher** than policy gradient's 1 bit.

**Intuition**: With {{< math >}}$T${{< /math >}} independent signals each carrying {{< math >}}$\log_2(B_\delta)${{< /math >}} bits, we get {{< math >}}$T \log_2(B_\delta)${{< /math >}} total bits. However, this assumes independence—an assumption violated by the bootstrap structure of TD learning.

<details>
<summary><strong>Detailed Proof (click to expand)</strong></summary>

We bound the entropy of the TD error sequence.

**Step 1: Chain Rule Decomposition**

By the chain rule for entropy:
{{< math >}}
$$H(\delta_0, \delta_1, \ldots, \delta_{T-1}) = \sum_{t=0}^{T-1} H(\delta_t | \delta_0, \ldots, \delta_{t-1})$$
{{< /math >}}

Using {{< math >}}$\delta_{<t} = (\delta_0, \ldots, \delta_{t-1})${{< /math >}}:
{{< math >}}
$$H(\{\delta_t\}) = \sum_{t=0}^{T-1} H(\delta_t | \delta_{<t})$$
{{< /math >}}

**Step 2: Bounding Conditional Entropy**

By Assumption A2', each {{< math >}}$\delta_t${{< /math >}} takes at most {{< math >}}$B_\delta${{< /math >}} values. For any random variable {{< math >}}$X${{< /math >}} with {{< math >}}$|X| \leq B_\delta${{< /math >}}:

{{< math >}}
$$H(X | Y) \leq \log_2(B_\delta)$$
{{< /math >}}

for any conditioning variable {{< math >}}$Y${{< /math >}}. This holds because:
- Conditional entropy is maximized when {{< math >}}$X${{< /math >}} is uniform over its support
- Even with conditioning, {{< math >}}$X${{< /math >}} still has at most {{< math >}}$B_\delta${{< /math >}} values

Therefore:
{{< math >}}
$$H(\delta_t | \delta_{<t}) \leq \log_2(B_\delta)$$
{{< /math >}}

**Critical observation**: This bound is **tight** only when {{< math >}}$\delta_t${{< /math >}} is nearly independent of {{< math >}}$\delta_{<t}${{< /math >}}:
{{< math >}}
$$H(\delta_t | \delta_{<t}) \approx H(\delta_t)$$
{{< /math >}}

If perfectly correlated: {{< math >}}$H(\delta_t | \delta_{<t}) = 0${{< /math >}}. Reality falls between these extremes.

**Step 3: Summing Over Time**

{{< math >}}
$$H(\{\delta_t\}) = \sum_{t=0}^{T-1} H(\delta_t | \delta_{<t}) \leq T \log_2(B_\delta)$$
{{< /math >}}

**Step 4: Information Flow Bound**

By the data processing inequality (TD errors {{< math >}}$\to \xi \to \pi^*${{< /math >}}):
{{< math >}}
$$I(\{\delta_t\}; \pi^*) \leq I(\{\delta_t\}; \xi) \leq H(\{\delta_t\}) \leq T \log_2(B_\delta)$$
{{< /math >}} ∎

</details>

**⚠️ Critical Caveat**: This bound assumes **independent TD errors**—an assumption fundamentally violated by bootstrap methods. Successive TD errors are structurally correlated: both {{< math >}}$\delta_t${{< /math >}} and {{< math >}}$\delta_{t+1}${{< /math >}} depend on {{< math >}}$V(s_{t+1})${{< /math >}}, sharing value function biases. This correlation is not merely an implementation issue but inherent to TD learning, making the {{< math >}}$T \log_2(B_\delta)${{< /math >}} bound loose. The actual achievable bound may be closer to what we observe empirically (10-100× speedup), with much of the gap representing fundamental rather than algorithmic barriers.

### Theory vs Practice: Understanding the Gap

For {{< math >}}$T=1000${{< /math >}}, {{< math >}}$B_\delta=256${{< /math >}}:

- **Policy gradient** (binary): {{< math >}}$\leq 1${{< /math >}} bit/episode
- **Actor-critic** (independence bound): {{< math >}}$\leq 8000${{< /math >}} bits/episode
- **Actor-critic** (empirical): ~10-100 bits/episode [1,2,3]

The gap between the mathematical bound (8000×) and empirical speedups (10-100×) reflects the looseness of the independence assumption. Since correlation is inherent to bootstrap methods, the empirical 10-100× may be closer to the **fundamental achievable bound** for TD learning, with only incremental improvements possible through better critic training.

---

## Part 4: Why LoRA Works

### The Capacity Argument

Consider typical RLHF setup:
- Episodes: {{< math >}}$N = 1000${{< /math >}}
- LoRA: rank {{< math >}}$r=8${{< /math >}}, dimension {{< math >}}$d=4096${{< /math >}}
- Binary preferences: {{< math >}}$B=2${{< /math >}}

**Information available**:
{{< math >}}
$$N \times \log_2(B) = 1000 \times 1 = 1000 \text{ bits}$$
{{< /math >}}

**LoRA capacity**:
- Parameters: {{< math >}}$2rd = 65{,}000${{< /math >}}
- Effective bits per parameter: 5-8 (between 32 and 256 distinguishable values after training)
- Total capacity: {{< math >}}$65{,}000 \times 5${{< /math >}} to {{< math >}}$65{,}000 \times 8${{< /math >}} = **325,000-520,000 bits**

**The ratio**: LoRA provides **300-500× more capacity** than the information ceiling.

### The Key Insight

**LoRA works because the parameter bottleneck isn't binding.** With policy gradient's sparse signals, you have far more capacity than information to store. The bottleneck is **signal density** (1 bit/episode), not model capacity.

**Why full fine-tuning is overkill**: A 7B model has ~7 billion parameters versus ~1000 bits of information—a factor of **7 million** excess capacity. LoRA's modest parameter count naturally matches policy gradient's information ceiling.

**Empirical consistency**: LLM-RL needs 1,000-10,000 episodes to converge, consistent with accumulating 1,000-10,000 bits at 1-3 bits/episode (depending on reward granularity).

### Implications for Actor-Critic

If actor-critic could achieve substantially better correlation management (e.g., 100 bits/episode):
- **With 1000 episodes**: 100,000 bits of information
- **LoRA capacity**: Still 3-5× excess capacity
- **Conclusion**: LoRA remains sufficient even for efficient actor-critic

Only with dramatic improvements approaching the theoretical ceiling would LoRA capacity become limiting—but this may be unachievable due to fundamental correlation in bootstrap methods.

---

## Part 5: Implications and Future Directions

### Current State of LLM Fine-Tuning

Policy gradient + LoRA dominates because:

- ✅ **Stable**: Single optimization, no critic training
- ✅ **Parameter-efficient**: Capacity exceeds ceiling by 300×
- ❌ **Sample-inefficient**: {{< math >}}$\leq 1${{< /math >}} bit/episode with binary preferences

### The Path Forward

The gap between the independence bound (8000×) and empirical speedups (10-100×) reveals important insights:

**What's uncertain**:
- How much of the gap is fundamental (inherent correlation) vs algorithmic (poor critics)?
- Can methods reduce correlation beyond current 10-100× speedups?
- Are there alternative formulations that avoid bootstrap correlation?

**Research directions**:
1. **Stable critic training** at LLM scale (low-rank value architectures, ensemble methods)
2. **Decorrelation techniques** (eligibility traces, multi-step returns, though these may have fundamental limits)
3. **Dense reward engineering** (process rewards, per-token feedback)
4. **Non-bootstrap alternatives** (Monte Carlo methods, model-based approaches)

**What's not needed**:
- More parameters (LoRA already has 300× excess capacity even for 100× speedups)

---

## Conclusion

This information-theoretic analysis reveals fundamental structure in RL for LLMs:

**The 1-bit bottleneck**: Policy gradient compresses rich token-level dynamics into scalar returns, creating a {{< math >}}$\leq \log_2(B)${{< /math >}} bits/episode ceiling. This explains:
- LoRA's success (capacity naturally matches ceiling)
- Sample inefficiency (1000s of episodes needed)
- Why more parameters don't help (bottleneck is signal sparsity)

**The dense signal bound**: Actor-critic has a mathematical upper bound of {{< math >}}$\leq T \log_2(B_\delta)${{< /math >}} bits/episode under independence assumptions—up to 8000× higher than policy gradient. However, structural correlation in bootstrap methods means the **achievable bound** is likely much lower. Empirical speedups of 10-100× may be closer to fundamental limits, with the remaining gap representing inherent constraints of TD learning rather than purely algorithmic opportunities.

**The path forward**: Future work should focus on understanding which parts of the 8000× gap are fundamental versus algorithmic. Better critic training may yield incremental improvements, but bootstrap correlation likely imposes a ceiling well below the independence bound. Progress requires either accepting this fundamental limit or exploring non-bootstrap alternatives.

The ceiling is information-theoretic and fundamental. Working within it requires understanding the tradeoffs between signal density, correlation, and approximation quality.

---

## References

**Empirical RL Speedups**:

[1] Mnih, V., et al. (2016). "Asynchronous Methods for Deep Reinforcement Learning." *ICML*. (A3C demonstrates 2-3× wall-clock speedup, higher in sample efficiency)

[2] Schulman, J., et al. (2015). "High-Dimensional Continuous Control Using Generalized Advantage Estimation." *ICLR*. (GAE shows 2-10× sample efficiency improvement)

[3] Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press. (Chapter 13 discusses actor-critic speedups)

[4] Recent work on value-based methods for LLMs (area under development)

**LLM Fine-Tuning**:
- Ouyang, L., et al. (2022). "Training language models to follow instructions with human feedback." *NeurIPS*. (InstructGPT)
- Hu, E. J., et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." *ICLR*.
- Schulman, J., et al. (2017). "Proximal Policy Optimization Algorithms." *arXiv*.

**Information Theory**:
- Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory* (2nd ed.). Wiley-Interscience.
- Russo, D., & Van Roy, B. (2014). "Learning to Optimize via Information-Directed Sampling." *Operations Research*, 66(1), 230-252.

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
