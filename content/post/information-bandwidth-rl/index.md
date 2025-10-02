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

**TL;DR**: Policy gradient uses **sparse episode-level signals** (one scalar per episode), while actor-critic uses **dense token-level signals** (one per token). This difference in signal density determines information bandwidth and explains both why LoRA works and where future improvements lie.

| Algorithm | Signal Type | Bandwidth Potential |
|-----------|-------------|---------------------|
| Policy Gradient | Episode return {{< math >}}$G${{< /math >}} | {{< math >}}$O(1)${{< /math >}} bits/episode |
| Actor-Critic | TD errors {{< math >}}$\{\delta_t\}${{< /math >}} | {{< math >}}$O(T)${{< /math >}} bits/episode |

With sequences of {{< math >}}$T \sim 1000${{< /math >}} tokens, the theoretical difference is ~1000×. Whether this potential is realized depends on implementation challenges, particularly training stable value functions for LLMs.

---

## Framework: Token-Level MDPs and Information Flow

### The Token Generation Process

When we fine-tune a language model with RL, we're working with a specific type of MDP:

**States**: Token sequences {{< math >}}$s = (x_1, \ldots, x_t)${{< /math >}}
**Actions**: Next token {{< math >}}$a = x_{t+1}${{< /math >}} from vocabulary
**Transitions**: Deterministic concatenation {{< math >}}$s' = s \circ a${{< /math >}}
**Rewards**: Determined by unknown parameter {{< math >}}$\xi${{< /math >}} (representing human preferences, task objectives, etc.)

The crucial property: **transitions are deterministic and known**. All uncertainty lives in the reward function.

### Bayesian RL Framework

Instead of treating rewards as fixed, we model uncertainty explicitly:

- Prior: {{< math >}}$\xi \sim p(\xi)${{< /math >}} over reward parameters
- This induces a distribution {{< math >}}$p(\pi^*)${{< /math >}} over optimal policies
- Each {{< math >}}$\xi${{< /math >}} determines a unique optimal policy {{< math >}}$\pi^*_\xi${{< /math >}}

This makes both the learning signal {{< math >}}$S${{< /math >}} and optimal policy {{< math >}}$\pi^*${{< /math >}} well-defined random variables, allowing us to rigorously compute mutual information {{< math >}}$I(S; \pi^*)${{< /math >}}.

### Information Bandwidth

We define the **information bandwidth** of an RL algorithm as:

{{< math >}}
$$\mathcal{B} = I(S; \pi^*)$$
{{< /math >}}

This measures how much the learning signal {{< math >}}$S${{< /math >}} reduces uncertainty about the optimal policy {{< math >}}$\pi^*${{< /math >}}. It's bounded above by the signal's entropy: {{< math >}}$I(S; \pi^*) \leq H(S)${{< /math >}}.

The key question: what is {{< math >}}$S${{< /math >}} for different algorithms?

---

## Policy Gradient: Episode-Level Signals

### The Algorithm

1. Sample trajectory {{< math >}}$\tau = (s_0, a_0, \ldots, s_T)${{< /math >}}
2. Observe return {{< math >}}$G = R_\xi(s_T)${{< /math >}} (typically sparse—reward only at episode end)
3. Update: {{< math >}}$\theta \leftarrow \theta + \alpha \nabla_\theta \log p_\theta(\tau) \cdot G${{< /math >}}

**Learning signal**: {{< math >}}$S = G${{< /math >}} (a single scalar)

### Information Analysis

The return {{< math >}}$G${{< /math >}} compresses the entire sequence of {{< math >}}$T \gg 1000${{< /math >}} tokens into one number. Even if we could perfectly distinguish reward values, the signal is fundamentally **sparse**:

{{< math >}}
$$\mathcal{B}_{\text{potential}} = H(G) = O(1) \text{ bits per episode}$$
{{< /math >}}

The constant depends on reward resolution:
- Binary preferences: ~1 bit
- 4-level Likert scale: ~2 bits
- 5-level scale: ~2.3 bits
- Continuous with effective resolution ~10 levels: ~3.3 bits

This creates a severe **information bottleneck**. No matter how long the sequence or how sophisticated the model, policy gradient learns at most a few bits per episode.

**Why the bound holds**: The signal {{< math >}}$S = G${{< /math >}} is scalar. Its entropy is bounded by the number of distinguishable values. This is independent of sequence length {{< math >}}$T${{< /math >}}—we could generate 1000 tokens or 10,000 tokens, but still extract only {{< math >}}$O(1)${{< /math >}} bits of information about which policy is optimal.

---

## Actor-Critic: Token-Level Signals

### The Algorithm

At each timestep {{< math >}}$t${{< /math >}}:
1. Select action {{< math >}}$a_t \sim \pi_\theta(\cdot | s_t)${{< /math >}}
2. Observe reward {{< math >}}$r_t = R_\xi(s_t)${{< /math >}} (often 0 for non-terminal states)
3. Compute TD error: {{< math >}}$\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)${{< /math >}}
4. Update actor and critic using {{< math >}}$\delta_t${{< /math >}}

**Learning signal**: {{< math >}}$S = \{\delta_t\}_{t=0}^{T-1}${{< /math >}} (one per token)

### Information Analysis

Instead of one scalar per episode, we get {{< math >}}$T${{< /math >}} signals:

{{< math >}}
$$\mathcal{B}_{\text{potential}} = \sum_{t=0}^{T-1} H(\delta_t | \text{past}) = O(T) \text{ bits per episode}$$
{{< /math >}}

For sequences of length {{< math >}}$T = 1000${{< /math >}}, this is theoretically ~1000× more information bandwidth than policy gradient.

**Critical requirement**: Realizing this potential requires a well-trained value function {{< math >}}$V_\phi${{< /math >}}. The TD errors provide useful information only if the critic accurately estimates future rewards.

### The Challenge for LLMs

In traditional RL (Atari, robotics), actor-critic methods like PPO achieve 10-100× speedups over policy gradient. This validates that dense signals provide real advantages when critics work.

**However**: For LLMs, stable critic training at scale remains unsolved. Challenges include:
- Value function training instability with billion-parameter models
- Long-horizon credit assignment ({{< math >}}$T \gg 1000${{< /math >}} tokens)
- Bootstrap error accumulation
- Computational overhead

This is why current LLM-RL systems predominantly use policy gradient despite its information bottleneck.

---

## Why LoRA Works: Matching Capacity to Information

Now we can explain why LoRA is so effective for RL fine-tuning.

### The Capacity Argument

Consider typical numbers:
- Training episodes: {{< math >}}$N \sim 1000${{< /math >}}
- LoRA rank: {{< math >}}$r = 8${{< /math >}}
- Model dimension: {{< math >}}$d = 4096${{< /math >}}

**LoRA capacity**: {{< math >}}$2rd = 2 \times 8 \times 4096 \approx 65{,}000${{< /math >}} trainable parameters

**Information accumulated**: With policy gradient providing {{< math >}}$O(1)${{< /math >}} bits per episode over ~1000 episodes, we accumulate roughly 1000-3000 bits of information about which policy is optimal.

**Degrees of freedom comparison**:
- Parameters to optimize: ~65,000
- Training samples: ~1,000
- Ratio: ~60× more parameters than episodes

This suggests the parameter bottleneck isn't binding—we have far more capacity than needed to represent the limited information extracted from episode-level signals.

### Why Full Fine-Tuning is Overkill

Full fine-tuning a 7B parameter model provides **100,000×** more capacity than LoRA. Given that policy gradient provides only {{< math >}}$O(1)${{< /math >}} bits per episode, this massive capacity is wasted:

- With 1000 episodes providing ~2000 bits of information
- Full fine-tuning offers billions of degrees of freedom
- The parameter space is vastly larger than the information content

LoRA's modest capacity naturally matches policy gradient's information bottleneck. This explains empirical findings that higher LoRA ranks (16, 32, 64) provide minimal gains—we're already not parameter-limited.

---

## Implications and Future Directions

### Current Practice: It Works, But Slowly

Policy gradient with LoRA is the dominant approach because:
- ✓ Stable and reliable at scale
- ✓ Parameter-efficient (LoRA capacity matches information flow)
- ✓ Simple to implement
- ✗ Sample-inefficient ({{< math >}}$O(1)${{< /math >}} bits/episode)

Typical LLM fine-tuning requires 1,000-10,000 episodes. This is consistent with extracting a few bits per episode.

### The 1000× Opportunity

Our analysis reveals a massive opportunity: **developing stable value-based methods for LLMs could improve sample efficiency by orders of magnitude**.

The information-theoretic potential is clear:
- Policy gradient: {{< math >}}$O(1)${{< /math >}} bits/episode → ~1000 episodes needed
- Actor-critic: {{< math >}}$O(T)${{< /math >}} bits/episode → ~10 episodes needed (if critics worked perfectly)

Traditional RL shows 10-100× speedups are achievable. For LLMs with {{< math >}}$T \sim 1000${{< /math >}} tokens, the potential gains are even larger.

**Current bottleneck**: We lack reliable recipes for training value functions at LLM scale. This is the highest-leverage research direction suggested by this analysis.

### Specific Research Directions

**1. Stable Critic Training**:
- Develop architectures and training procedures for value learning at scale
- Handle long-horizon credit assignment ({{< math >}}$T \gg 1000${{< /math >}})
- Leverage pre-trained representations

**2. Token-Level Reward Design**:
- Current systems use sparse outcome-based rewards
- Dense token-level rewards could unlock {{< math >}}$O(T)${{< /math >}} bandwidth
- Challenge: generating meaningful rewards without manual annotation at each token

**3. Hybrid Approaches**:
- Combine Monte Carlo returns with value bootstrapping
- Use reward models to generate dense pseudo-rewards
- Partial value information could provide {{< math >}}$O(T/k)${{< /math >}} bits for some {{< math >}}$k < T${{< /math >}}

**4. Low-Rank Value Functions**:
- If full critic is too costly, can we learn compressed value representations?
- The critic only needs to capture {{< math >}}$O(T)${{< /math >}} bits about expected rewards
- Explore hierarchical or factored value decomposition

---

## Practical Recommendations

### For Current Practice

**LoRA rank selection**: With {{< math >}}$O(1)${{< /math >}} bits/episode, modest ranks suffice:
- {{< math >}}$r = 8${{< /math >}}: Sufficient for 1000-5000 episodes
- {{< math >}}$r = 16${{< /math >}}: Ample margin for most use cases
- {{< math >}}$r > 32${{< /math >}}: Unlikely to help unless training for 10,000+ episodes

**Convergence expectations**:
- 1000 episodes: Basic alignment
- 3000-5000 episodes: Strong performance
- 10,000+ episodes: Diminishing returns (approaching information saturation)

**Multi-task RL**: When fine-tuning on {{< math >}}$K${{< /math >}} tasks simultaneously, information accumulates faster ({{< math >}}$O(K)${{< /math >}} per episode). Consider higher LoRA ranks: {{< math >}}$r = 16K${{< /math >}} for {{< math >}}$K${{< /math >}} tasks.

### For Research

**Priority**: Solve stable critic training for LLMs. This single breakthrough could unlock 100-1000× sample efficiency gains.

**Next steps**:
1. Benchmark value function learning on medium-scale models (1B-7B parameters)
2. Develop training recipes that work reliably
3. Scale successful approaches to 70B+ parameter models

The information-theoretic lens shows this isn't just an engineering challenge—it's the fundamental bottleneck in LLM-RL efficiency.

---

## Conclusion

This information-theoretic framework reveals a clean story:

**Policy gradient's bottleneck**: Compressing {{< math >}}$T \gg 1000${{< /math >}} tokens into scalar returns creates an {{< math >}}$O(1)${{< /math >}} bits/episode ceiling. This naturally matches LoRA's modest capacity.

**Actor-critic's potential**: Token-level TD errors could provide {{< math >}}$O(T)${{< /math >}} bits/episode—orders of magnitude more information. But realizing this requires solving stable critic training for LLMs.

**The path forward**: Current methods work because LoRA capacity matches policy gradient's information flow. Future breakthroughs will come from unlocking the dense signal advantage through effective value function learning.

The "1 bit per episode" observation isn't just a curiosity—it explains why current methods work and where the biggest opportunities lie.

---

## Technical Notes

**What this analysis proves rigorously**:
- Policy gradient uses sparse signals with {{< math >}}$O(1)${{< /math >}} entropy per episode
- Actor-critic uses dense signals with {{< math >}}$O(T)${{< /math >}} entropy per episode
- Signal density determines information bandwidth potential

**What remains conjectural**:
- Specific bit counts (1-4 bits) depend on reward resolution
- Actor-critic achieving {{< math >}}$O(T)${{< /math >}} bits requires unproven assumptions about critic quality
- Sample complexity predictions depend on optimization dynamics

**Scope**: This framework applies to stationary reward learning with known dynamics (autoregressive generation). Extensions to exploration, partial observability, or agentic RL with unknown environment dynamics require additional analysis.

**For the full mathematical treatment**: See [index.md](index.md) for detailed proofs, assumption analysis, and technical appendices.

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
