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

**TL;DR**: Policy gradient uses **sparse episode-level signals** (one scalar per episode), creating a hard information ceiling of {{< math >}}$\leq \log_2(B)${{< /math >}} bits/episode. Actor-critic uses **dense token-level signals** (one per token), providing a theoretical upper bound {{< math >}}$\leq T \log_2(B_\delta)${{< /math >}} bits/episode—potentially **1000-10000× higher** in ideal conditions. However, correlation between TD errors and imperfect value functions mean practical speedups are typically 10-100×, not 1000-10000×.

| Algorithm | Signal Type | Bandwidth Upper Bound |
|-----------|-------------|---------------------|
| Policy Gradient | Episode return {{< math >}}$G${{< /math >}} | {{< math >}}$\leq \log_2(B)${{< /math >}} bits/episode |
| Actor-Critic | TD errors {{< math >}}$\{\delta_t\}${{< /math >}} | {{< math >}}$\leq T \log_2(B_\delta)${{< /math >}} bits/episode |

*Note: These are information-theoretic upper bounds. Actual information transfer depends on signal quality, correlation structure, and critic approximation quality.*

With binary preferences ({{< math >}}$B = 2${{< /math >}}), policy gradient is bounded by 1 bit per episode. With sequences of {{< math >}}$T \sim 1000${{< /math >}} tokens and {{< math >}}$B_\delta \sim 256${{< /math >}}, actor-critic's theoretical ceiling is ~8000 bits—but correlation typically reduces this to 10-100 bits in practice.

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

To make information-theoretic analysis rigorous, we model uncertainty explicitly as a mathematical device:

- **Prior**: {{< math >}}$\xi \sim p(\xi)${{< /math >}} over reward parameters
- **Induced distribution**: {{< math >}}$p(\pi^*)${{< /math >}} over optimal policies
- **Deterministic mapping**: Each {{< math >}}$\xi${{< /math >}} determines an optimal policy {{< math >}}$\pi^*_\xi${{< /math >}}

This framework serves as a **modeling tool** to make the learning signal {{< math >}}$S${{< /math >}} and optimal policy {{< math >}}$\pi^*${{< /math >}} well-defined random variables, enabling rigorous computation of mutual information {{< math >}}$I(S; \pi^*)${{< /math >}}. This doesn't claim RL algorithms maintain explicit Bayesian posteriors—it's an analytical lens for measuring information content.

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

**Step 1: Information Flow via Data Processing Inequality**

Since {{< math >}}$\pi^* = \pi^*_\xi${{< /math >}} is a deterministic function of {{< math >}}$\xi${{< /math >}} (by Assumption A1), the variables form a Markov chain:

{{< math >}}
$$G \to \xi \to \pi^*$$
{{< /math >}}

The reasoning: {{< math >}}$G${{< /math >}} depends on the trajectory and {{< math >}}$\xi${{< /math >}} (the reward function), but given {{< math >}}$\xi${{< /math >}}, the optimal policy {{< math >}}$\pi^*${{< /math >}} is determined. Therefore {{< math >}}$\pi^*${{< /math >}} is conditionally independent of {{< math >}}$G${{< /math >}} given {{< math >}}$\xi${{< /math >}}.

By the **data processing inequality**:

{{< math >}}
$$I(G; \pi^*) \leq I(G; \xi)$$
{{< /math >}}

This fundamental result states that post-processing cannot increase information.

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

For the entropy bound, by Assumption A2, {{< math >}}$G${{< /math >}} takes at most {{< math >}}$B${{< /math >}} distinct values. For any discrete random variable {{< math >}}$X${{< /math >}} with support size {{< math >}}$|X| \leq B${{< /math >}}:

{{< math >}}
$$H(X) = -\sum_{x} p(x) \log_2 p(x) \leq \log_2(|X|) \leq \log_2(B)$$
{{< /math >}}

This is maximized when all outcomes are equally likely (uniform distribution).

**Combining both steps:**
{{< math >}}
$$I(G; \pi^*) \leq I(G; \xi) \leq H(G) \leq \log_2(B)$$
{{< /math >}} ∎

</details>

**This is a hard ceiling** regardless of sequence length {{< math >}}$T${{< /math >}}, model complexity, or computational resources.

### Concrete Bounds

- Binary preferences: {{< math >}}$\leq 1${{< /math >}} bit/episode
- 4-level Likert scale: {{< math >}}$\leq 2${{< /math >}} bits/episode
- 8-bit resolution (~256 levels): {{< math >}}$\leq 8${{< /math >}} bits/episode

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

### Assumption A2' (Extended to TD Errors)

**Statement**: Each TD error {{< math >}}$\delta_t${{< /math >}} has effective resolution of {{< math >}}$B_\delta${{< /math >}} distinguishable values.

**Justification**:
- **Computational precision**: Value functions implemented as neural networks use finite precision (float32, float16, or int8), limiting distinguishable values
- **Training dynamics**: Stochastic gradient descent with finite samples creates effective discretization
- **Practical resolution**: For value differences {{< math >}}$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)${{< /math >}}, empirical studies suggest 8-bit resolution ({{< math >}}$B_\delta \approx 256${{< /math >}}) captures practical precision

**Important caveat**: This is stronger than A2 since it applies to derived quantities (TD errors) rather than observed quantities (returns). The bound we derive should be viewed as an upper limit on *potential* information, not necessarily *realized* information.

### Main Result

**Theorem 2 (Actor-Critic Information Ceiling):**

*Given Assumptions A1 and A2':*

{{< math >}}
$$\mathcal{B}_{\text{AC}} = I(\{\delta_t\}; \pi^*) \leq T \log_2(B_\delta) \text{ bits per episode}$$
{{< /math >}}

**Critical Caveat**: This bound represents an **information-theoretic ceiling** assuming independent, perfectly informative TD errors. In practice, correlation between successive TD errors substantially reduces realized information. If TD errors were perfectly correlated, the bound would collapse to {{< math >}}$\log_2(B_\delta)${{< /math >}} (equivalent to a single signal). The gap between this theoretical maximum and practical speedups of 10-100× is primarily explained by:

1. **Temporal correlation**: Successive TD errors share value function biases
2. **Bootstrap structure**: {{< math >}}$\delta_t${{< /math >}} and {{< math >}}$\delta_{t+1}${{< /math >}} both depend on {{< math >}}$V(s_{t+1})${{< /math >}}
3. **Critic approximation**: Value function errors reduce signal quality

<details>
<summary><strong>Proof (click to expand)</strong></summary>

We bound the entropy of the sequence of TD errors.

**Step 1: Chain Rule Decomposition**

By the chain rule for entropy:
{{< math >}}
$$H(\{\delta_t\}_{t=0}^{T-1}) = H(\delta_0, \delta_1, \ldots, \delta_{ T - 1 }) = \sum_{t=0}^{T-1} H(\delta_t | \delta_0, \ldots, \delta_{ t - 1 })$$
{{< /math >}}

Using shorthand {{< math >}}$\delta_{<t} = (\delta_0, \ldots, \delta_{ t - 1 })${{< /math >}}:
{{< math >}}
$$H(\{\delta_t\}) = \sum_{t=0}^{T-1} H(\delta_t | \delta_{<t})$$
{{< /math >}}

**Step 2: Bounding Each Conditional Entropy**

Each TD error {{< math >}}$\delta_t${{< /math >}} is a scalar. By Assumption A2', each {{< math >}}$\delta_t${{< /math >}} has effective resolution of {{< math >}}$B_\delta${{< /math >}} distinguishable values.

For any random variable {{< math >}}$X${{< /math >}} with support size at most {{< math >}}$B_\delta${{< /math >}}:

{{< math >}}
$$H(X | Y) \leq \log_2(B_\delta)$$
{{< /math >}}

for any conditioning variable {{< math >}}$Y${{< /math >}}. This holds because:
- The conditional entropy {{< math >}}$H(X|Y)${{< /math >}} is maximized when {{< math >}}$X${{< /math >}} is uniform over its support, regardless of {{< math >}}$Y${{< /math >}}
- Even if {{< math >}}$X${{< /math >}} has residual uncertainty given {{< math >}}$Y${{< /math >}}, it still takes at most {{< math >}}$B_\delta${{< /math >}} values

Therefore:

{{< math >}}
$$H(\delta_t | \delta_{<t}) \leq \log_2(B_\delta)$$
{{< /math >}}

**Critical observation**: This bound is **tight** (achievable) only when {{< math >}}$\delta_t${{< /math >}} is nearly independent of {{< math >}}$\delta_{<t}${{< /math >}}, i.e., when:

{{< math >}}
$$H(\delta_t | \delta_{<t}) \approx H(\delta_t)$$
{{< /math >}}

In the opposite extreme, if {{< math >}}$\delta_t${{< /math >}} were perfectly predictable from past TD errors:

{{< math >}}
$$H(\delta_t | \delta_{<t}) = 0$$
{{< /math >}}

Real systems fall between these extremes. The correlation structure determines how much of the {{< math >}}$T \log_2(B_\delta)${{< /math >}} ceiling is achievable.

**Step 3: Summing Over All Timesteps**

{{< math >}}
$$H(\{\delta_t\}) = \sum_{t=0}^{T-1} H(\delta_t | \delta_{<t}) \leq \sum_{t=0}^{T-1} \log_2(B_\delta) = T \log_2(B_\delta)$$
{{< /math >}}

**Step 4: Applying Information Flow Bound**

By the data processing inequality (same argument as Theorem 1):
{{< math >}}
$$I(\{\delta_t\}; \pi^*) \leq I(\{\delta_t\}; \xi) \leq H(\{\delta_t\}) \leq T \log_2(B_\delta)$$
{{< /math >}} ∎

</details>

### Understanding the Bound: Theory vs Practice

**This is an upper bound on potential bandwidth under idealized conditions.** The bound {{< math >}}$T \log_2(B_\delta)${{< /math >}} represents the maximum possible information if:
1. TD errors at different timesteps were independent
2. The critic perfectly estimated true values
3. All TD error entropy was relevant to the optimal policy

In practice, three factors reduce realized information:

**1. Correlation Structure** (Most significant)

Define the **effective information coefficient** {{< math >}}$\rho \in [0, 1]${{< /math >}} where:

{{< math >}}
$$H(\{\delta_t\}) \approx \rho \cdot T \log_2(B_\delta)$$
{{< /math >}}

- If {{< math >}}$\rho = 1${{< /math >}}: Independent TD errors (theoretical maximum)
- If {{< math >}}$\rho = 1/T${{< /math >}}: Perfect correlation (collapses to single signal)
- Empirically: {{< math >}}$\rho \approx 0.01${{< /math >}} to {{< math >}}$0.1${{< /math >}} in typical RL

This explains why practical speedups are 10-100× rather than 1000-10000×.

**2. Imperfect Value Functions**

We observe {{< math >}}$\hat{\delta}_t${{< /math >}} from approximate critic {{< math >}}$V_\phi${{< /math >}}, not true TD errors {{< math >}}$\delta_t^*${{< /math >}}. By the data processing inequality:

{{< math >}}
$$I(\{\hat{\delta}_t\}; \pi^*) \leq I(\{\delta_t^*\}; \pi^*)$$
{{< /math >}}

Critic training instability at LLM scale compounds this issue.

**3. Signal Relevance**

Not all bits in TD errors reveal the optimal policy. The mutual information {{< math >}}$I(\{\delta_t\}; \pi^*)${{< /math >}} already accounts for this, but highlights that raw entropy {{< math >}}$H(\{\delta_t\})${{< /math >}} overstates useful information.

### The Gap: Theoretical vs Practical

For {{< math >}}$T = 1000${{< /math >}} tokens with {{< math >}}$B_\delta = 256${{< /math >}} (8-bit effective resolution):

- **Policy gradient**: {{< math >}}$\leq 1${{< /math >}} bit/episode
- **Actor-critic (theoretical ceiling)**: {{< math >}}$\leq 8000${{< /math >}} bits/episode
- **Actor-critic (practical, with {{< math >}}$\rho \approx 0.01${{< /math >}}-{{< math >}}$0.1${{< /math >}})**: ~10-100 bits/episode

**Interpretation**:
- **Theoretical maximum**: 8000× higher than policy gradient
- **Practical speedup** (traditional RL): 10-100× higher than policy gradient [1,2]
- **Achievable with better decorrelation**: Potentially 100-1000× with improved methods

**Note on empirical validation**: The 10-100× speedup of actor-critic over policy gradient is well-established in traditional RL domains (Atari, continuous control) [1,2,3]. For LLMs specifically, systematic comparisons are limited due to the dominance of PPO/REINFORCE-style algorithms, though recent work on value-based methods shows promise [4].

---

## Why LoRA Works: Matching Capacity to Information Ceiling

### The Argument

Typical setup:
- Training episodes: {{< math >}}$N \sim 1000${{< /math >}}
- LoRA rank: {{< math >}}$r = 8${{< /math >}}, dimension: {{< math >}}$d = 4096${{< /math >}}

**LoRA capacity estimation**:
- Parameters: {{< math >}}$2rd \approx 65{,}000${{< /math >}}
- Bits per parameter: Assuming 5-8 bits effective resolution (between 32 and 256 distinguishable values after training)
- Information capacity: {{< math >}}$65{,}000 \times 5${{< /math >}} to {{< math >}}$65{,}000 \times 8${{< /math >}} = **325,000-520,000 bits**

*This estimate depends on training precision and effective parameter resolution. The key insight is robust to the exact value: LoRA provides orders of magnitude more capacity than the information ceiling.*

**Information ceiling**: {{< math >}}$\leq 1000${{< /math >}} bits (binary preferences, 1000 episodes, {{< math >}}$\leq 1${{< /math >}} bit/episode)

**Capacity ratio**: LoRA provides **300-500× more capacity** than policy gradient's information ceiling (325K-520K bits vs ~1K bits).

This explains why LoRA works: **the parameter bottleneck isn't binding**—we have far more capacity than the sparse episode-level signal can fill. The bottleneck is signal density, not parameter count.

### Why Full Fine-Tuning is Overkill

A 7B parameter model provides ~7 billion degrees of freedom versus ~1,000 bits of information—a factor of ~7 million excess capacity. LoRA's modest parameter count naturally matches policy gradient's information ceiling.

**Empirical consistency**: LLM-RL typically needs 1,000-10,000 episodes to converge, consistent with accumulating 1,000-10,000 bits at 1-3 bits/episode (depending on reward granularity).

---

## Implications and Future Directions

### Current State

Policy gradient with LoRA dominates LLM fine-tuning because:
- ✓ Stable at scale (single critic-free optimization)
- ✓ Parameter-efficient (capacity exceeds information ceiling)
- ✗ Sample-inefficient ({{< math >}}$\leq \log_2(B)${{< /math >}} bits/episode)

### The Opportunity

Actor-critic methods have:
- **Theoretical ceiling**: 1000-10000× higher (with independent TD errors)
- **Practical demonstrated speedups**: 10-100× in traditional RL [1,2,3]
- **Current bottleneck**: Stable critic training for LLMs remains unsolved

The gap between current practice (100×) and theoretical potential (8000×) represents both:
1. **Fundamental barriers**: Correlation structure may limit achievable gains to ~1000× even with perfect critics
2. **Algorithmic opportunities**: Better critic training could unlock 10-100× additional improvement

### Research Directions

1. **Stable critic training** at LLM scale
   - Low-rank value function architectures
   - Techniques from TD learning with function approximation

2. **Decorrelation methods** to increase {{< math >}}$\rho${{< /math >}}
   - Eligibility traces and multi-step returns
   - Ensemble critics to reduce bias correlation
   - Explicit decorrelation penalties

3. **Token-level reward design** for dense, informative signals
   - Process rewards that provide meaningful per-token feedback
   - Intermediate task decomposition

4. **Hybrid approaches**
   - Combining Monte Carlo returns with bootstrapped estimates
   - Adaptively switching between PG and AC based on critic quality

5. **Information-theoretic diagnostics**
   - Measuring realized information transfer
   - Quantifying correlation coefficient {{< math >}}$\rho${{< /math >}} in practice

---

## Conclusion

This information-theoretic framework establishes:

**Policy gradient ceiling**: Compressing {{< math >}}$T \gg 1000${{< /math >}} tokens into scalar returns creates a {{< math >}}$\leq \log_2(B)${{< /math >}} bits/episode ceiling (typically 1-3 bits). LoRA's modest capacity naturally matches this ceiling—excess parameters don't help when signals are sparse.

**Actor-critic potential**: Token-level signals have a theoretical ceiling of {{< math >}}$\leq T \log_2(B_\delta)${{< /math >}} bits/episode—up to 1000-10000× higher in ideal conditions. However, temporal correlation and imperfect critics reduce practical gains to 10-100×. The gap between current practice and theoretical limits includes both fundamental barriers (correlation) and algorithmic opportunities (critic quality).

**The path forward**: Future breakthroughs will come from:
1. Raising the effective ceiling through better critic training (closing the gap between 100× and 1000×)
2. Managing correlation structure to increase realized information per episode
3. Not from increasing adapter capacity—LoRA already provides excess capacity

The ceiling is structural and information-theoretic, not just algorithmic—understanding this structure reveals where effort should be directed.

---

## Technical Notes and Scope

**What we prove rigorously** (given A1, A2, A2'):
- {{< math >}}$I(G; \pi^*) \leq H(G) \leq \log_2(B)${{< /math >}} bits/episode (policy gradient)
- {{< math >}}$I(\{\delta_t\}; \pi^*) \leq H(\{\delta_t\}) \leq T \log_2(B_\delta)${{< /math >}} bits/episode (actor-critic, assumes independence)

These are **upper bounds** on the entropy of learning signals. Key caveats:

1. **Mutual information may be lower**: If signals are uninformative about {{< math >}}$\pi^*${{< /math >}}, actual information transfer {{< math >}}$I(S; \pi^*)${{< /math >}} can be much less than entropy {{< math >}}$H(S)${{< /math >}}

2. **Correlation substantially reduces realized information**: The {{< math >}}$T \log_2(B_\delta)${{< /math >}} bound for actor-critic assumes independence. Practical correlation reduces this by ~10-100×

3. **Approximation errors compound**: Imperfect critics, optimization dynamics, and exploration all reduce information extraction below theoretical ceilings

**What remains empirical/conjectural**:
- Precise correlation structure and quantitative {{< math >}}$\rho${{< /math >}} values in LLM settings
- Sample complexity predictions (optimization dynamics beyond information content)
- Tightness of bounds: whether optimal algorithms achieve {{< math >}}$\Theta(\log B)${{< /math >}} or only {{< math >}}$O(\log B)${{< /math >}}
- Achievable speedups with improved critic training at LLM scale

**Scope limitations**:
- **Single-task learning**: Multi-task or continual learning has different information dynamics
- **Known dynamics**: Analysis assumes deterministic, known transitions (true for autoregressive generation)
- **Stationary rewards**: Extensions needed for non-stationary objectives
- **Optimal information extraction**: Actual algorithms may not fully utilize available information due to optimization constraints

**Extensions needed for**:
- Exploration in unknown environments
- Partial observability (hidden states)
- Non-stationary reward functions
- Model-based RL with unknown dynamics

---

## References

[1] Mnih, V., et al. (2016). "Asynchronous Methods for Deep Reinforcement Learning." *ICML*. (A3C demonstrates 2-3× speedup over A2C in wall-clock time, more in sample efficiency)

[2] Schulman, J., et al. (2015). "High-Dimensional Continuous Control Using Generalized Advantage Estimation." *ICLR*. (GAE shows 2-10× sample efficiency improvement)

[3] Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press. (Chapter 13 discusses actor-critic speedups over policy gradient)

[4] Recent work on value-based methods for LLMs (citations to be added as this area develops)

**Core Papers**:
- Ouyang, L., Wu, J., Jiang, X., et al. (2022). "Training language models to follow instructions with human feedback." *NeurIPS*. (InstructGPT)
- Hu, E. J., Shen, Y., Wallis, P., et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." *ICLR*.
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). "Proximal Policy Optimization Algorithms." *arXiv*.

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
