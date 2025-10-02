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

**Policy gradient's hard limit**: Compressing 1000+ tokens into one scalar reward creates an information ceiling of {{< math >}}$\leq \log_2(B)${{< /math >}} bits per episode. For binary feedback, this is {{< math >}}$\leq 1${{< /math >}} bit/episode—explaining why training needs thousands of episodes and why LoRA's modest capacity (300-500× excess) suffices.

**Actor-critic's theoretical potential**: By bootstrapping historical knowledge through a learned critic, actor-critic methods generate dense per-token feedback. Under independence assumptions, this gives an upper bound of {{< math >}}$\leq T \log_2(B_\delta)${{< /math >}} bits/episode. For {{< math >}}$T=1000${{< /math >}} tokens and 8-bit TD errors, this ceiling is {{< math >}}$\leq 8000${{< /math >}} bits/episode—potentially 8000× higher than policy gradient.

**The practical implication**: LoRA already provides 300-500× excess capacity relative to policy gradient's information ceiling. Even with substantial improvements in actor-critic methods, LoRA's capacity appears sufficient for foreseeable applications.

| Algorithm | Signal Density | Information Upper Bound |
|-----------|---------------|---------------------|
| Policy Gradient | 1 scalar/episode | {{< math >}}$\leq 1${{< /math >}} bit/episode (binary) |
| Actor-Critic | {{< math >}}$T${{< /math >}} scalars/episode | {{< math >}}$\leq 8000${{< /math >}} bits/episode (theoretical ceiling) |

**Note**: The 8000 bits/episode ceiling assumes independent TD errors—an assumption violated in practice by bootstrap methods. The actual achievable information bandwidth remains an open empirical question.

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

Since {{< math >}}$\pi^*_\xi${{< /math >}} is a deterministic function of {{< math >}}$\xi${{< /math >}} alone (by Assumption A1), the optimal policy {{< math >}}$\pi^*${{< /math >}} contains no information beyond what {{< math >}}$\xi${{< /math >}} specifies. Therefore, once we condition on {{< math >}}$\xi${{< /math >}}, learning {{< math >}}$G${{< /math >}} provides no additional information about {{< math >}}$\pi^*${{< /math >}}. Formally: {{< math >}}$\pi^* \perp G | \xi${{< /math >}}.

This conditional independence gives us the Markov chain:

{{< math >}}
$$G \to \xi \to \pi^*$$
{{< /math >}}

Reasoning: The optimal policy {{< math >}}$\pi^*${{< /math >}} is a deterministic function of {{< math >}}$\xi${{< /math >}} alone (by A1). Therefore, once {{< math >}}$\xi${{< /math >}} is given, {{< math >}}$\pi^*${{< /math >}} is fixed. Any other variable, including {{< math >}}$G${{< /math >}}, becomes conditionally independent of {{< math >}}$\pi^*${{< /math >}} given {{< math >}}$\xi${{< /math >}}.

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

**Learning signal**: {{< math >}}$S = \{\delta_t\}_{t=0}^{ T - 1 }${{< /math >}} (one signal per token)

Instead of waiting until the end for one scalar, we get feedback at **every step**.

### Where Does the Dense Signal Come From?

A natural question: "The environment only provides rewards at certain steps. How can actor-critic get more information per episode?"

**The key insight**: The extra information doesn't come from the environment *in the current episode*. It comes from the **critic's accumulated knowledge from all past episodes**.

The critic {{< math >}}$V_\phi(s)${{< /math >}} acts as compressed memory of all historical data. It learns to predict expected future returns from any state {{< math >}}$s${{< /math >}} based on thousands of previous rollouts.

Let's re-examine the TD error with this perspective:

{{< math >}}
$$\delta_t = \underbrace{(r_t + \gamma V_\phi(s_{t+1}))}_{\text{Observed outcome}} - \underbrace{V_\phi(s_t)}_{\text{Historical expectation}}$$
{{< /math >}}

- **{{< math >}}$V_\phi(s_t)${{< /math >}}**: The critic's prediction, representing the **historical average** of what should happen from state {{< math >}}$s_t${{< /math >}}
- **{{< math >}}$r_t + \gamma V_\phi(s_{t+1})${{< /math >}}**: The **observed outcome** of taking action {{< math >}}$a_t${{< /math >}}, incorporating one step of real environmental feedback {{< math >}}$r_t${{< /math >}}

The TD error {{< math >}}$\delta_t${{< /math >}} is a "surprise" signal—how much better or worse reality was compared to historical expectation. This signal is information-rich precisely *because* it compares against a learned model of the reward structure. The critic **bootstraps** knowledge from the past to provide dense, step-by-step feedback in the present.

**In short**: Actor-critic achieves higher information bandwidth by efficiently reusing historical data via the critic. Instead of treating each episode as independent (like policy gradient), the critic allows every step in the current episode to be evaluated against distilled knowledge of all previous trials.

**The tradeoff**: This bootstrapping comes with an inherent cost—because all TD errors depend on the same learned value function {{< math >}}$V_\phi${{< /math >}}, they become structurally correlated. The very mechanism that enables dense signals also introduces the correlation barrier that prevents us from achieving the full {{< math >}}$T \log_2(B_\delta)${{< /math >}} bound.

### Why TD Errors Are Fundamentally Correlated

Let's make the correlation concrete with an example:

**Scenario**: Suppose your critic systematically overestimates values by 20% everywhere (a common early-training phenomenon), where {{< math >}}$V_{\text{true}}${{< /math >}} denotes the true value function.

Then for a sequence of states:
- {{< math >}}$\delta_0 = r_0 + \gamma(1.2 \cdot V_{\text{true}}(s_1)) - 1.2 \cdot V_{\text{true}}(s_0)${{< /math >}}
- {{< math >}}$\delta_1 = r_1 + \gamma(1.2 \cdot V_{\text{true}}(s_2)) - 1.2 \cdot V_{\text{true}}(s_1)${{< /math >}}
- {{< math >}}$\delta_2 = r_2 + \gamma(1.2 \cdot V_{\text{true}}(s_3)) - 1.2 \cdot V_{\text{true}}(s_2)${{< /math >}}

Notice that **all TD errors share the same 1.2× bias**. If {{< math >}}$\delta_0${{< /math >}} is surprisingly negative (indicating overestimation), then {{< math >}}$\delta_1${{< /math >}} and {{< math >}}$\delta_2${{< /math >}} are likely also negative—they're positively correlated.

This correlation isn't a bug or implementation flaw. It's **inherent to bootstrap methods**: using the same approximate value function {{< math >}}$V_\phi${{< /math >}} across all states creates shared biases that induce correlation. Even as the critic improves, bootstrap methods maintain correlation structure because consecutive TD errors depend on overlapping value estimates.

This structural correlation is why the independence-based bound of {{< math >}}$T \log_2(B_\delta)${{< /math >}} is likely not achievable in practice—the actual achievable information bandwidth remains an open question.

### Extended Assumption

**Assumption A2' (Effective TD Resolution)**: For the purpose of upper bound analysis, we model each TD error {{< math >}}$\delta_t${{< /math >}} as having effective resolution {{< math >}}$B_\delta${{< /math >}} distinguishable values.

**This is a modeling assumption for theoretical analysis.** TD errors are continuous in practice, but we model their effective information content as finite due to:
- Finite precision arithmetic (float32: ~7 significant digits)
- Neural network optimization noise
- Gradient update granularity

We use **{{< math >}}$B_\delta = 256${{< /math >}} (8 bits) as an illustrative example**, representing roughly 2-3 significant digits of precision. The actual effective resolution is task-dependent and difficult to measure empirically. **The resulting bound should be interpreted as demonstrating the potential order-of-magnitude advantage** (hundreds to thousands of times higher than policy gradient) **rather than a precise quantitative prediction.**

For other reasonable choices:
- {{< math >}}$B_\delta = 16${{< /math >}} (4 bits): Upper bound becomes 4000 bits/episode
- {{< math >}}$B_\delta = 1024${{< /math >}} (10 bits): Upper bound becomes 10,000 bits/episode

The qualitative conclusion—that actor-critic has dramatically higher theoretical potential—holds across this range.

**Important caveat**: Unlike A2 (which applies to actual observations), A2' applies to derived learning signals. The resulting bound {{< math >}}$I(S; \pi^*) \leq T \log_2(B_\delta)${{< /math >}} is a *mathematical upper limit* on potential information flow, not a tight characterization of actual information.

### The Information Ceiling

**Theorem 2 (Actor-Critic Information Upper Bound):**

*Under assumptions A1 and A2', actor-critic's information bandwidth satisfies:*

{{< math >}}
$$I(\{\delta_t\}; \pi^*) \leq T \log_2(B_\delta) \text{ bits per episode}$$
{{< /math >}}

*This upper bound is mathematically sound but likely not achievable in practice due to correlation between TD errors.*

**For {{< math >}}$T=1000${{< /math >}} and {{< math >}}$B_\delta=256${{< /math >}}**: This theoretical ceiling is {{< math >}}$8000${{< /math >}} bits/episode—**8000× higher** than policy gradient's 1 bit.

**Intuition**: With {{< math >}}$T${{< /math >}} independent signals each carrying {{< math >}}$\log_2(B_\delta)${{< /math >}} bits, we get {{< math >}}$T \log_2(B_\delta)${{< /math >}} total bits. However, this assumes independence—an assumption violated by the bootstrap structure of TD learning.

<details>
<summary><strong>Detailed Proof (click to expand)</strong></summary>

We bound the entropy of the TD error sequence.

**Step 1: Chain Rule Decomposition**

By the chain rule for entropy:
{{< math >}}
$$H(\delta_0, \delta_1, \ldots, \delta_{ T - 1 }) = \sum_{t=0}^{ T - 1 } H(\delta_t | \delta_0, \ldots, \delta_{ t - 1 })$$
{{< /math >}}

Using {{< math >}}$\delta_{ < t } = (\delta_0, \ldots, \delta_{ t - 1 })${{< /math >}}:
{{< math >}}
$$H(\{\delta_t\}) = \sum_{t=0}^{ T - 1 } H(\delta_t | \delta_{ < t })$$
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
$$H(\delta_t | \delta_{ < t }) \leq \log_2(B_\delta)$$
{{< /math >}}

**Critical observation**: This bound is **tight** only when {{< math >}}$\delta_t${{< /math >}} is nearly independent of {{< math >}}$\delta_{ < t }${{< /math >}}:
{{< math >}}
$$H(\delta_t | \delta_{ < t }) \approx H(\delta_t)$$
{{< /math >}}

If perfectly correlated: {{< math >}}$H(\delta_t | \delta_{ < t }) = 0${{< /math >}}. Reality falls between these extremes.

**Step 3: Summing Over Time**

{{< math >}}
$$H(\{\delta_t\}) = \sum_{t=0}^{ T - 1 } H(\delta_t | \delta_{ < t }) \leq T \log_2(B_\delta)$$
{{< /math >}}

**Step 4: Information Flow Bound**

Since {{< math >}}$\pi^*${{< /math >}} is a deterministic function of {{< math >}}$\xi${{< /math >}} (by Assumption A1), we have:
{{< math >}}
$$I(\{\delta_t\}; \pi^*) = I(\{\delta_t\}; \xi)$$
{{< /math >}}

Mutual information is always bounded by the entropy of either variable:
{{< math >}}
$$I(\{\delta_t\}; \xi) \leq H(\{\delta_t\})$$
{{< /math >}}

Combining with Step 3:
{{< math >}}
$$I(\{\delta_t\}; \pi^*) = I(\{\delta_t\}; \xi) \leq H(\{\delta_t\}) \leq T \log_2(B_\delta)$$
{{< /math >}} ∎

**Critical note on the proof**: This bound treats TD errors as information-carrying signals about the reward parameter {{< math >}}$\xi${{< /math >}}. In practice, TD errors also depend on the learned value function {{< math >}}$V_\phi${{< /math >}} and training history. The bound represents an idealized scenario where we model what information about {{< math >}}$\xi${{< /math >}} (and thus {{< math >}}$\pi^*${{< /math >}}) could theoretically be extracted from the TD error sequence, assuming the value function perfectly captures relevant information from {{< math >}}$\xi${{< /math >}}. This idealization, combined with the independence assumption, makes this an upper limit on what could theoretically be achieved rather than a characterization of actual algorithms.

</details>

**⚠️ Critical Caveat**: This bound assumes **independent TD errors**—an assumption fundamentally violated by bootstrap methods. Successive TD errors are structurally correlated: both {{< math >}}$\delta_t${{< /math >}} and {{< math >}}$\delta_{t+1}${{< /math >}} depend on {{< math >}}$V(s_{t+1})${{< /math >}}, sharing value function biases. This correlation is not merely an implementation issue but inherent to TD learning, preventing the {{< math >}}$T \log_2(B_\delta)${{< /math >}} bound from being achievable. The actual achievable information bandwidth remains an open empirical question.

### Understanding the Gap Between Theory and Practice

For {{< math >}}$T=1000${{< /math >}}, {{< math >}}$B_\delta=256${{< /math >}}:

- **Policy gradient** (binary): {{< math >}}$\leq 1${{< /math >}} bit/episode
- **Actor-critic** (theoretical ceiling): {{< math >}}$\leq 8000${{< /math >}} bits/episode

The theoretical ceiling represents what could be achieved with independent TD errors. In practice, the achievable information bandwidth is likely much lower due to inherent correlation in bootstrap methods.

**Multiple factors contribute to this gap**:

1. **TD error correlation** (our main focus): Bootstrap methods create structural dependencies between consecutive TD errors

2. **Value function approximation error**: Neural networks can't perfectly represent value functions, introducing systematic biases beyond correlation

3. **Finite sample effects**: Each state is visited finitely often, creating statistical uncertainty in value estimates

4. **Optimization challenges**: Critic training is unstable, sensitive to hyperparameters, and may converge slowly

5. **Information utilization efficiency**: Even with perfect TD signals, gradient descent may not efficiently extract all available information

**Our hypothesis**: Correlation (factor 1) is a substantial barrier that accounts for a significant portion of the gap. However, disentangling these factors empirically is an important direction for future work.

---

## Part 4: Why LoRA Works—The Capacity-Information Match

### The Capacity Argument

Consider typical RLHF setup:
- Episodes: {{< math >}}$N = 1000${{< /math >}}
- LoRA: rank {{< math >}}$r=8${{< /math >}}, dimension {{< math >}}$d=4096${{< /math >}}
- Binary preferences: {{< math >}}$B=2${{< /math >}}

**Information available**:
{{< math >}}
$$N \times \log_2(B) = 1000 \times 1 = 1000 \text{ bits}$$
{{< /math >}}

**LoRA capacity** (rough estimate):
- Parameters: {{< math >}}$2rd = 65{,}000${{< /math >}}
- Effective bits per parameter: **5-8 (rough estimate)**
  - *Justification*: After training, parameters likely take on hundreds to thousands of distinguishable values that meaningfully affect behavior. This is much less than float32 precision (23 bits) but more than crude quantization (2-3 bits). The 5-8 bit range represents an educated guess rather than a measured quantity.
- Total capacity: {{< math >}}$65{,}000 \times 5${{< /math >}} to {{< math >}}$65{,}000 \times 8${{< /math >}} = **~300,000-500,000 bits**

**The ratio**: LoRA provides **~300-500× more capacity** than policy gradient's 1000-bit ceiling.

**Interpretation**: Even if our "bits per parameter" estimate is off by 2-3×, the qualitative conclusion holds: LoRA has substantial excess capacity relative to policy gradient's information ceiling. This explains why LoRA works well despite its parameter efficiency.

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

**Terminology clarification**: We use "fundamental" in two senses:

1. **Information-theoretic fundamentals**: The bounds in Theorems 1 and 2 are fundamental in that they cannot be exceeded by any algorithm using those signal types, regardless of computational resources

2. **Fundamental to TD learning**: Correlation between TD errors appears fundamental to bootstrap methods specifically—using {{< math >}}$V(s')${{< /math >}} to estimate the target for {{< math >}}$V(s)${{< /math >}} creates structural dependencies. This may not be fundamental to RL in general (e.g., Monte Carlo methods avoid this)

**Key open questions**:
- How much of the gap between the theoretical ceiling and achievable performance is fundamental (inherent correlation) vs algorithmic (poor critics)?
- Can methods substantially reduce TD error correlation?
- Are there alternative formulations that avoid bootstrap correlation?

**Research directions**:
1. **Stable critic training** at LLM scale (low-rank value architectures, ensemble methods)
2. **Decorrelation techniques** (eligibility traces, multi-step returns, though these may have fundamental limits)
3. **Dense reward engineering** (process rewards, per-token feedback)
4. **Non-bootstrap alternatives** (Monte Carlo methods, model-based approaches)

**What's not needed**:
- More parameters (LoRA already has 300× excess capacity relative to policy gradient's ceiling)

---

## Limitations and Future Work

**Theoretical limitations**:
1. Our bounds assume deterministic optimal policies (A1), which may not hold exactly in stochastic or degenerate settings
2. Assumption A2' (effective TD resolution) is not empirically validated—the choice of {{< math >}}$B_\delta = 256${{< /math >}} is illustrative
3. We model algorithms using idealized Bayesian inference, which doesn't capture actual optimization dynamics

**Empirical gaps**:
1. The achievable information bandwidth for actor-critic methods remains unmeasured
2. We don't empirically validate the correlation hypothesis or quantify its contribution
3. The practical gap between theoretical ceilings and actual performance needs experimental investigation

**Future work could**:
- Develop methods to directly measure information bandwidth in trained models
- Empirically quantify TD error correlation across different architectures and tasks
- Test whether decorrelation techniques (e.g., eligibility traces) improve information utilization
- Explore whether the theory extends to other RL settings (model-based, offline, multi-agent)
- Investigate the relationship between value function approximation quality and achievable information bandwidth

---

## Conclusion

This information-theoretic analysis reveals fundamental structure in RL for LLMs:

**The 1-bit bottleneck**: Policy gradient compresses rich token-level dynamics into scalar returns, creating a {{< math >}}$\leq \log_2(B)${{< /math >}} bits/episode ceiling. This explains:
- LoRA's success (capacity naturally matches ceiling)
- Sample inefficiency (1000s of episodes needed)
- Why more parameters don't help (bottleneck is signal sparsity)

**The dense signal bound**: By using a critic to **bootstrap historical information** into dense, token-level signals, actor-critic methods have a theoretical ceiling of {{< math >}}$\leq T \log_2(B_\delta)${{< /math >}} bits/episode under independence assumptions—up to 8000× higher than policy gradient. However, bootstrap correlation makes this ceiling likely unachievable in practice, with the actual achievable information bandwidth remaining an open empirical question.

**The path forward**: Future work should focus on empirically quantifying the achievable information bandwidth and understanding how much of the gap to the theoretical ceiling is fundamental versus algorithmic. Progress requires either:
1. **Better utilizing existing signals**: Improving critic training and exploring decorrelation techniques
2. **Circumventing bootstrap correlation**: Exploring Monte Carlo methods, model-based RL, or hybrid approaches that avoid systematic value function reuse
3. **Denser ground truth**: Engineering process rewards or per-token feedback that increases {{< math >}}$B${{< /math >}} directly

Understanding these tradeoffs—between signal density, correlation, and approximation quality—is essential for the next generation of LLM fine-tuning methods.

---

## References

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
