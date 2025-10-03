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
projects: []
---

## Understanding Sample Efficiency Through Signal Density

When I first read the "[LoRA Without Regret](https://thinkingmachines.ai/blog/lora/)" blog post, one claim caught my attention: policy gradient algorithms learn roughly **1 bit of information per episode**. This insight elegantly explains why LoRA—with its mere thousands of trainable parameters—works so remarkably well for RL fine-tuning of large language models.

But what does this actually mean? And if policy gradients learn so little per episode, how much do other RL algorithms learn? In this post, I'll work through an information-theoretic framework to answer these questions rigorously.

---

## TL;DR: The Main Results

**Policy gradient's hard limit**: The REINFORCE gradient $g = \nabla \log p_\theta(\tau) \cdot \text{Adv}$ has a structure where, given the training history, the direction term $\nabla \log p_\theta(\tau)$ is independent of the reward function—only the scalar advantage carries reward information. This creates an information ceiling of $\leq \log_2(B)$ bits per episode. For binary feedback, this is $\leq 1$ bit/episode.

**Actor-critic's reward-dependent bound**: Actor-critic methods preserve temporal structure through TD errors $\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$. Since this transformation is invertible, the information ceiling equals the reward entropy: $I(g; \pi^\star \mid \mathcal{H}) \leq H(\mathbf{r} \mid \tau, \mathcal{H})$. For terminal rewards only, this reduces to $\leq 1$ bit (same as policy gradient). For dense, independent rewards with $T=1000$ timesteps, this can reach $\leq 1580$ bits/episode.

**Why this matters**: 
- Policy gradient loses information by aggregating $T$ rewards into one scalar
- Actor-critic preserves information by using rewards separately at each timestep
- The advantage depends critically on reward structure (terminal vs. dense, independent vs. correlated)

| Algorithm | Reward Structure | Information Upper Bound |
|-----------|-----------------|---------------------|
| Policy Gradient | Terminal only ($B=2$) | ≤ 1 bit/episode |
| Actor-Critic | Terminal only ($B_r=2$) | ≤ 1 bit/episode |
| Policy Gradient | Dense ($T=1000$, $B_r=3$) | ≤ 8 bits/episode |
| Actor-Critic | Dense independent ($T=1000$, $B_r=3$) | ≤ 1585 bits/episode |

The remainder of this post derives these bounds rigorously and explores their implications.

---

## Part 1: The Mathematical Framework

### Setup: Language Model Fine-Tuning as an MDP

When you fine-tune an LLM with RL, you're working with a special kind of problem:

- **States** $s$: Token sequences $(x_1, \ldots, x_t)$
- **Actions** $a$: Next token $x_{t+1}$ from vocabulary
- **Transitions**: Deterministic (append token: $s' = s \circ a$)
- **Rewards** $R_\xi$: Determined by unknown parameter $\xi$ (preferences, objectives)

**Key property**: Transitions are known and deterministic. All uncertainty is in the reward function $\xi$.

### Information-Theoretic Lens

To analyze this rigorously, we need a mathematical trick. Put a prior $p(\xi)$ over reward parameters—this induces a distribution $p(\pi^\star)$ over optimal policies, where each $\xi$ determines a unique optimal policy $\pi^\star_\xi$.

We don't claim algorithms actually maintain Bayesian posteriors. This is just a modeling device that lets us reason about information flow: it makes both the learning signal (the gradient) and the optimal policy $\pi^\star$ well-defined random variables, so we can compute their mutual information.

**Definition (Information Bandwidth)**:

The information bandwidth measures the maximum information that can be gained per episode:

$$\mathcal{B} = \sup_{\mathcal{H}} I(g; \pi^\star \mid \mathcal{H})$$

where $g$ is the gradient from a single episode and $\mathcal{H}$ is the history of all previous episodes.

**Interpretation**: $I(g; \pi^\star \mid \mathcal{H})$ asks "how much does this gradient tell me about the optimal policy, given what I already know?" The **bandwidth** $\mathcal{B}$ is the maximum—the algorithm's capacity limit regardless of training progress. In practice, history gets compressed into parameters $\theta$, but conditioning on the complete history $\mathcal{H}$ makes the information-theoretic interpretation most transparent: it directly captures "what we've already learned" from all past episodes.

### Connection to the Original Insight

This formalization directly implements the information-theoretic argument from "[LoRA Without Regret](https://thinkingmachines.ai/blog/lora/)." Their key observation: for the REINFORCE gradient $g = \nabla \log p_\theta(\tau) \cdot \text{Adv}$, the component $\nabla \log p_\theta(\tau)$ is independent of the reward function $R$ given the history $\mathcal{H}$ (which determines the policy), so **all information about $R$ must flow through the scalar advantage**.

We extend this to show what's theoretically possible with denser signals and why current practice makes sense.

### Two Minimal Assumptions

**Assumption A1 (Unique Optimum)**: Each $\xi$ determines a unique optimal policy $\pi^\star_\xi$.

*Justification*: Generic for neural networks with many parameters. Floating-point precision breaks ties; exact degeneracy is measure-zero.

**Assumption A2 (Finite Resolution)**: The reward-dependent scalar(s) in the gradient have finite effective resolution—they can take at most $B$ distinguishable values.

*Justification*: Holds exactly for binary preferences ($B=2$) or Likert scales ($B=4$-$7$). Approximately true for continuous signals with noise, finite precision, or practical distinguishability limits.

---

## Part 2: Policy Gradient's Information Ceiling

### The Algorithm

Policy gradient (REINFORCE) works like this:

1. Sample trajectory $\tau = (s_0, a_0, \ldots, s_{T-1}, a_{T-1}, s_T)$ using policy $\pi_\theta$
2. Observe rewards $\mathbf{r} = (r_0, r_1, \ldots, r_{T-1})$ where $r_t = R_\xi(s_t, a_t)$
3. Compute **return**: $G = \sum_{t=0}^{T-1} \gamma^t r_t$
4. Compute **advantage**: $\text{Adv} = G - b$ (where $b$ is a baseline)
5. Compute **gradient**: $g = \nabla \log p_\theta(\tau) \cdot \text{Adv}$
6. Update: $\theta \leftarrow \theta + \alpha g$

**Key observation**: The gradient has two components:
- $\nabla \log p_\theta(\tau)$: Independent of the reward function $R$ given the history $\mathcal{H}$
- $\text{Adv}$: A scalar that encodes all reward-dependent information

### The Information Ceiling

**Theorem 1 (Policy Gradient Information Ceiling):**

*Under assumptions A1 and A2, the information bandwidth of policy gradient satisfies:*

$$I(g; \pi^\star \mid \mathcal{H}) \leq \log_2(B) \text{ bits per episode}$$

where $g = \nabla \log p_\theta(\tau) \cdot \text{Adv}$ is the REINFORCE gradient and $\mathcal{H}$ is the history of all previous episodes.

**Intuition**: Given the history $\mathcal{H}$ (which determines the current policy $\theta$), the trajectory sampling doesn't depend on $\xi$. All new information must flow through the scalar advantage, creating a hard ceiling of $\log_2(B)$ bits.

<details>
<summary><strong>Detailed Proof (click to expand)</strong></summary>

**Step 1: Information Chain**

By data processing inequality, since $g$ is a deterministic function of $(\tau, \text{Adv})$ given $\mathcal{H}$:
$$I(g; \pi^\star \mid \mathcal{H}) \leq I((\tau, \text{Adv}); \pi^\star \mid \mathcal{H})$$

By chain rule for mutual information:
$$I((\tau, \text{Adv}); \pi^\star \mid \mathcal{H}) = I(\tau; \pi^\star \mid \mathcal{H}) + I(\text{Adv}; \pi^\star \mid \tau, \mathcal{H})$$

---

**Step 2: Trajectory Contains No Information About $\xi$**

Given $\mathcal{H}$, the policy parameters $\theta = \theta(\mathcal{H})$ are deterministic. The trajectory is sampled from:
$$\tau \sim p_\theta(\tau)$$

This distribution depends only on $\theta$ (and the known, deterministic environment dynamics), not on the reward parameter $\xi$.

Therefore: $\tau \perp \xi \mid \mathcal{H}$

Since $\xi$ determines $\pi^\star$ deterministically by Assumption A1, we also have $\tau \perp \pi^\star \mid \mathcal{H}$.

Therefore:
$$I(\tau; \pi^\star \mid \mathcal{H}) = 0$$

---

**Step 3: All Information Flows Through Advantage**

From Steps 1-2:
$$I(g; \pi^\star \mid \mathcal{H}) \leq I(\text{Adv}; \pi^\star \mid \tau, \mathcal{H})$$

---

**Step 4: Bound Advantage Information**

By the fundamental bound on mutual information:
$$I(\text{Adv}; \pi^\star \mid \tau, \mathcal{H}) \leq H(\text{Adv} \mid \tau, \mathcal{H})$$

---

**Step 5: Apply Finite Resolution Assumption**

By Assumption A2, the advantage takes at most $B$ distinct values. Therefore:
$$H(\text{Adv} \mid \tau, \mathcal{H}) \leq \log_2(B)$$

This holds because entropy is maximized when uniform: $H(X) \leq \log_2(|X|)$

---

**Step 6: Connect to Optimal Policy**

By Assumption A1, $\pi^\star = f(\xi)$ deterministically. This creates:
$$\text{Adv} \to \xi \to \pi^\star$$

By data processing:
$$I(\text{Adv}; \pi^\star \mid \tau, \mathcal{H}) \leq I(\text{Adv}; \xi \mid \tau, \mathcal{H})$$

---

**Final Result**

Combining all steps:
$$I(g; \pi^\star \mid \mathcal{H}) \leq H(\text{Adv} \mid \tau, \mathcal{H}) \leq \log_2(B)$$

∎

</details>

This ceiling holds regardless of sequence length $T$, model size, or computational budget—it's an inherent consequence of the scalar advantage bottleneck.

### Where Does Information Get Lost?

The bound $\leq \log_2(B)$ is tight for some reward structures but loose for others:

$$\mathbf{r} \xrightarrow{\text{sum}} G \xrightarrow{\text{subtract baseline}} \text{Adv} \xrightarrow{\text{finite resolution}} \text{bounded by } B$$

#### Case 1: Terminal Reward Only

**Setup**: $r_t = 0$ for $t < T-1$, only $r_{T-1} \in \{-1, +1\}$ is non-zero

**Return**: $G = \gamma^{T-1} r_{T-1}$

**Advantage**: $\text{Adv} = \gamma^{T-1} r_{T-1} - b$

Since the mapping $r_{T-1} \leftrightarrow \text{Adv}$ is bijective:
$$H(\text{Adv} \mid \tau, \mathcal{H}) = H(r_{T-1} \mid \tau, \mathcal{H}) = 1 \text{ bit}$$

**No information loss** from aggregation. The bound is **tight**.

---

#### Case 2: Dense Independent Rewards

**Setup**: $r_t \in \{-1, 0, +1\}$ at each timestep, with factorized $\xi = (\xi_0, \xi_1, \ldots, \xi_{T-1})$ where each $\xi_t$ is independent

**Available information**:
$$H(\mathbf{r} \mid \tau, \mathcal{H}) = \sum_{t=0}^{T-1} H(r_t \mid \tau, \mathcal{H}) = T \log_2(3) \approx 1.58T \text{ bits}$$

For $T = 1000$: approximately 1585 bits available.

**Return**: $G = \sum_{t=0}^{T-1} \gamma^t r_t$

This is a **many-to-one mapping**. Different temporal patterns give the same sum:
- $(+1, -1, +1, -1, \ldots)$ → sum ≈ 0
- $(-1, +1, -1, +1, \ldots)$ → sum ≈ 0
- $(0, 0, 0, \ldots, 0)$ → sum = 0

For $\gamma = 1$, the return $G \in \{-1000, -999, \ldots, 999, 1000\}$ has:
$$H(G \mid \tau, \mathcal{H}) \leq \log_2(2001) \approx 11 \text{ bits}$$

**Advantage**: With noise and finite precision (Assumption A2 with $B = 256$):
$$H(\text{Adv} \mid \tau, \mathcal{H}) \leq \log_2(256) = 8 \text{ bits}$$

**Information loss**: Starting with 1585 bits, policy gradient retains only ~8 bits.

**Loss factor**: ~200× reduction

**Why?** The summation loses temporal structure. The algorithm can't distinguish whether early tokens or late tokens were good.

---

### Concrete Examples

- **Binary preferences** ($B=2$): $\leq 1$ bit/episode
- **Likert scale** ($B=5$): $\leq 2.3$ bits/episode
- **8-bit resolution** ($B=256$): $\leq 8$ bits/episode

### Implications

A typical LLM generation has $T \sim 1000$ tokens. The REINFORCE gradient compresses all this rich structure into **one scalar** (the advantage) that modulates a fixed direction $\nabla \log p_\theta(\tau)$.

This explains why:
- **Training needs thousands of episodes**: With 1 bit/episode and binary feedback, accumulating meaningful information takes many episodes
- **Adding parameters doesn't help**: The bottleneck is the scalar advantage, not model capacity

The policy update $\Delta \theta = \alpha g$ happens in a high-dimensional space, but the **information content** of this update about the optimal policy is limited by the scalar advantage.

---

## Part 3: Actor-Critic's Information Ceiling

### The Algorithm

Actor-critic methods (A3C, PPO with value function) maintain two components:

- **Actor** $\pi_\theta(a|s)$: The policy with parameters $\theta$
- **Critic** $V_\phi(s)$: Value function estimating expected return from state $s$, with parameters $\phi$

**Training loop**: For each episode:

1. **Rollout**: Generate trajectory $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots, s_T)$ using current policy $\pi_\theta$

2. **Compute TD errors** at each timestep $t$:
   $$\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$$

3. **Update critic** toward the TD target:
   $$\phi \leftarrow \phi + \alpha_\phi \cdot \delta_t \cdot \nabla_\phi V_\phi(s_t)$$

4. **Update actor** using the gradient:
   $$\theta \leftarrow \theta + \alpha_\theta \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \delta_t$$

**Learning signal**: $g = \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \delta_t$

**Key difference from policy gradient**: Instead of one scalar advantage per episode, we get $T$ TD errors—one per timestep.

### Where Does Actor-Critic's Advantage Come From?

"The environment only provides rewards at certain steps. How can actor-critic learn more per episode than policy gradient?"

The answer is **not** that actor-critic gets more information from the environment—the total information available is the same. Instead, actor-critic **preserves information** that policy gradient destroys.

**Policy Gradient** aggregates all rewards into one scalar:
$$\text{Adv} = \sum_{t=0}^{T-1} \gamma^t r_t - b$$

This is a **many-to-one mapping**: different temporal patterns can give the same advantage. For $T=1000$ tokens with $r_t \in \{-1, 0, +1\}$:
- Total available information: $H(\mathbf{r}) = 1000 \times \log_2(3) \approx 1585$ bits
- After summation: $H(\text{Adv}) \leq 8$ bits
- **Information lost**: ~99%

**Actor-Critic** transforms rewards into TD errors:
$$\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$$

This is a **bijection** (given $\tau, \mathcal{H}$): 
$$\mathbf{r} = \boldsymbol{\delta} - \mathbf{c}$$
where $\mathbf{c} = (\gamma V_\phi(s_{t+1}) - V_\phi(s_t))$ is deterministic.

You can recover the exact reward sequence from TD errors. **No information is lost**: $H(\boldsymbol{\delta}) = H(\mathbf{r})$.

**What does the critic do?**

The critic $V_\phi$, learned from past episodes stored in $\mathcal{H}$, provides state-dependent baselines that enable:
- **Credit assignment**: Each timestep gets feedback relative to learned expectations for that state
- **Variance reduction**: Baselines adapt to state values, reducing gradient variance
- **Temporal preservation**: Each reward $r_t$ primarily affects its corresponding $\delta_t$

But the critic **doesn't create new information**—it redistributes the information already present in the current episode's rewards across $T$ timesteps instead of collapsing it into one scalar.

### The Information Ceiling

**Theorem 2 (Actor-Critic Information Ceiling):**

*Under assumptions A1 and A2 (applied to rewards with resolution $B_r$), actor-critic's information bandwidth satisfies:*

$$I(g; \pi^\star \mid \mathcal{H}) \leq H(\mathbf{r} \mid \tau, \mathcal{H})$$

where $\mathbf{r} = (r_0, \ldots, r_{T-1})$ is the reward sequence.

**Special cases**:

1. **Terminal reward only**: $H(\mathbf{r} \mid \tau, \mathcal{H}) = H(r_{T-1} \mid \tau, \mathcal{H}) \leq \log_2(B_r)$

2. **Independent rewards** (factorized $\xi$): $H(\mathbf{r} \mid \tau, \mathcal{H}) = \sum_t H(r_t \mid \tau, \mathcal{H}) \leq T \log_2(B_r)$

3. **General case**: $\log_2(B_r) \leq H(\mathbf{r} \mid \tau, \mathcal{H}) \leq T \log_2(B_r)$

**Key insight**: The bound depends on **reward entropy**, not on TD error structure. The critic redistributes information temporally but doesn't create new information.

<details>
<summary><strong>Complete Rigorous Proof (click to expand)</strong></summary>

**Step 1: From Gradient to TD Errors**

The actor gradient is:
$$g = \sum_{t=0}^{T-1} \nabla \log \pi_\theta(a_t|s_t) \cdot \delta_t$$

Given $(\tau, \mathcal{H})$:
- The policy $\theta = \theta(\mathcal{H})$ is deterministic
- All states and actions in $\tau$ are known
- The gradient is a deterministic function of $(\tau, \boldsymbol{\delta}, \mathcal{H})$ where $\boldsymbol{\delta} = (\delta_0, \ldots, \delta_{T-1})$

By data processing:
$$I(g; \pi^\star \mid \mathcal{H}) \leq I((\tau, \boldsymbol{\delta}); \pi^\star \mid \mathcal{H})$$

By chain rule:
$$I((\tau, \boldsymbol{\delta}); \pi^\star \mid \mathcal{H}) = I(\tau; \pi^\star \mid \mathcal{H}) + I(\boldsymbol{\delta}; \pi^\star \mid \tau, \mathcal{H})$$

As in policy gradient, $\tau \perp \xi \mid \mathcal{H}$, so:
$$I(\tau; \pi^\star \mid \mathcal{H}) = 0$$

Therefore:
$$I(g; \pi^\star \mid \mathcal{H}) \leq I(\boldsymbol{\delta}; \pi^\star \mid \tau, \mathcal{H})$$

---

**Step 2: From TD Errors to Rewards (The Key Step)**

The TD error at timestep $t$ is:
$$\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$$

Given $(\tau, \mathcal{H})$:
- The critic $\phi = \phi(\mathcal{H})$ is deterministic
- All states $s_0, \ldots, s_T$ are known (determined by $\tau$)
- Therefore $V_\phi(s_t)$ and $V_\phi(s_{t+1})$ are deterministic

So each TD error can be written as:
$$\delta_t = r_t + c_t$$

where $c_t = \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$ is deterministic given $(\tau, \mathcal{H})$.

**The TD error vector is an affine transformation of the reward vector**:
$$\boldsymbol{\delta} = \mathbf{r} + \mathbf{c}$$

where $\mathbf{c} = (c_0, \ldots, c_{T-1})$ is deterministic given $(\tau, \mathcal{H})$.

**This transformation is invertible**: Given $\boldsymbol{\delta}$ and $\mathbf{c}$ (which is known from $\tau, \mathcal{H}$), we can recover:
$$\mathbf{r} = \boldsymbol{\delta} - \mathbf{c}$$

Since the transformation is a bijection with deterministic offset:
$$H(\boldsymbol{\delta} \mid \tau, \mathcal{H}) = H(\mathbf{r} \mid \tau, \mathcal{H})$$

And:
$$I(\boldsymbol{\delta}; \pi^\star \mid \tau, \mathcal{H}) = I(\mathbf{r}; \pi^\star \mid \tau, \mathcal{H})$$

**No information loss** in the TD transformation!

---

**Step 3: From Rewards to $\xi$**

By Assumption A1, $\pi^\star = f(\xi)$ deterministically. This gives:
$$\mathbf{r} \to \xi \to \pi^\star$$

(conditioned on $\tau, \mathcal{H}$)

By data processing:
$$I(\mathbf{r}; \pi^\star \mid \tau, \mathcal{H}) \leq I(\mathbf{r}; \xi \mid \tau, \mathcal{H}) \leq H(\mathbf{r} \mid \tau, \mathcal{H})$$

---

**Final Result**

Combining Steps 1-3:
$$I(g; \pi^\star \mid \mathcal{H}) \leq H(\mathbf{r} \mid \tau, \mathcal{H})$$

∎

</details>

### Analyzing the Reward Entropy

The bound $H(\mathbf{r} \mid \tau, \mathcal{H})$ depends on the **structure of the reward function** $R_\xi$.

Using the chain rule:
$$H(\mathbf{r} \mid \tau, \mathcal{H}) = \sum_{t=0}^{T-1} H(r_t \mid r_{<t}, \tau, \mathcal{H})$$

where $r_{<t} = (r_0, \ldots, r_{t-1})$.

Each term satisfies:
$$H(r_t \mid r_{<t}, \tau, \mathcal{H}) \leq \log_2(B_r)$$

The sum satisfies:
$$H(\mathbf{r} \mid \tau, \mathcal{H}) \leq T \log_2(B_r)$$

**Equality holds** if and only if rewards are informationally independent: $r_t \perp r_{<t} \mid (\tau, \mathcal{H})$ for all $t$.

This requires **factorized $\xi$**: each state-action pair has its own independent reward parameter.

### Case Analysis

#### Case 1: Terminal Reward Only

**Setup**: Only $r_{T-1} \neq 0$, all other rewards are zero

**Reward entropy**:
$$H(\mathbf{r} \mid \tau, \mathcal{H}) = H(r_{T-1} \mid \tau, \mathcal{H}) \leq \log_2(B_r)$$

For binary rewards: $\leq 1$ bit

**TD errors**: For $t < T-1$:
$$\delta_t = 0 + \gamma V_\phi(s_{t+1}) - V_\phi(s_t) \quad \text{(deterministic given } \tau, \mathcal{H})$$

Only $\delta_{T-1} = r_{T-1} - V_\phi(s_{T-1})$ is random.

**Information ceiling**: $I(g; \pi^\star \mid \mathcal{H}) \leq 1$ bit

**Conclusion**: Same as policy gradient! When there's only terminal reward, actor-critic has no information advantage.

---

#### Case 2: Dense Independent Rewards

**Setup**: Factorized reward parameter $\xi = (\xi_0, \xi_1, \ldots, \xi_{T-1})$ where each $\xi_t$ is independent, and $r_t = R_{\xi_t}(s_t, a_t)$

**This means**: Observing $r_0$ reveals information about $\xi_0$ but nothing about $\xi_1, \ldots, \xi_{T-1}$.

**Reward entropy**:
$$H(\mathbf{r} \mid \tau, \mathcal{H}) = \sum_{t=0}^{T-1} H(r_t \mid \tau, \mathcal{H}) = T \log_2(B_r)$$

**Example** ($T=1000$, $r_t \in \{-1, 0, +1\}$ so $B_r=3$):
$$H(\mathbf{r} \mid \tau, \mathcal{H}) = 1000 \times \log_2(3) \approx 1585 \text{ bits}$$

**Actor-critic information ceiling**: Up to 1585 bits/episode

**Policy gradient information ceiling**: $\leq 8$ bits/episode (from aggregating into scalar advantage)

**Advantage**: $\sim 200\times$ more information preserved!

**Why the difference?**
- Policy gradient: $\mathbf{r} \to \sum_t \gamma^t r_t$ loses temporal structure
- Actor-critic: $\mathbf{r} \to \boldsymbol{\delta}$ preserves all information

---

### Summary

| Algorithm | Reward Structure | Information Ceiling | Why |
|-----------|-----------------|---------------------|-----|
| Policy Gradient | Terminal only | ≤ 1 bit | Scalar advantage with binary reward |
| Actor-Critic | Terminal only | ≤ 1 bit | Only one TD error is random |
| Policy Gradient | Dense independent | ≤ 8 bits | Aggregation loses temporal structure |
| Actor-Critic | Dense independent | ≤ 1585 bits | Temporal structure preserved |

---

## Part 4: Implications

### Two Types of Barriers

Our analysis reveals two distinct types of barriers:

**1. Information-theoretic ceilings** (fundamental, cannot be exceeded):
- Policy gradient: $\leq \log_2(B)$ bits/episode—structural limit from scalar advantage
- Actor-critic: $\leq H(\mathbf{r} \mid \tau, \mathcal{H})$—depends on reward structure

**2. Practical implementation challenges**:
- Terminal vs. dense rewards (data collection and design)
- Reward correlation structure (inherent to many tasks)
- Optimization efficiency (value function approximation, gradient descent)
- Training stability (hyperparameters, learning dynamics)

### Why Current Practice Makes Sense

**Most LLM-RL uses terminal rewards**: With $H(\mathbf{r}) = 1$ bit, actor-critic has no theoretical advantage over policy gradient. This explains why simple policy gradient methods dominate—the complexity of actor-critic isn't justified when the information ceiling is the same.

**Actor-critic's advantage requires dense rewards**: Only with independent or weakly correlated rewards at each timestep does actor-critic's ability to preserve temporal structure matter. But collecting dense, high-quality rewards is expensive.

**Signal density matters more than capacity**: The information bottleneck (1 bit/episode for binary preferences) suggests that for current methods, the fundamental constraint is signal density rather than model capacity. This helps explain why low-rank adaptation methods can be effective for RL fine-tuning.

### Open Questions

Several important questions remain:

1. **Reward correlation in practice**: How much correlation exists in realistic reward functions? This determines the gap between $\log_2(B_r)$ and $T \log_2(B_r)$.

2. **Optimization efficiency**: What fraction of the theoretical ceiling $H(\mathbf{r})$ do practical algorithms extract? This requires empirical measurement.

3. **Alternative paradigms**: Can we design algorithms that circumvent these limitations? Model-based methods, Monte Carlo approaches, or hybrid algorithms may offer different trade-offs.

---

## Limitations

**Theoretical limitations**:
1. Our bounds assume deterministic optimal policies (A1), which may not hold exactly in stochastic settings
2. We model algorithms using idealized Bayesian inference, which doesn't capture actual optimization dynamics
3. The finite resolution assumption (A2) is approximate for continuous rewards

**Empirical gaps**:
1. We do not empirically measure information gain per episode in real training runs
2. The relationship between reward correlation structure and practical performance is not validated
3. The actual effective resolution for advantages and TD errors is task-dependent and unmeasured

**Future work should**:
- Measure information flow empirically: estimate $H(\mathbf{r} \mid \tau, \mathcal{H})$ in real tasks
- Test whether sample efficiency scales with signal density as predicted
- Explore whether the theory extends to other RL settings (model-based, offline RL, multi-agent)

---

## References

**LLM Fine-Tuning**:
- Ouyang, L., et al. (2022). "Training language models to follow instructions with human feedback." *NeurIPS*. (InstructGPT)
- Hu, E. J., et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." *ICLR*.

**Reinforcement Learning**:
- Sutton, R. S., McAllester, D., Singh, S., & Mansour, Y. (1999). "Policy gradient methods for reinforcement learning with function approximation." *NIPS*, 12.
- Konda, V., & Tsitsiklis, J. (1999). "Actor-critic algorithms." *NIPS*, 12.
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