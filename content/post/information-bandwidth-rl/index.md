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

**Policy gradient's hard limit**: The REINFORCE gradient $G = \nabla \log p_\theta(\tau) \cdot \text{Adv}$ has a structure where, given the training history, the direction term $\nabla \log p_\theta(\tau)$ is independent of the reward function—only the scalar advantage carries reward information. This creates an information ceiling of $\leq \log_2(B)$ bits per episode. For binary feedback, this is $\leq 1$ bit/episode—explaining why training needs thousands of episodes and why LoRA's modest capacity (300-500× excess) suffices.

**Actor-critic's reward-dependent bound**: Actor-critic methods preserve temporal structure through TD errors $\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$. Since this transformation is invertible, the information ceiling equals the reward entropy: $I(G; \pi^\star \mid \mathcal{H}) \leq H(\mathbf{r} \mid \tau, \mathcal{H})$. For terminal rewards only, this reduces to $\leq 1$ bit (same as policy gradient). For dense, independent rewards with $T=1000$ timesteps, this can reach $\leq 1580$ bits/episode—but practical correlation reduces this to 10-100 bits.

**Why this matters**: 
- Policy gradient loses information by aggregating $T$ rewards into one scalar
- Actor-critic preserves information by using rewards separately at each timestep
- The advantage depends critically on reward structure (terminal vs. dense, correlated vs. independent)
- LoRA provides 300-500× excess capacity for current methods and remains sufficient even for improved actor-critic

| Algorithm | Reward Structure | Information Upper Bound | Practical Achievable |
|-----------|-----------------|---------------------|---------------------|
| Policy Gradient | Terminal only ($B=2$) | ≤ 1 bit/episode | ~1 bit/episode |
| Actor-Critic | Terminal only ($B_r=2$) | ≤ 1 bit/episode | ~1 bit/episode |
| Policy Gradient | Dense ($T=1000$, $B_r=3$) | ≤ 8 bits/episode | ~8 bits/episode |
| Actor-Critic | Dense independent ($T=1000$, $B_r=3$) | ≤ 1580 bits/episode | ~100-300 bits/episode |

The remainder of this post derives these bounds rigorously and explores their implications for algorithm design and parameter efficiency.

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

$$\mathcal{B} = \sup_{\mathcal{H}} I(G; \pi^\star \mid \mathcal{H})$$

where $G$ is the gradient from a single episode and $\mathcal{H}$ is the history of all previous episodes.

**Interpretation**: $I(G; \pi^\star \mid \mathcal{H})$ asks "how much does this gradient tell me about the optimal policy, given what I already know?" Early in training, each gradient is informative. Late in training, we've learned most patterns, so each gradient adds less. The **bandwidth** $\mathcal{B}$ is the maximum—the algorithm's capacity limit regardless of training progress.

Our goal is to find upper bounds on $\mathcal{B}$ that depend only on the **structure of the learning signal** (e.g., whether it's a scalar or a vector of scalars), not on the training state. If we can show that $I(G; \pi^\star \mid \mathcal{H}) \leq C$ for **every** possible history $\mathcal{H}$, then:

$$\mathcal{B} = \sup_{\mathcal{H}} I(G; \pi^\star \mid \mathcal{H}) \leq C$$

This bound $C$ characterizes the algorithm's inherent information capacity.

**Note on $\theta$ and history**: In practice, history gets compressed into parameters $\theta$. Our framework conditions on complete history for mathematical cleanliness, but the bounds apply either way.

### Connection to the Original Insight

This formalization directly implements the information-theoretic argument from "[LoRA Without Regret](https://thinkingmachines.ai/blog/lora/)." Their key observation: for the REINFORCE gradient $G = \nabla \log p_\theta(\tau) \cdot \text{Adv}$, the component $\nabla \log p_\theta(\tau)$ is independent of the reward function $R$ given the history $\mathcal{H}$ (which determines the policy), so **all information about $R$ must flow through the scalar advantage**.

We're extending this in three ways. First, we measure information about the optimal policy directly—asking "what do we learn about $\pi^\star$?" rather than "what do we learn about $R$?". Second, we're explicit about history: the conditioning on $\mathcal{H}$ makes "what we already know" mathematically precise. Third, we formalize when the advantage has bounded entropy (Assumption A2).

Since the optimal policy is a deterministic function of the reward parameters ($\pi^\star = f(\xi)$), learning about $\xi$ through the gradient is equivalent to learning about $\pi^\star$. The core insight remains: **scalar feedback creates an information bottleneck**. We extend this to show what's theoretically possible with denser signals and why current practice makes sense.

### Two Minimal Assumptions

**Assumption A1 (Unique Optimum)**: Each $\xi$ determines a unique optimal policy $\pi^\star_\xi$.

*Justification*: Generic for neural networks with many parameters. Floating-point precision breaks ties; exact degeneracy is measure-zero.

**Assumption A2 (Finite Resolution)**: The reward-dependent scalar(s) in the gradient have finite effective resolution—they can take at most $B$ distinguishable values.

*Justification*: Holds exactly for binary preferences ($B=2$) or Likert scales ($B=4$-$7$). Approximately true for continuous signals with noise, finite precision, or practical distinguishability limits.

---

## Part 2: Policy Gradient's 1-Bit Ceiling

### The Algorithm

Policy gradient (REINFORCE) works like this:

1. Sample trajectory $\tau = (s_0, a_0, \ldots, s_{T-1}, a_{T-1}, s_T)$ using policy $\pi_\theta$
2. Observe rewards $\mathbf{r} = (r_0, r_1, \ldots, r_{T-1})$ where $r_t = R_\xi(s_t, a_t)$
3. Compute **return**: $G_\tau = \sum_{t=0}^{T-1} \gamma^t r_t$
4. Compute **advantage**: $\text{Adv} = G_\tau - b$ (where $b$ is a baseline)
5. Compute **gradient**: $G = \nabla \log p_\theta(\tau) \cdot \text{Adv}$
6. Update: $\theta \leftarrow \theta + \alpha G$

**Learning signal**: $G = \nabla \log p_\theta(\tau) \cdot \text{Adv}$ (the gradient)

**Key observation**: The gradient has two components:
- $\nabla \log p_\theta(\tau)$: Independent of the reward function $R$ given the history $\mathcal{H}$ (which determines the current policy $\theta$)
- $\text{Adv}$: A scalar that encodes all reward-dependent information

### The Information Ceiling

**Theorem 1 (Policy Gradient Information Ceiling):**

*Under assumptions A1 and A2, the information bandwidth of policy gradient satisfies:*

$$I(G; \pi^\star \mid \mathcal{H}) \leq \log_2(B) \text{ bits per episode}$$

where $G = \nabla \log p_\theta(\tau) \cdot \text{Adv}$ is the REINFORCE gradient and $\mathcal{H}$ is the history of all previous episodes.

**Intuition**: Given the history $\mathcal{H}$ (which determines the current policy $\theta$), the trajectory sampling doesn't depend on $\xi$. All new information must flow through the scalar advantage, creating a hard ceiling of $\log_2(B)$ bits.

<details>
<summary><strong>Detailed Proof (click to expand)</strong></summary>

**Step 1: Information Chain**

By data processing inequality, since $G$ is a deterministic function of $(\tau, \text{Adv})$ given $\mathcal{H}$:
$$I(G; \pi^\star \mid \mathcal{H}) \leq I((\tau, \text{Adv}); \pi^\star \mid \mathcal{H})$$

By chain rule for mutual information:
$$I((\tau, \text{Adv}); \pi^\star \mid \mathcal{H}) = I(\tau; \pi^\star \mid \mathcal{H}) + I(\text{Adv}; \pi^\star \mid \tau, \mathcal{H})$$

---

**Step 2: Trajectory Contains No Information About $\xi$**

Given $\mathcal{H}$, the policy parameters $\theta = \theta(\mathcal{H})$ are deterministic. The trajectory is sampled from:
$$\tau \sim p_\theta(\tau)$$

This distribution depends only on $\theta$ (and the known, deterministic environment dynamics), not on the reward parameter $\xi$.

Therefore: $\tau \perp \xi \mid \mathcal{H}$

By the Markov chain $\tau \to \xi \to \pi^\star$ (conditioned on $\mathcal{H}$):
$$I(\tau; \pi^\star \mid \mathcal{H}) = 0$$

---

**Step 3: All Information Flows Through Advantage**

From Steps 1-2:
$$I(G; \pi^\star \mid \mathcal{H}) \leq I(\text{Adv}; \pi^\star \mid \tau, \mathcal{H})$$

---

**Step 4: Bound Advantage Information**

By the fundamental bound on mutual information:
$$I(\text{Adv}; \pi^\star \mid \tau, \mathcal{H}) \leq H(\text{Adv} \mid \tau, \mathcal{H})$$

---

**Step 5: Apply Finite Resolution Assumption**

By Assumption A2, the advantage takes at most $B$ distinct values. Therefore:
$$H(\text{Adv} \mid \tau, \mathcal{H}) \leq \log_2(B)$$

This holds because:
- Even conditioned on $\tau$ and $\mathcal{H}$, the advantage still has at most $B$ values in its support
- Entropy is maximized when uniform: $H(X) \leq \log_2(|X|)$

---

**Step 6: Connect to Optimal Policy**

By Assumption A1, $\pi^\star = f(\xi)$ deterministically. This creates:
$$\text{Adv} \to \xi \to \pi^\star$$

(The advantage depends on rewards which depend on $\xi$, and $\pi^\star$ depends only on $\xi$)

By data processing:
$$I(\text{Adv}; \pi^\star \mid \tau, \mathcal{H}) \leq I(\text{Adv}; \xi \mid \tau, \mathcal{H})$$

---

**Final Result**

Combining all steps:
$$I(G; \pi^\star \mid \mathcal{H}) \leq H(\text{Adv} \mid \tau, \mathcal{H}) \leq \log_2(B)$$

∎

</details>

This ceiling holds regardless of sequence length $T$, model size, or computational budget—it's an inherent consequence of the scalar advantage bottleneck.

### Where Does Information Get Lost?

The bound $\leq \log_2(B)$ is tight for some reward structures but loose for others. Let's trace where information is lost:

$$\mathbf{r} \xrightarrow{\text{sum}} G_\tau \xrightarrow{\text{subtract baseline}} \text{Adv} \xrightarrow{\text{finite resolution}} \text{bounded by } B$$

#### Case 1: Terminal Reward Only

**Setup**: $r_t = 0$ for $t < T-1$, only $r_{T-1} \in \{-1, +1\}$ is non-zero

**Return**: $G_\tau = \gamma^{T-1} r_{T-1}$

**Advantage**: $\text{Adv} = \gamma^{T-1} r_{T-1} - b$

Since the mapping $r_{T-1} \leftrightarrow \text{Adv}$ is bijective (one-to-one):
$$H(\text{Adv} \mid \tau, \mathcal{H}) = H(r_{T-1} \mid \tau, \mathcal{H}) = 1 \text{ bit}$$

**No information loss** from aggregation. The bound is **tight**.

---

#### Case 2: Dense Independent Rewards

**Setup**: $r_t \in \{-1, 0, +1\}$ at each timestep, with factorized $\xi = (\xi_0, \xi_1, \ldots, \xi_{T-1})$ where each $\xi_t$ is independent

**Available information**:
$$H(\mathbf{r} \mid \tau, \mathcal{H}) = \sum_{t=0}^{T-1} H(r_t \mid \tau, \mathcal{H}) = T \log_2(3) \approx 1.58T \text{ bits}$$

For $T = 1000$: approximately 1580 bits available.

**Return**: $G_\tau = \sum_{t=0}^{T-1} \gamma^t r_t$

This is a **many-to-one mapping**. Different temporal patterns give the same sum:
- $(+1, -1, +1, -1, \ldots)$ → sum ≈ 0
- $(-1, +1, -1, +1, \ldots)$ → sum ≈ 0
- $(0, 0, 0, \ldots, 0)$ → sum = 0

For $\gamma = 1$, the return $G_\tau \in \{-1000, -999, \ldots, 999, 1000\}$ has:
$$H(G_\tau \mid \tau, \mathcal{H}) \leq \log_2(2001) \approx 11 \text{ bits}$$

**Advantage**: With noise and finite precision (Assumption A2 with $B = 256$):
$$H(\text{Adv} \mid \tau, \mathcal{H}) \leq \log_2(256) = 8 \text{ bits}$$

**Information loss**: Starting with 1580 bits, policy gradient retains only ~8 bits!

**Loss factor**: ~200× reduction

**Why?** The summation loses temporal structure. The algorithm can't distinguish whether early tokens or late tokens were good.

---

### Concrete Examples

- **Binary preferences** ($B=2$): $\leq 1$ bit/episode
- **Likert scale** ($B=5$): $\leq 2.3$ bits/episode
- **8-bit resolution** ($B=256$): $\leq 8$ bits/episode

### Why This Matters

A typical LLM generation has $T \sim 1000$ tokens, each chosen from hundreds of possibilities. The REINFORCE gradient compresses all this rich structure—which words worked well, where the response went wrong, which reasoning steps succeeded—into **one scalar** (the advantage) that modulates a fixed direction $\nabla \log p_\theta(\tau)$.

This structural compression explains why:
- **Training needs thousands of episodes**: With 1 bit/episode and binary feedback, 1000 episodes gives $\leq 1000$ bits total
- **LoRA works well**: LoRA provides 300-500× more capacity than this ceiling
- **Adding parameters doesn't help**: The bottleneck is the scalar advantage, not model capacity

One subtlety: The policy update $\Delta \theta = \alpha G$ happens in a high-dimensional space (the gradient $G$ has millions of dimensions). But the **information content** of this update about the optimal policy is limited by the scalar advantage. The gradient direction $\nabla \log p_\theta(\tau)$ determines *where* the update points, but the scalar advantage determines *what we learn* about which policies are better.

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

**Learning signal**: $G = \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \delta_t$

**Key difference from policy gradient**: Instead of one scalar advantage per episode, we get $T$ TD errors—one per timestep.

### Where Does the Dense Signal Come From?

"The environment only provides rewards at certain steps. How can actor-critic get more information per episode?"

The extra information doesn't come from the environment *in the current episode*—it comes from the **critic's accumulated knowledge from all past episodes**. The critic $V_\phi(s)$ acts as compressed memory, learning to predict expected returns based on thousands of previous rollouts.

The TD error:
$$\delta_t = \underbrace{(r_t + \gamma V_\phi(s_{t+1}))}_{\text{Observed outcome}} - \underbrace{V_\phi(s_t)}_{\text{Historical expectation}}$$

captures how much the observed outcome surprised the critic's learned expectations. Instead of treating each episode as independent (like policy gradient), every step gets evaluated against distilled knowledge of all previous trials.

### The Information Ceiling

**Theorem 2 (Actor-Critic Information Ceiling):**

*Under assumptions A1 and A2 (applied to rewards with resolution $B_r$), actor-critic's information bandwidth satisfies:*

$$I(G; \pi^\star \mid \mathcal{H}) \leq H(\mathbf{r} \mid \tau, \mathcal{H})$$

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
$$G = \sum_{t=0}^{T-1} \nabla \log \pi_\theta(a_t|s_t) \cdot \delta_t$$

Given $(\tau, \mathcal{H})$:
- The policy $\theta = \theta(\mathcal{H})$ is deterministic
- All states and actions in $\tau$ are known
- The gradient is a deterministic function of $(\tau, \boldsymbol{\delta}, \mathcal{H})$ where $\boldsymbol{\delta} = (\delta_0, \ldots, \delta_{T-1})$

By data processing:
$$I(G; \pi^\star \mid \mathcal{H}) \leq I((\tau, \boldsymbol{\delta}); \pi^\star \mid \mathcal{H})$$

By chain rule:
$$I((\tau, \boldsymbol{\delta}); \pi^\star \mid \mathcal{H}) = I(\tau; \pi^\star \mid \mathcal{H}) + I(\boldsymbol{\delta}; \pi^\star \mid \tau, \mathcal{H})$$

As in policy gradient, $\tau \perp \xi \mid \mathcal{H}$, so:
$$I(\tau; \pi^\star \mid \mathcal{H}) = 0$$

Therefore:
$$I(G; \pi^\star \mid \mathcal{H}) \leq I(\boldsymbol{\delta}; \pi^\star \mid \tau, \mathcal{H})$$

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
$$I(G; \pi^\star \mid \mathcal{H}) \leq H(\mathbf{r} \mid \tau, \mathcal{H})$$

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

**Equality holds** if and only if rewards are informationally independent (which requires factorized $\xi$).

### Case Analysis

#### Case 1: Terminal Reward Only

**Setup**: Only $r_{T-1} \neq 0$

**Reward entropy**:
$$H(\mathbf{r} \mid \tau, \mathcal{H}) = H(r_{T-1} \mid \tau, \mathcal{H}) \leq \log_2(B_r) = 1 \text{ bit (binary)}$$

**TD errors**: For $t < T-1$:
$$\delta_t = 0 + \gamma V_\phi(s_{t+1}) - V_\phi(s_t) \quad \text{(deterministic given } \tau, \mathcal{H})$$

Only $\delta_{T-1} = r_{T-1} - V_\phi(s_{T-1})$ is random.

**Information**: $I(G; \pi^\star \mid \mathcal{H}) \leq 1$ bit

**Same as policy gradient!** No advantage from actor-critic when there's only terminal reward.

---

#### Case 2: Dense Independent Rewards

**Setup**: Each $(s_t, a_t)$ has independent reward parameter $\xi_t \sim p(\xi_t)$, so $R_\xi(s_t, a_t) = R_{\xi_t}(s_t, a_t)$

**Reward entropy**:
$$H(\mathbf{r} \mid \tau, \mathcal{H}) = \sum_{t=0}^{T-1} H(r_t \mid \tau, \mathcal{H}) = T \log_2(B_r)$$

**For $T=1000$, $B_r=3$**:
$$H(\mathbf{r} \mid \tau, \mathcal{H}) = 1000 \times 1.58 = 1580 \text{ bits}$$

**Actor-critic can access**: Up to 1580 bits (through separate TD errors preserving temporal structure)

**Policy gradient accesses**: Only $\sim 8$ bits (through scalar advantage that loses temporal structure)

**Advantage**: $\sim 200\times$ more information!

---

#### Case 3: Partially Shared Rewards (Realistic)

**Setup**: Rewards have both global and local components
$$R_\xi(s_t, a_t) = \xi_{\text{global}} \cdot g(\tau) + \xi_t \cdot h(s_t, a_t)$$

**Reward entropy**: Intermediate
$$\log_2(B_r) < H(\mathbf{r} \mid \tau, \mathcal{H}) < T \log_2(B_r)$$

Observing early rewards reveals $\xi_{\text{global}}$, making later rewards more predictable.

Rough estimate: $H(\mathbf{r} \mid \tau, \mathcal{H}) \approx \alpha T \log_2(B_r)$ where $\alpha \in [0.1, 0.5]$ represents the "effective independence fraction."

**For $\alpha = 0.2$, $T=1000$, $B_r=3$**:
$$H(\mathbf{r} \mid \tau, \mathcal{H}) \approx 0.2 \times 1580 \approx 316 \text{ bits}$$

**Actor-critic advantage**: $\sim 40\times$ over policy gradient

---

### Why The Critic Doesn't Add Information

**Key insight**: The TD transformation $\mathbf{r} \to \boldsymbol{\delta}$ is invertible:
$$\delta_t = r_t + [\gamma V_\phi(s_{t+1}) - V_\phi(s_t)]$$

Given $\boldsymbol{\delta}$ and $V_\phi$ (which is known from $\mathcal{H}$), we can recover $\mathbf{r}$.

**The critic provides**:
- **Better credit assignment**: $\delta_t$ weights each timestep appropriately based on learned expectations
- **Variance reduction**: Learned baseline adapts to state
- **Temporal preservation**: Each reward affects its own timestep, avoiding lossy aggregation

**The critic does NOT provide**:
- **More information**: $I(\boldsymbol{\delta}; \pi^\star \mid \tau, \mathcal{H}) = I(\mathbf{r}; \pi^\star \mid \tau, \mathcal{H})$
- **Higher capacity**: Bound is still $H(\mathbf{r} \mid \tau, \mathcal{H})$

**The advantage comes from** avoiding the lossy aggregation $\mathbf{r} \to \sum_t \gamma^t r_t$ that policy gradient performs.

### Practical Achievability

The bound $H(\mathbf{r} \mid \tau, \mathcal{H})$ is an **upper limit**. Practical actor-critic achieves less due to:

**Bootstrap correlation**: All TD errors share the same $V_\phi$. If the critic systematically overestimates everywhere, all $\delta_t$ become correlated:
- If $\delta_0$ is surprisingly negative → $\delta_1, \delta_2, \ldots$ likely also negative
- This reduces effective information: $H(\delta_t \mid \delta_{<t}) < H(\delta_t)$

**Estimated practical information** (for dense, partially correlated rewards):
- Theoretical ceiling: 316-1580 bits/episode
- With correlation $\rho \approx 0.5$: ~50-200 bits/episode
- Empirical observations: 2-10× speedup over policy gradient suggests ~10-100 bits/episode

---

## Part 4: Why LoRA Works—The Capacity-Information Match

### The Capacity Argument

Consider typical RLHF setup:
- Episodes: $N = 1000$
- LoRA: rank $r=8$, dimension $d=4096$
- Binary preferences: $B=2$

**Information gained** (from policy gradient):

Over $N = 1000$ episodes with binary preferences, policy gradient learns at most:
$$N \times \log_2(B) = 1000 \times 1 = 1000 \text{ bits}$$

of new information about the optimal policy $\pi^\star$.

**LoRA capacity** (order-of-magnitude estimate):
- Parameters: $2rd = 65{,}000$
- Effective bits per parameter: **5-8**
  - *Justification*: After training, parameters take on hundreds to thousands of distinguishable values that meaningfully affect behavior. This is much less than float32 precision (23 bits) but more than crude quantization (2-3 bits). We can bound this:
    - **Upper bound**: Optimization noise and finite training limit effective precision to ~12-16 bits
    - **Lower bound**: LoRA training demonstrably learns more than crude 2-3 bit quantization
    - **Reasonable range**: 5-8 bits balances these constraints
- Total capacity: $65{,}000 \times 5$ to $65{,}000 \times 8$ = **~300,000-500,000 bits**

**The ratio**: LoRA provides **~300-500× more capacity** than the information conveyed by policy gradient's learning signals.

Even if this estimate is off by 2-3×, the qualitative conclusion holds: LoRA has substantial excess capacity.

### The Key Insight

LoRA works because the parameter bottleneck isn't binding. With policy gradient's sparse learning signals (scalar advantages), you have far more capacity than information to store. The bottleneck is **signal density** (1 bit/episode), not model capacity.

Full fine-tuning is overkill: with ~7 billion parameters trying to store ~1000 bits of information, you have 7 million times more capacity than needed. LoRA's parameter count naturally matches what policy gradient's learning signals can actually convey.

**Empirical consistency**: LLM-RL needs 1,000-10,000 episodes to converge, consistent with accumulating 1,000-10,000 bits at 1-3 bits/episode (depending on reward granularity).

### Implications for Actor-Critic

If actor-critic could achieve substantially better information extraction (e.g., 100 bits/episode through dense process rewards):
- **With 1000 episodes**: 100,000 bits of information
- **LoRA capacity**: Still 3-5× excess capacity
- **Conclusion**: LoRA remains sufficient even for efficient actor-critic

Only with dramatic improvements approaching the theoretical ceiling (1000+ bits/episode) would LoRA capacity become limiting—but this requires:
1. Dense rewards at every timestep (expensive to collect)
2. Informationally independent rewards (requires factorized reward parameter)
3. Managing bootstrap correlation in the critic

These conditions are difficult to achieve in practice.

---

## Part 5: Implications and Future Directions

### Why Policy Gradient + LoRA Dominates

Policy gradient + LoRA dominates current LLM fine-tuning because this combination achieves a practical equilibrium:
- **Stable**: No critic training instability
- **Parameter-efficient**: LoRA provides 300-500× excess capacity relative to the 1 bit/episode ceiling
- **Simple**: One scalar advantage per episode, easy to implement

The tradeoff is sample efficiency: at ≤1 bit/episode with binary preferences, thousands of episodes are required. Yet practitioners accept this because the stability advantage outweighs the sample cost.

### Why Actor-Critic Methods Haven't Displaced Policy Gradient

Despite the theoretical potential for 100-1000× higher information bandwidth, actor-critic methods typically achieve only 2-10× sample efficiency gains over policy gradient in practice.

**Three barriers limit actor-critic**:

1. **Terminal reward structure**: Most LLM RLHF uses only terminal preferences. With $H(\mathbf{r}) = 1$ bit, actor-critic has no advantage.

2. **Reward correlation**: Even with dense rewards, if they share global components (style, overall quality), $H(\mathbf{r}) \ll T \log_2(B_r)$.

3. **Bootstrap correlation**: All TD errors share the same $V_\phi$, creating systematic dependencies that reduce effective information bandwidth.

4. **Optimization instability**: Joint actor-critic training is more complex and less stable than pure policy gradient.

This explains why policy gradient + LoRA remains dominant despite its information-theoretic limitations.

### Two Types of Barriers

Our analysis reveals two distinct barriers:

**1. Structural information limits** (fundamental):
- Policy gradient: $\leq \log_2(B)$ bits/episode—cannot be exceeded by any algorithm using scalar advantages
- Actor-critic: $\leq H(\mathbf{r} \mid \tau, \mathcal{H})$—depends on reward structure, not algorithm

**2. Practical implementation barriers**:
- Terminal vs. dense rewards (data collection cost)
- Reward correlation structure (inherent to task)
- Bootstrap correlation (inherent to TD learning)
- Optimization stability (engineering challenge)

### Research Directions

**Priority 1: Engineer denser ground-truth signals**
- **Process rewards**: Per-token human annotations or learned reward models
- **Multi-resolution feedback**: Combine episode-level and step-level rewards
- **Why this helps**: Directly increases $H(\mathbf{r})$, benefiting both policy gradient (via higher $B$) and actor-critic (via higher $T \times B_r$)

**Priority 2: Test the framework's predictions empirically**
- Measure actual information gain per episode in trained models
- Quantify reward correlation structure in real tasks
- Test whether sample efficiency scales with $H(\mathbf{r})$ as predicted

**Priority 3: Improve actor-critic for LLM scale**
- Low-rank value function architectures (matching LoRA structure)
- Ensemble critics to reduce systematic bias
- Better optimization techniques for joint training

**Priority 4: Explore decorrelation techniques**
- Eligibility traces ($\lambda$-returns) to diversify bootstrap targets
- Multi-step returns with varying horizons
- Note: These may have fundamental limits due to shared $V_\phi$

**Priority 5: Alternative paradigms**
- Monte Carlo methods (no bootstrapping → no correlation, but high variance)
- Model-based RL (learn dynamics, plan without bootstrap)
- Hybrid MC/TD approaches

**Not needed**: More parameters. LoRA already provides 300-500× excess capacity relative to policy gradient's ceiling. Even with 100× improved actor-critic (100 bits/episode × 1000 episodes = 100,000 bits), LoRA would still have 3-5× excess capacity.

---

## Limitations and Future Work

**Theoretical limitations**:
1. Our bounds assume deterministic optimal policies (A1), which may not hold exactly in stochastic or degenerate settings
2. The "effective bits per parameter" (5-8 bits) for LoRA is an order-of-magnitude estimate, not precisely measured
3. We model algorithms using idealized Bayesian inference, which doesn't capture actual optimization dynamics
4. The reward correlation structure ($\alpha \in [0.1, 0.5]$) is speculative

**Empirical gaps**:
1. We do not empirically measure information gain per episode in real training runs
2. The relationship between reward correlation and practical actor-critic performance is not validated
3. The actual effective resolution for advantages and TD errors is task-dependent and unmeasured

**Future work should**:
- Measure information flow empirically in trained RL systems
- Quantify reward correlation structure across different tasks
- Test whether sample efficiency improvements scale with signal density as predicted
- Empirically determine effective parameter capacity in trained LoRA modules
- Explore whether the theory extends to other RL settings (model-based, offline, multi-agent)

---

## References

**LLM Fine-Tuning**:
- Ouyang, L., et al. (2022). "Training language models to follow instructions with human feedback." *NeurIPS*. (InstructGPT)
- Hu, E. J., et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." *ICLR*.

**Reinforcement Learning**:
- Sutton, R. S., McAllester, D., Singh, S., & Mansour, Y. (1999). "Policy gradient methods for reinforcement learning with function approximation." *Advances in Neural Information Processing Systems*, 12.
- Konda, V., & Tsitsiklis, J. (1999). "Actor-critic algorithms." *Advances in Neural Information Processing Systems*, 12.
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