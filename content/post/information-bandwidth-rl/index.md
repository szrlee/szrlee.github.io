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

**Policy gradient's hard limit**: The REINFORCE gradient $G = \nabla \log p_\theta(\tau) \cdot \text{Adv}$ has only a scalar advantage carrying reward information. This creates an information ceiling of $\leq \log_2(B)$ bits per episode. For binary feedback, this is $\leq 1$ bit/episode—explaining why training needs thousands of episodes and why LoRA's modest capacity (300-500× excess) suffices.

**Actor-critic's theoretical potential**: By bootstrapping historical knowledge through a learned critic, actor-critic methods use $T$ scalar TD errors to construct gradients. Under independence assumptions, this gives an upper bound of $\leq T \log_2(B_\delta)$ bits/episode. For $T=1000$ tokens and 8-bit TD errors, this ceiling is $\leq 8000$ bits/episode—potentially 8000× higher than policy gradient.

**Why this matters**: LoRA already provides 300-500× excess capacity relative to policy gradient's information ceiling. Even with substantial improvements in actor-critic methods, LoRA's capacity appears sufficient for foreseeable applications.

| Algorithm | Learning Signal | Information Upper Bound | Achievability |
|-----------|----------------|---------------------|---------------|
| Policy Gradient | Gradient with 1 scalar advantage | ≤ 1 bit/episode (binary) | Tight bound |
| Actor-Critic | Gradient with T scalar TD errors | ≤ 8000 bits/episode* | Upper bound only |

*Assumes independent TD errors—violated by bootstrap correlation in practice.

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

**Interpretation**: Think of it this way: $I(G; \pi^\star \mid \mathcal{H})$ asks "how much does this gradient tell me about the optimal policy, given what I already know?" Early in training, each gradient is informative. Late in training, we've learned most patterns, so each gradient adds less. The **bandwidth** $\mathcal{B}$ is the maximum—the algorithm's capacity limit regardless of training progress.

Our goal is to find upper bounds on $\mathcal{B}$ that depend only on the **structure of the learning signal** (e.g., whether it's a scalar or a vector of scalars), not on the training state. If we can show that $I(G; \pi^\star \mid \mathcal{H}) \leq C$ for **every** possible history $\mathcal{H}$, then:

$$\mathcal{B} = \sup_{\mathcal{H}} I(G; \pi^\star \mid \mathcal{H}) \leq C$$

This bound $C$ characterizes the algorithm's inherent information capacity.

**Note on $\theta$ and history**: In practice, history gets compressed into parameters $\theta$. Our framework conditions on complete history for mathematical cleanliness, but the bounds apply either way.

### Connection to the Original Insight

This formalization directly implements the information-theoretic argument from "[LoRA Without Regret](https://thinkingmachines.ai/blog/lora/)." Their key observation: for the REINFORCE gradient $G = \nabla \log p_\theta(\tau) \cdot \text{Adv}$, the component $\nabla \log p_\theta(\tau)$ is independent of the reward function $R$ given the policy, so **all information about $R$ must flow through the scalar advantage**.

We're extending this in three ways. First, we measure information about the optimal policy directly—asking "what do we learn about $\pi^\star$?" rather than "what do we learn about $R$?". Second, we're explicit about history: the conditioning on $\mathcal{H}$ makes "what we already know" mathematically precise. Third, we formalize when the advantage has bounded entropy (Assumption A2).

Since the optimal policy is a deterministic function of the reward parameters ($\pi^\star = f(\xi)$), learning about $\xi$ through the gradient is equivalent to learning about $\pi^\star$. The core insight remains: **scalar feedback creates an information bottleneck**. We extend this to show what's theoretically possible with denser signals and why current practice makes sense.

### Two Minimal Assumptions

**Assumption A1 (Unique Optimum)**: Each $\xi$ determines a unique optimal policy $\pi^\star_\xi$.

*Justification*: Generic for neural networks with many parameters. Floating-point precision breaks ties; exact degeneracy is measure-zero.

**Assumption A2 (Finite Resolution)**: The reward-dependent scalar(s) in the gradient have finite effective resolution—they can take at most $B$ distinguishable values each.

*Justification*: Holds exactly for binary preferences ($B=2$) or Likert scales ($B=4$-$7$). Approximately true for continuous signals with noise, finite precision, or practical distinguishability limits.

---

## Part 2: Policy Gradient's 1-Bit Ceiling

### The Algorithm

Policy gradient (REINFORCE) works like this:

1. Sample trajectory $\tau = (s_0, a_0, \ldots, s_T)$ using policy $\pi_\theta$
2. Observe return $G_\tau = \sum_{t=0}^T r_t$
3. Compute advantage: $\text{Adv} = G_\tau - b$ (where $b$ is a baseline)
4. Compute gradient estimate: $G = \nabla \log p_\theta(\tau) \cdot \text{Adv}$
5. Update: $\theta \leftarrow \theta + \alpha G$

**Learning signal**: $G = \nabla \log p_\theta(\tau) \cdot \text{Adv}$ (the gradient)

**Key observation**: The gradient has two components:
- $\nabla \log p_\theta(\tau)$: Independent of the reward function $R$ given the current policy
- $\text{Adv}$: A scalar that encodes all reward-dependent information

### The Information Ceiling

**Theorem 1 (Policy Gradient Information Ceiling):**

*Under assumptions A1 and A2, the information bandwidth of policy gradient satisfies:*

$$I(G; \pi^\star \mid \mathcal{H}) \leq \log_2(B) \text{ bits per episode}$$

where $G = \nabla \log p_\theta(\tau) \cdot \text{Adv}$ is the REINFORCE gradient (the learning signal) and $\mathcal{H}$ is the history of all previous episodes.

**Intuition**: The quantity $I(G; \pi^\star \mid \mathcal{H})$ measures **information gain per episode**—how much we learn about the optimal policy from the current gradient, given what we've already learned from all past episodes. The key is that given the history $\mathcal{H}$ (which determines the current policy $\theta$), the trajectory sampling doesn't depend on $\xi$, so all new information must flow through the scalar advantage, creating a hard ceiling of $\log_2(B)$ bits.

<details>
<summary><strong>Detailed Proof (click to expand)</strong></summary>

**Proof Strategy**: We work within the Bayesian framework where both the history $\mathcal{H}$ and the reward parameter $\xi$ are random variables. The key is that given the history (which determines the current policy), the trajectory sampling doesn't depend on $\xi$. All information about $\xi$ (and thus $\pi^\star$) learned in the current episode must flow through the scalar advantage.

---

**Step 1: The Setup**

Recall our Bayesian setup:
- Prior over reward parameters: $p(\xi)$
- This induces a prior over optimal policies: $p(\pi^\star)$ via $\pi^\star = f(\xi)$
- During training, the policy parameters $\theta$ evolve based on observed episode history $\mathcal{H}$

At any given iteration, we have:
- History of past episodes: $\mathcal{H} = \{(\tau_1, r_1), \ldots, (\tau_{n-1}, r_{n-1})\}$
- Current policy parameters: $\theta$ (determined by $\mathcal{H}$ through gradient descent)
- Sampled trajectory: $\tau \sim p_\theta(\tau)$
- Observed advantage: $\text{Adv}$ (computed from rewards $R_\xi$)
- Gradient: $G = \nabla \log p_\theta(\tau) \cdot \text{Adv}$

**Key insight**: The history $\mathcal{H}$ and $\xi$ are dependent (past rewards came from $R_\xi$), and $\mathcal{H}$ determines the current policy $\theta$. But within the current episode, given $\mathcal{H}$ (and thus given $\theta$), the trajectory sampling doesn't depend on $\xi$. So $\nabla \log p_\theta(\tau)$ is independent of $\xi$ given $\mathcal{H}$.

---

**Step 2: Conditional Independence**

**Claim**: Given the history $\mathcal{H}$ of past episodes, the log-probability gradient is independent of the reward parameter $\xi$:

$$\nabla \log p_\theta(\tau) \perp \xi \mid \mathcal{H}$$

**Justification**:

The log-probability gradient is:
$$\nabla \log p_\theta(\tau) = \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t | s_t)$$

This depends only on:
1. The trajectory $\tau = (s_0, a_0, \ldots, s_T, a_T)$
2. The policy parameters $\theta$

The trajectory is sampled from $p_\theta(\tau)$, which is determined entirely by $\theta$ and the (known, deterministic) environment dynamics. The sampling distribution $p_\theta(\tau)$ does not depend on the reward function $R_\xi$.

Therefore, conditioned on $\mathcal{H}$ (which determines $\theta$):
- $\nabla \log p_\theta(\tau)$ is a function of $(\theta, \tau)$
- $\tau$ is sampled from $p_\theta(\tau)$, which doesn't depend on $\xi$
- Hence $\nabla \log p_\theta(\tau) \perp \xi \mid \mathcal{H}$

---

**Step 3: Decomposition**

The gradient can be written as:
$$G = \nabla \log p_\theta(\tau) \cdot \text{Adv}$$

Let $X = \nabla \log p_\theta(\tau)$ and $Y = \text{Adv}$. We have:
- $X \perp \xi \mid \mathcal{H}$ (from Step 2)
- $Y$ depends on $\xi$ (through rewards $R_\xi$)

---

**Step 4: Bounding MI**

We bound the information in $G$ about $\xi$, conditioned on $\mathcal{H}$.

Since $G$ is a deterministic function of $(X, Y)$, by the data processing inequality:
$$I(G; \xi \mid \mathcal{H}) \leq I((X, Y); \xi \mid \mathcal{H})$$

By the chain rule for conditional mutual information:
$$I((X, Y); \xi \mid \mathcal{H}) = I(X; \xi \mid \mathcal{H}) + I(Y; \xi \mid X, \mathcal{H})$$

From Step 2, we have $X \perp \xi \mid \mathcal{H}$, which means:
$$I(X; \xi \mid \mathcal{H}) = 0$$

Therefore:
$$I(G; \xi \mid \mathcal{H}) \leq I(Y; \xi \mid X, \mathcal{H})$$

Since conditioning can only reduce mutual information:
$$I(Y; \xi \mid X, \mathcal{H}) \leq I(Y; \xi \mid \mathcal{H})$$

And by the fundamental bound on mutual information:
$$I(Y; \xi \mid \mathcal{H}) \leq H(Y \mid \mathcal{H})$$

---

**Step 5: The Bound Holds for Any History**

We've shown $I(G; \xi \mid \mathcal{H}) \leq H(Y \mid \mathcal{H})$ where $Y = \text{Adv}$.

By Assumption A2, the advantage takes at most $B$ distinct values. This holds for binary preferences ($B=2$: thumbs up/down giving ±1), Likert scales ($B=5$: ratings 1-5), or continuous rewards with effective finite resolution ($B=256$: distinguishable reward levels accounting for noise and precision).

For any random variable with at most $B$ values, even when conditioned on another variable, the conditional entropy satisfies:
$$H(Y \mid \mathcal{H}) \leq \log_2(B)$$

This holds because the advantage still takes at most $B$ values conditioned on $\mathcal{H}$, and entropy is maximized when uniform over the support.

Therefore, for ANY history $\mathcal{H}$:
$$I(G; \xi \mid \mathcal{H}) \leq H(Y \mid \mathcal{H}) \leq \log_2(B)$$

**Interpretation**: This bound applies uniformly to every possible history $\mathcal{H}$—whether early in training (little accumulated knowledge) or late in training (policy nearly converged). The information gain per episode is always limited by the scalar advantage bottleneck.

---

**Step 6: Data Processing**

By **Assumption A1** (Unique Optimum), we have $\pi^\star = f(\xi)$ where $f$ is deterministic.

This creates a Markov chain:
$$G \to \xi \to \pi^\star$$

even when conditioning on $\mathcal{H}$, since $\pi^\star$ depends only on $\xi$, not on $\mathcal{H}$ or $G$.

By the **data processing inequality**:
$$I(G; \pi^\star \mid \mathcal{H}) \leq I(G; \xi \mid \mathcal{H})$$

---

**Final Result**

Combining Steps 4-6:
$$I(G; \pi^\star \mid \mathcal{H}) \leq I(G; \xi \mid \mathcal{H}) \leq \log_2(B)$$

Therefore:
$$I(G; \pi^\star \mid \mathcal{H}) \leq \log_2(B) \text{ bits per episode}$$ ∎

</details>

This ceiling holds regardless of sequence length $T$, model size, or computational budget—it's an inherent consequence of the scalar advantage bottleneck.

### Concrete Examples

- **Binary preferences** ($B=2$): $\leq 1$ bit/episode
- **Likert scale** ($B=5$): $\leq 2.3$ bits/episode
- **8-bit resolution** ($B=256$): $\leq 8$ bits/episode

### Why This Matters

A typical LLM generation has $T \sim 1000$ tokens, each chosen from hundreds of possibilities. The REINFORCE gradient compresses all this rich structure—which words worked well, where the response went wrong, which reasoning steps succeeded—into **one scalar** (the advantage) that modulates a fixed direction $\nabla \log p_\theta(\tau)$.

This structural compression explains why:
- **Training needs thousands of episodes**: With 1 bit/episode and binary feedback, 1000 episodes gives $\leq 1000$ bits total
- **LoRA works well**: LoRA provides 300-500× more capacity than this ceiling
- **Adding parameters doesn't help**: The bottleneck is the scalar advantage, not model capacity or the dimensionality of $\nabla \log p_\theta(\tau)$

One subtlety: The policy update $\Delta \theta = \alpha G$ happens in a high-dimensional space (the gradient $G$ has millions of dimensions). But the **information content** of this update about the optimal policy is limited by the scalar advantage. The gradient direction $\nabla \log p_\theta(\tau)$ determines *where* the update points, but the scalar advantage determines *what we learn* about which policies are better.

---

## Part 3: Actor-Critic's Dense Signal Upper Bound

**Scope of this analysis**: We derive an information-theoretic upper bound for actor-critic methods under an independence assumption that bootstrap methods fundamentally violate. This bound establishes the theoretical ceiling—showing what's mathematically possible with dense signals—rather than characterizing practical performance. The gap between this ceiling (≤8000 bits/episode) and policy gradient's hard limit (≤1 bit/episode) frames an open question: how much of this theoretical advantage can be realized in practice?

### The Algorithm

Actor-critic methods (A3C, PPO with value function) maintain two components:

- **Actor** $\pi_\theta(a|s)$: The policy with parameters $\theta$
- **Critic** $V_\phi(s)$: Value function estimating expected return from state $s$, with parameters $\phi$

**Training loop**: For each episode, perform updates at each timestep:

1. **Rollout**: Generate trajectory $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots, s_T)$ using current policy $\pi_\theta$

2. **Compute TD errors** at each timestep $t$:
   $$\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$$

   This measures how much better/worse the observed outcome was compared to the critic's expectation.

3. **Update critic** toward the TD target:
   $$\phi \leftarrow \phi + \alpha_\phi \cdot \delta_t \cdot \nabla_\phi V_\phi(s_t)$$

4. **Update actor** using the gradient:
   $$\theta \leftarrow \theta + \alpha_\theta \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \delta_t$$

**Learning signal structure**: The actor gradient is:
$$G = \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \delta_t$$

This has the same structure as REINFORCE, but with $T$ terms instead of one. Each term has:
- $\nabla_\theta \log \pi_\theta(a_t|s_t)$: Independent of $\xi$ given the policy (like $\nabla \log p_\theta(\tau)$ in REINFORCE)
- $\delta_t$: A scalar carrying reward-dependent information (like the advantage in REINFORCE)

**Key observation**: Since the gradient $G$ is a deterministic function of the TD errors {{< math >}} $\{\delta_t\}_{t=0}^{T-1}$ {{< /math >}}, by the data processing inequality:

{{< math >}}
$$I(G; \pi^\star \mid \mathcal{H}) \leq I(\{\delta_t\}; \pi^\star \mid \mathcal{H})$$
{{< /math >}}

So to bound the information in the gradient, we can bound the information in the TD error sequence. This is the approach we take.

**Key difference from policy gradient**: Instead of one scalar advantage per episode, we get $T$ TD errors—one per timestep. This potentially allows much higher information bandwidth.

### Where Does the Dense Signal Come From?

"The environment only provides rewards at certain steps. How can actor-critic get more information per episode?"

The extra information doesn't come from the environment *in the current episode*—it comes from the **critic's accumulated knowledge from all past episodes**. The critic $V_\phi(s)$ acts as compressed memory of historical data, learning to predict expected future returns based on thousands of previous rollouts.

The TD error:

{{< math >}}
$$\delta_t = \underbrace{(r_t + \gamma V_\phi(s_{t+1}))}_{\text{Observed outcome}} - \underbrace{V_\phi(s_t)}_{\text{Historical expectation}}$$
{{< /math >}}

captures how much the observed outcome surprised the critic's learned expectations. The critic **bootstraps** knowledge from the past to provide dense, step-by-step feedback in the present. Instead of treating each episode as independent (like policy gradient), every step gets evaluated against distilled knowledge of all previous trials.

### Extended Assumption

**Assumption A2' (Effective TD Resolution)**: For the purpose of upper bound analysis, we model each TD error $\delta_t$ as having effective resolution $B_\delta$ distinguishable values.

This is a mathematical modeling device, not a claim about actual TD error statistics. Unlike A2 (which applies to actual observations like binary preferences), A2' applies to derived signals. We use $B_\delta = 256$ (8 bits) as an illustrative example based on finite precision arithmetic (float32: ~7 significant digits), neural network optimization noise, and gradient update granularity.

The resulting bound represents a mathematical upper limit on potential information flow, demonstrating the order-of-magnitude advantage (hundreds to thousands of times higher than policy gradient) rather than a precise quantitative prediction.

### The Information Ceiling

**Theorem 2 (Actor-Critic Information Upper Bound):**

*Under assumptions A1 and A2', actor-critic's information bandwidth satisfies:*

{{< math >}}
$$I(G; \pi^\star \mid \mathcal{H}) \leq I(\{\delta_t\}; \pi^\star \mid \mathcal{H}) \leq T \log_2(B_\delta) \text{ bits per episode}$$
{{< /math >}}

where $G = \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \delta_t$ is the actor gradient (the learning signal), {{< math >}} $\{\delta_t\}_{t=0}^{T-1}$ {{< /math >}}
 are the TD errors, and $\mathcal{H}$ is the history of all previous episodes.

**Interpretation**: The conditioning on $\mathcal{H}$ represents what we've already learned from past episodes. The history determines both the current actor parameters $\theta$ and critic parameters $\phi$. The bound measures **information gained in the current episode**, given this accumulated knowledge. The first inequality follows from the data processing inequality (the gradient is a deterministic function of the TD errors). The second inequality bounds the information in the TD error sequence, conditioned on history. We focus on deriving the second bound, which also bounds the gradient's information content.

**⚠️ Warning**: *This bound assumes TD errors are independent **given** $\mathcal{H}$—an assumption violated by bootstrap methods. Even conditioned on history (which determines $V_\phi$), all TD errors $\delta_t$ share the same value function $V_\phi$, creating systematic dependencies. If the critic overestimates everywhere, all TD errors are correlated. This correlation is not a bug but fundamental to TD learning. The bound represents an unachievable theoretical ceiling, not a characterization of practical performance.*

**For $T=1000$ and $B_\delta=256$**: This theoretical ceiling is $8000$ bits/episode—**8000× higher** than policy gradient's 1 bit.

**Intuition**: With $T$ independent signals each carrying $\log_2(B_\delta)$ bits, we get $T \log_2(B_\delta)$ total bits. But this assumes independence—an assumption violated by the bootstrap structure of TD learning.

<details>
<summary><strong>Detailed Proof (click to expand)</strong></summary>

We bound the entropy of the TD error sequence.

**Step 1: Chain Rule**

By the chain rule for entropy:
$$H(\delta_0, \delta_1, \ldots, \delta_{T-1}) = \sum_{t=0}^{T-1} H(\delta_t | \delta_0, \ldots, \delta_{t-1})$$

Using $\delta_{<t} = (\delta_0, \ldots, \delta_{t-1})$: 
{{< math >}} $$H(\{\delta_t\}) = \sum_{t=0}^{T-1} H(\delta_t | \delta_{<t})$$ {{< /math >}}

**Step 2: Bound Per Step**

By Assumption A2', each $\delta_t$ takes at most $B_\delta$ values. For any random variable $X$ with $|X| \leq B_\delta$:

$$H(X | Y) \leq \log_2(B_\delta)$$

for any conditioning variable $Y$. This holds because:
- Conditional entropy is maximized when $X$ is uniform over its support
- Even with conditioning, $X$ still has at most $B_\delta$ values

Therefore:
$$H(\delta_t | \delta_{<t}) \leq \log_2(B_\delta)$$

**Critical observation**: This bound is tight only when $\delta_t$ is nearly independent of $\delta_{<t}$:
$$H(\delta_t | \delta_{<t}) \approx H(\delta_t)$$

If perfectly correlated: $H(\delta_t | \delta_{<t}) = 0$. Reality falls between these extremes.

**Step 3: Sum Over Time**

{{< math >}}
$$H(\{\delta_t\}) = \sum_{t=0}^{T-1} H(\delta_t | \delta_{<t}) \leq T \log_2(B_\delta)$$
{{< /math >}}

**Step 4: Information Flow**

We now relate the information in TD errors to information about $\pi^\star$, accounting for the conditioning on history $\mathcal{H}$.

The TD error sequence {{< math >}}$\{\delta_t\}$ {{< /math >}} depends on $\xi$ through the reward function $R_\xi(\cdot)$ that generates the immediate rewards $r_t$ in the formula $\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$.

Even when conditioning on $\mathcal{H}$ (which determines both $\theta$ and $\phi$), the TD errors still provide information about $\xi$ through the observed rewards. The optimal policy $\pi^\star = f(\xi)$ is a deterministic function of $\xi$ alone (by Assumption A1), not of $\mathcal{H}$.

These TD errors provide information about $\pi^\star$ only indirectly—by revealing information about the reward parameter $\xi$. More formally:
- $\pi^\star = f(\xi)$ is deterministic given $\xi$
- The TD errors {{< math >}} $\{\delta_t\}$ {{< /math >}} depend on $\xi$ through the reward structure
- Given $\xi$, $\pi^\star$ is fully determined
- Therefore, conditioned on $\xi$, the TD errors provide no additional information about $\pi^\star$

This establishes the conditional independence {{< math >}} $\pi^\star \perp \{\delta_t\} \mid (\xi, \mathcal{H})$ {{< /math >}}, giving us the Markov chain (even when conditioning on {{< math >}} $\mathcal{H}$ {{< /math >}}):
{{< math >}}
$$\{\delta_t\} \to \xi \to \pi^\star$$
{{< /math >}}

By the **data processing inequality**:
{{< math >}}
$$I(\{\delta_t\}; \pi^\star \mid \mathcal{H}) \leq I(\{\delta_t\}; \xi \mid \mathcal{H})$$
{{< /math >}}

The full inequality chain:
{{< math >}}
$$I(\{\delta_t\}; \xi \mid \mathcal{H}) \leq H(\{\delta_t\} \mid \mathcal{H}) \leq H(\{\delta_t\}) \leq T \log_2(B_\delta)$$
{{< /math >}}

where:
- First inequality: fundamental bound on mutual information
- Second inequality: conditioning reduces entropy
- Third inequality: from Steps 1-3 above

Since the gradient $G$ is a deterministic function of the TD errors:
{{< math >}}
$$I(G; \pi^\star \mid \mathcal{H}) \leq I(\{\delta_t\}; \pi^\star \mid \mathcal{H})$$ ∎
{{< /math >}}

</details>

### Understanding the Gap Between Theory and Practice

For $T=1000$, $B_\delta=256$:

- **Policy gradient** (binary): $I(G; \pi^\star \mid \mathcal{H}) \leq 1$ bit/episode
- **Actor-critic** (theoretical ceiling): $I(G; \pi^\star \mid \mathcal{H}) \leq 8000$ bits/episode
- **Actor-critic** (achievable): Unknown (likely 10-100 bits/episode due to TD correlation)

**The gap has two types of causes:**

**Primary barrier: Structural correlation (inherent to bootstrap methods)**

TD error correlation is not a bug—it's fundamental to how TD learning works. All TD errors share the same learned $V_\phi$, creating systematic dependencies.

**Concrete example**: If the critic systematically overestimates values by 20% everywhere (a common early-training phenomenon), then all TD errors share this 1.2× bias. If $\delta_0$ is surprisingly negative (indicating overestimation), then $\delta_1, \delta_2, \ldots$ are likely also negative—they're positively correlated.

**Impact**: If TD errors have correlation $\rho \approx 0.5$, this alone could reduce bandwidth by 2× or more. With $\rho \approx 0.8$, reduction could be 5×+.

**Secondary factors: Implementation and approximation**
- Value function approximation error
- Finite sample effects
- Optimization instability
- Gradient descent efficiency

We hypothesize that the correlation barrier likely accounts for most of the gap, reducing achievable bandwidth by perhaps 10-100×. This hypothesis is not tested in this work and remains speculative.

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

**LoRA capacity** (rough estimate):
- Parameters: $2rd = 65{,}000$
- Effective bits per parameter: **5-8 (rough estimate)**
  - *Justification*: After training, parameters likely take on hundreds to thousands of distinguishable values that meaningfully affect behavior. This is much less than float32 precision (23 bits) but more than crude quantization (2-3 bits). The 5-8 bit range represents an educated guess rather than a measured quantity.
- Total capacity: $65{,}000 \times 5$ to $65{,}000 \times 8$ = **~300,000-500,000 bits**

**The ratio**: LoRA provides **~300-500× more capacity** than the information conveyed by policy gradient's learning signals.

Even if our "bits per parameter" estimate is off by 2-3×, the qualitative conclusion holds: LoRA has substantial excess capacity relative to policy gradient's information ceiling.

### The Key Insight

LoRA works because the parameter bottleneck isn't binding. With policy gradient's sparse learning signals (scalar advantages), you have far more capacity than information to store. The bottleneck is **signal density** (1 bit/episode), not model capacity.

Full fine-tuning is overkill: with ~7 billion parameters trying to store ~1000 bits of information, you have 7 million times more capacity than needed. LoRA's parameter count naturally matches what policy gradient's learning signals can actually convey.

**Empirical consistency**: LLM-RL needs 1,000-10,000 episodes to converge, consistent with accumulating 1,000-10,000 bits at 1-3 bits/episode (depending on reward granularity).

### Implications for Actor-Critic

If actor-critic could achieve substantially better correlation management (e.g., 100 bits/episode):
- **With 1000 episodes**: 100,000 bits of information
- **LoRA capacity**: Still 3-5× excess capacity
- **Conclusion**: LoRA remains sufficient even for efficient actor-critic

Only with dramatic improvements approaching the theoretical ceiling would LoRA capacity become limiting—but this may be unachievable due to fundamental correlation in bootstrap methods.

---

## Part 5: Implications and Future Directions

### Why Policy Gradient + LoRA Dominates

Policy gradient + LoRA dominates current LLM fine-tuning because this combination achieves a practical equilibrium. It's stable—no critic training instability—and parameter-efficient, with LoRA providing 300-500× more capacity than the information conveyed by policy gradient's learning signals. The tradeoff is sample efficiency: at ≤1 bit/episode with binary preferences, thousands of episodes are required. Yet practitioners accept this because the stability advantage outweighs the sample cost.

### Why Actor-Critic Methods Haven't Displaced Policy Gradient

Despite the theoretical ceiling being 8000× higher, actor-critic methods typically achieve only 2-10× sample efficiency gains over policy gradient in practice. Bootstrap correlation—where all TD errors share the same learned value function—substantially reduces achievable information bandwidth. Combined with optimization instabilities in joint actor-critic training, this explains why policy gradient + LoRA remains dominant despite its information-theoretic limitations.

### Two Types of Barriers

Our analysis reveals two distinct barriers:

**1. Information-theoretic ceilings** (provably unavoidable):
- Policy gradient: $\leq \log_2(B)$ bits/episode—cannot be exceeded by any algorithm using scalar advantages
- Actor-critic: $\leq T \log_2(B_\delta)$ bits/episode—cannot be exceeded by any algorithm using $T$ TD errors with resolution $B_\delta$

**2. Bootstrap correlation barrier** (specific to TD methods):
- The $T \log_2(B_\delta)$ ceiling assumes independent TD errors
- Bootstrap methods inherently violate this: all $\delta_t$ share the same learned $V_\phi$
- The actual achievable information bandwidth remains unmeasured

**Key open question**: How much of the gap between 1 bit/episode (policy gradient) and 8000 bits/episode (theoretical ceiling) is bridgeable? Is there a practical middle ground, or does bootstrap correlation prevent substantial improvements?

### Research Directions

**Priority 1: Test the framework's predictions empirically**
- Measure TD error correlation in trained actor-critic models
- Test whether decorrelation techniques (e.g., eligibility traces) improve sample efficiency
- Quantify the relationship between learning signal density and convergence rates

**Priority 2: Improve actor-critic stability at LLM scale**
- Low-rank value function architectures (matching LoRA structure)
- Ensemble critics to reduce bias
- Better optimization techniques for joint actor-critic training

**Priority 3: Explore decorrelation techniques**
- Eligibility traces ($\lambda$-returns) to diversify bootstrap targets
- Multi-step returns with varying horizons
- Note: These may have fundamental limits due to bootstrap structure

**Priority 4: Circumvent bootstrap correlation entirely**
- Monte Carlo methods (no bootstrapping, but high variance)
- Model-based RL (learn environment dynamics, plan without bootstrap)
- Hybrid approaches that blend MC and TD

**Priority 5: Engineer denser ground-truth signals**
- Process rewards provide intermediate feedback beyond just outcomes
- Per-token human annotations increase signal granularity
- Both approaches increase $B$ directly, generating richer learning signals

**Not needed**: More parameters. LoRA already provides 300-500× excess capacity relative to policy gradient's information ceiling. Even with 100× improved actor-critic (100 bits/episode × 1000 episodes = 100,000 bits), LoRA would still have 3-5× excess capacity.

---

## Limitations and Future Work

**Theoretical limitations**:
1. Our bounds assume deterministic optimal policies (A1), which may not hold exactly in stochastic or degenerate settings
2. Assumption A2' (effective TD resolution) is not empirically validated—the choice of $B_\delta = 256$ is illustrative
3. We model algorithms using idealized Bayesian inference, which doesn't capture actual optimization dynamics

**Empirical gaps**:
1. We do not empirically validate the TD error correlation hypothesis
2. The "effective bits per parameter" (5-8 bits) for LoRA is a rough estimate, not measured
3. The actual effective resolution for TD errors is task-dependent and unmeasured

**Future work should**:
- Measure TD error correlation in real training runs to validate the bootstrap correlation barrier
- Test whether sample efficiency improvements scale with learning signal density as predicted
- Empirically determine effective parameter capacity in trained LoRA modules
- Explore whether the theory extends to other RL settings (model-based, offline, multi-agent)
- Investigate the relationship between value function approximation quality and achievable information bandwidth

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
```
