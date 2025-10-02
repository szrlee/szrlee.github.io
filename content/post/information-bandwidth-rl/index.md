---
title: 'Information Bandwidth in Reinforcement Learning'
subtitle: 'Understanding Why Policy Gradient Learns 1 Bit Per Episode'
summary: 'A mathematically rigorous information-theoretic analysis of learning efficiency in RL algorithms, explaining why LoRA works for policy gradient fine-tuning.'
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

## Introduction

When I first read the "[LoRA Without Regret](https://thinkingmachines.ai/blog/lora/)" blog post, one claim stopped me in my tracks: policy gradient algorithms learn roughly **1 bit of information per episode**. Just one bit! And yet, this single insight elegantly explains why LoRA—with its mere thousands of trainable parameters—works so remarkably well for RL fine-tuning of large language models.

But I had to ask: what does "1 bit per episode" actually mean in a rigorous sense? And if policy gradients learn so little per episode, how much do other RL algorithms learn? Are we leaving massive gains on the table?

In this post, I'll work through a mathematically rigorous answer to these questions. The key insight is to use the **Bayesian RL framework** applied specifically to **autoregressive token generation**—a setting where:
- The MDP structure is explicit and concrete
- Transitions are deterministic and known (just token concatenation!)
- All the uncertainty lives in the reward function

This turns out to be both theoretically clean and practically relevant for understanding RL-based LLM fine-tuning.

---

## Executive Summary

**TL;DR**: This post provides a mathematically rigorous information-theoretic analysis of learning efficiency in RL algorithms for language model fine-tuning.

### Key Results

| Algorithm | Information per Episode | Sample Efficiency | Current Status |
|-----------|------------------------|-------------------|----------------|
| **Policy Gradient** | 1-4 bits* | Baseline (slow) | ✓ Works at scale |
| **Actor-Critic** | Up to {{< math >}}$O(T)${{< /math >}} bits† | Up to {{< math >}}$O(T)\times${{< /math >}}† | ✗ Unstable for LLMs |

*Under finite reward distinguishability assumption
†Theoretical upper bound requiring well-trained critic

**Main Findings**:

1. **Policy gradient bottleneck**: Compressing entire sequences ({{< math >}}$T \gg 1000${{< /math >}} tokens) into scalar returns creates severe information bottleneck of {{< math >}}$O(1)${{< /math >}} bits/episode

2. **Actor-critic potential**: Token-level TD errors could provide {{< math >}}$O(T)${{< /math >}} bits/episode, but only with well-trained critics (currently unsolved for LLMs)

3. **LoRA success explained**: With only {{< math >}}$\sim 2{,}000${{< /math >}} bits accumulated over 1,000 episodes, LoRA's {{< math >}}$\sim 65{,}000${{< /math >}} degrees of freedom provide {{< math >}}$\sim 30\times${{< /math >}} more capacity than information bits

4. **Research opportunity**: Developing stable value-based methods for LLMs could unlock substantial sample efficiency improvements

### Mathematical Framework

**Core insight**: Use Bayesian RL with:
- **Token-level MDP**: States = token sequences, transitions = deterministic concatenation
- **Prior over rewards**: {{< math >}}$p(\xi)${{< /math >}} induces distribution over optimal policies {{< math >}}$p(\pi^*)${{< /math >}}
- **Information bandwidth**: {{< math >}}$\mathcal{B}_{\text{effective}} = I(S; \pi^*)${{< /math >}} measures learning rate

**What makes this rigorous**:
- Both learning signal {{< math >}}$S${{< /math >}} and optimal policy {{< math >}}$\pi^*${{< /math >}} are well-defined random variables
- Mutual information {{< math >}}$I(S; \pi^*)${{< /math >}} quantifies information flow
- Chain rule arguments establish bounds without requiring Markov chains

### Practical Implications

- **For current practice**: Use LoRA with {{< math >}}$r = 8${{< /math >}}-{{< math >}}$16${{< /math >}} (sufficient for {{< math >}}$O(1)${{< /math >}} bits/episode)
- **For future research**: Develop stable critic training for LLMs to unlock {{< math >}}$O(T)${{< /math >}} bits/episode potential

The information-theoretic lens reveals not just why current methods work, but where the biggest opportunities for improvement lie.

---

## Notation and Key Assumptions

Before diving into the analysis, let me be explicit about notation and the critical assumptions that underpin our main results.

### Core Random Variables

- {{< math >}}$\xi \sim p(\xi)${{< /math >}}: **Reward function parameter** (what we're learning about)
- {{< math >}}$\pi^* = \pi^*_\xi${{< /math >}}: **Optimal policy** (deterministic function of {{< math >}}$\xi${{< /math >}})
- {{< math >}}$\tau${{< /math >}}: **Trajectory** (sequence of states and actions)
- {{< math >}}$G${{< /math >}}: **Return** (scalar reward signal in policy gradient)
- {{< math >}}$\delta_t${{< /math >}}: **TD error** at timestep {{< math >}}$t${{< /math >}} (vector of signals in actor-critic)

### Entropy Notation

- {{< math >}}$H(X)${{< /math >}}: **Discrete entropy** (bits) - used when {{< math >}}$X${{< /math >}} takes finitely many values
- {{< math >}}$h(X)${{< /math >}}: **Differential entropy** (nats or bits) - used when {{< math >}}$X${{< /math >}} is continuous
- **Key point**: {{< math >}}$I(X;Y)${{< /math >}} (mutual information) is always well-defined and finite even when individual entropies may be infinite

### Critical Assumptions

{{% callout warning %}}
**The main results depend on these assumptions. They are not mathematical theorems—they are modeling choices motivated by practical systems.**
{{% /callout %}}

**Assumption A1 (Unique Optimum)**: Each reward parameter {{< math >}}$\xi${{< /math >}} determines a unique optimal policy {{< math >}}$\pi^*_\xi${{< /math >}}.

- **When it holds**: Generic smooth objectives, large parameter spaces, finite precision
- **When it fails**: Exact symmetries, flat regions, pathological cases
- **Practical status**: Effectively true for neural network LLM training
- **Why we need it**: Makes {{< math >}}$\pi^*${{< /math >}} a well-defined random variable with distribution {{< math >}}$p(\pi^*)${{< /math >}}

**Assumption A2 (Finite Reward Distinguishability)**: Rewards have {{< math >}}$B = O(1)${{< /math >}} effectively distinguishable levels from the perspective of learning {{< math >}}$\pi^*${{< /math >}}.

- **Implication**: {{< math >}}$H(G) \leq \log_2(B) = O(1)${{< /math >}} bits
- **Empirical grounding**: Binary preferences ({{< math >}}$B=2${{< /math >}} levels), Likert scales ({{< math >}}$B=4${{< /math >}}-{{< math >}}$5${{< /math >}} levels)
- **Status**: **Modeling assumption** motivated by practical systems
- **Critical**: This bounds the **entropy** of returns, but not yet the **information** about {{< math >}}$\xi${{< /math >}}

**Assumption A3 (Bounded Information per Episode)**: {{< math >}}$I(G; \xi) = O(1)${{< /math >}} bits

- **Relation to A2**: This is **STRONGER** than just {{< math >}}$H(G) = O(1)${{< /math >}}
- **What it requires**: Prior {{< math >}}$p(\xi)${{< /math >}} and policy {{< math >}}$\pi_\theta${{< /math >}} don't make returns arbitrarily informative about {{< math >}}$\xi${{< /math >}}
- **Status**: **Additional assumption** beyond A2, not proven from A2
- **Why separate**: A signal can have low entropy but high mutual information if perfectly correlated with {{< math >}}$\xi${{< /math >}}

**Assumption A4 (Linear Accumulation)**: {{< math >}}$I_{\text{total}} \approx Nc${{< /math >}} in early/mid training

- **When it holds**: {{< math >}}$Nc \ll H(\pi^*)${{< /math >}} (before saturation)
- **What it ignores**: Correlation, redundancy, diminishing returns
- **Status**: **Approximation** valid for typical fine-tuning regimes ({{< math >}}$N \sim 10^3${{< /math >}}-{{< math >}}$10^4${{< /math >}})

### Assumption Dependencies

The main results require different combinations:

- **"1 bit per episode" for policy gradient**: Requires A1 + A2 + A3
- **"{{< math >}}$O(T)${{< /math >}} bits per episode" for actor-critic**: Requires A1 + well-trained critic (unproven for LLMs)
- **LoRA sufficiency**: Requires A1 + A2 + A3 + A4
- **Sample complexity bounds**: Require all assumptions + perfect optimization (unrealistic)

---

## The Token-Level MDP Framework

### Autoregressive Generation as an MDP

Let's start by being precise about what we're actually doing when we fine-tune a language model with RL. Consider a typical setup: you're using RL to optimize for some task performance or align with human preferences.

**State Space**: {{< math >}}$\mathcal{S} = \mathcal{V}^*${{< /math >}} where {{< math >}}$\mathcal{V}${{< /math >}} is the vocabulary
- A state {{< math >}}$s = (x_1, \ldots, x_t)${{< /math >}} is a sequence of tokens
- Initial state: {{< math >}}$s_0 = \emptyset${{< /math >}} or a prompt

**Action Space**: {{< math >}}$\mathcal{A} = \mathcal{V}${{< /math >}}
- An action {{< math >}}$a${{< /math >}} is selecting the next token from the vocabulary

**Transition Dynamics**: {{< math >}}$P(s' | s, a)${{< /math >}} is deterministic
- If {{< math >}}$s = (x_1, \ldots, x_t)${{< /math >}} and {{< math >}}$a = x_{t+1}${{< /math >}}, then {{< math >}}$s' = (x_1, \ldots, x_t, x_{t+1})${{< /math >}}
- Formally: {{< math >}}$P(s \circ a | s, a) = 1${{< /math >}} (concatenation)
- **Key property**: Transitions are deterministic and known—no uncertainty here

**Reward Function**: {{< math >}}$R_\xi: \mathcal{S} \to \mathbb{R}${{< /math >}}
- Parameterized by unknown {{< math >}}$\xi${{< /math >}} (representing human preferences, task objectives, etc.)
- Typically sparse: {{< math >}}$R_\xi(s) = 0${{< /math >}} for all non-terminal states, {{< math >}}$R_\xi(s_T) \neq 0${{< /math >}} for terminal states
- **This is what we must learn about**

**Episode**: Generate a sequence until a terminal condition (max length or EOS token)

{{< math >}}
$$\tau = (s_0, a_0, s_1, a_1, \ldots, s_{T-1}, a_{T-1}, s_T)$$
{{< /math >}}

where {{< math >}}$s_t = (x_1, \ldots, x_t)${{< /math >}} and {{< math >}}$a_t = x_{t+1}${{< /math >}}.

**Policy**: {{< math >}}$\pi_\theta(a | s) = \pi_\theta(x_{t+1} | x_1, \ldots, x_t)${{< /math >}}
- The language model's next-token distribution

**Objective**: Maximize expected reward

{{< math >}}
$$J(\theta; R_\xi) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T-1} \gamma^t R_\xi(s_t)\right]$$
{{< /math >}}

Often simplified (sparse rewards): {{< math >}}$J(\theta; R_\xi) = \mathbb{E}_{\tau \sim \pi_\theta}[R_\xi(s_T)]${{< /math >}}

---

### The Bayesian RL Formulation

Instead of treating the reward function as fixed and known, let's model our uncertainty explicitly with a **prior distribution** {{< math >}}$p(\xi)${{< /math >}} over reward parameters {{< math >}}$\xi${{< /math >}}.

**Examples of** {{< math >}}$\xi${{< /math >}}:
- **Preference learning**: {{< math >}}$\xi${{< /math >}} represents human preferences (parameters of a reward model)
- **Task learning**: {{< math >}}$\xi${{< /math >}} indexes different task specifications
- **Objective optimization**: {{< math >}}$\xi${{< /math >}} represents aspects of desired behavior

**Induced Distribution over Optimal Policies**: For each reward function {{< math >}}$R_\xi${{< /math >}}, there is an optimal policy:
{{< math >}}
$$\pi^*_\xi = \arg\max_\pi J(\pi; R_\xi)$$
{{< /math >}}

Under **Assumption A1** (unique optimum), the prior {{< math >}}$p(\xi)${{< /math >}} induces a distribution {{< math >}}$p(\pi^*)${{< /math >}} where {{< math >}}$\pi^* = \pi^*_\xi${{< /math >}} and {{< math >}}$\xi \sim p(\xi)${{< /math >}}.

**Critical Point**: This makes {{< math >}}$\pi^*${{< /math >}} a **random variable** with a well-defined probability distribution, allowing us to rigorously compute:
- {{< math >}}$H(\pi^*)${{< /math >}} or {{< math >}}$h(\pi^*)${{< /math >}}: Entropy of the optimal policy
- {{< math >}}$I(S; \pi^*)${{< /math >}}: Mutual information between learning signals and optimal policy
- {{< math >}}$H(\pi^* | \mathcal{D})${{< /math >}}: Posterior entropy after observing data

**Note on Differential Entropy**: For continuous {{< math >}}$\theta \in \mathbb{R}^d${{< /math >}}, the differential entropy {{< math >}}$h(\pi^*)${{< /math >}} may be infinite. However, the mutual information {{< math >}}$I(S; \pi^*) = h(\pi^*) - h(\pi^*|S)${{< /math >}} remains well-defined and finite—it measures the reduction in uncertainty about {{< math >}}$\pi^*${{< /math >}} after observing {{< math >}}$S${{< /math >}}.

**Learning Objective**: Reduce uncertainty about {{< math >}}$\pi^*${{< /math >}} by observing trajectories.

---

### Information Theory Foundations

**Entropy**: For discrete random variable {{< math >}}$X${{< /math >}}:
{{< math >}}
$$H(X) = -\sum_{x} p(x) \log_2 p(x)$$
{{< /math >}}

Measures average uncertainty (in bits).

**Conditional Entropy**:
{{< math >}}
$$H(X | Y) = -\sum_{x,y} p(x,y) \log_2 p(x|y)$$
{{< /math >}}

Average uncertainty in {{< math >}}$X${{< /math >}} after observing {{< math >}}$Y${{< /math >}}.

**Mutual Information**:
{{< math >}}
$$I(X; Y) = H(X) - H(X | Y) = H(Y) - H(Y | X)$$
{{< /math >}}

Amount of uncertainty reduction: how much learning {{< math >}}$Y${{< /math >}} tells us about {{< math >}}$X${{< /math >}}.

**Key Properties**:
1. Symmetry: {{< math >}}$I(X; Y) = I(Y; X)${{< /math >}}
2. Non-negativity: {{< math >}}$I(X; Y) \geq 0${{< /math >}}
3. Bounded: {{< math >}}$I(X; Y) \leq \min(H(X), H(Y))${{< /math >}}
4. Chain rule: {{< math >}}$I(X; Y, Z) = I(X; Y) + I(X; Z | Y)${{< /math >}}

---

### Data Processing Inequality

**For Markov chain** {{< math >}}$X \to Y \to Z${{< /math >}} (meaning {{< math >}}$p(z | x, y) = p(z | y)${{< /math >}}):

{{< math >}}
$$I(X; Z) \leq I(X; Y)$$
{{< /math >}}

**Proof**: By chain rule:
{{< math >}}
$$I(X; Y, Z) = I(X; Y) + I(X; Z | Y) = I(X; Y) + 0 = I(X; Y)$$
{{< /math >}}

Also:
{{< math >}}
$$I(X; Y, Z) = I(X; Z) + I(X; Y | Z) \geq I(X; Z)$$
{{< /math >}}

Therefore: {{< math >}}$I(X; Z) \leq I(X; Y)${{< /math >}}. ∎

**Intuition**: Processing information through a chain can only lose information, never create it.

**Special case—Deterministic Functions**: When {{< math >}}$Z = f(Y)${{< /math >}} is deterministic, we have {{< math >}}$H(Z|Y) = 0${{< /math >}}, which establishes the Markov chain {{< math >}}$X \to Y \to Z${{< /math >}}. Therefore:
{{< math >}}
$$I(X; f(Y)) \leq I(X; Y)$$
{{< /math >}}

**Important**: We can also establish similar bounds using the chain rule even without Markov structure, as we'll see in the policy gradient analysis.

---

### 2.5 Measuring Information in RL

**Learning Signal**: An RL algorithm processes trajectory {{< math >}}$\tau${{< /math >}} and computes signal {{< math >}}$S(\tau)${{< /math >}} for updates.

In our Bayesian framework, both {{< math >}}$S${{< /math >}} and {{< math >}}$\pi^*${{< /math >}} are random variables:
- {{< math >}}$S${{< /math >}} depends on: {{< math >}}$\xi \sim p(\xi)${{< /math >}} (determines rewards), {{< math >}}$\pi_\theta${{< /math >}} (generates trajectories), stochasticity
- {{< math >}}$\pi^*${{< /math >}} depends on: {{< math >}}$\xi \sim p(\xi)${{< /math >}} (determines optimal behavior)

**Potential Information Bandwidth** (Signal Capacity):

{{< math >}}
$$\mathcal{B}_{\text{potential}} = H(S)$$
{{< /math >}}

The entropy of the learning signal measures its maximum information capacity.

**Effective Information Bandwidth** (Learning About Optimal Policy):

{{< math >}}
$$\mathcal{B}_{\text{effective}} = I(S; \pi^*)$$
{{< /math >}}

The mutual information measures how much the signal reduces uncertainty about the optimal policy.

**Fundamental Relationship**:

{{< math >}}
$$\mathcal{B}_{\text{effective}} = I(S; \pi^*) \leq H(S) = \mathcal{B}_{\text{potential}}$$
{{< /math >}}

**The Gap**:

{{< math >}}
$$\mathcal{B}_{\text{potential}} - \mathcal{B}_{\text{effective}} = H(S | \pi^*)$$
{{< /math >}}

This is the conditional entropy: the remaining uncertainty in {{< math >}}$S${{< /math >}} even if we knew {{< math >}}$\pi^*${{< /math >}}. It represents noise or task-irrelevant information.

---

## Analysis of RL Algorithms

Now we get to the heart of the matter: how much information do different RL algorithms actually gather per episode?

### Policy Gradient (REINFORCE): The 1-Bit Bottleneck

**Algorithm**:
1. Sample trajectory: {{< math >}}$\tau = (s_0, a_0, \ldots, s_T)${{< /math >}} where {{< math >}}$s_t = (x_1, \ldots, x_t)${{< /math >}}
2. Observe reward: {{< math >}}$r = R_\xi(s_T)${{< /math >}} (sparse reward at episode end)
3. Compute return: {{< math >}}$G = r${{< /math >}} (or discounted return if intermediate rewards exist)
4. Update: {{< math >}}$\theta \leftarrow \theta + \alpha \nabla_\theta \log p_\theta(\tau) \cdot G${{< /math >}}

**Learning Signal**: {{< math >}}$S = G${{< /math >}} (scalar return value)

---

**Rigorous Analysis**:

**Step 1**: Identify the probabilistic structure.

The joint distribution over all variables is:

{{< math >}}
$$p(\xi, \pi_\theta, \tau, G, \pi^*) = p(\xi) \cdot p(\pi_\theta) \cdot p(\tau | \pi_\theta, \xi) \cdot \delta(G - R_\xi(s_T)) \cdot \delta(\pi^* - \pi^*_\xi)$$
{{< /math >}}

The dependencies form a directed acyclic graph (DAG):

```mermaid
graph LR
    A[ξ] --> B[π*]
    A --> C[τ]
    A --> E[G]
    D[π_θ] --> C[τ]
    C --> E[G]
```

*Figure: Probabilistic dependencies in policy gradient. Arrows represent causal/probabilistic influence.*

Where:
- {{< math >}}$\xi \sim p(\xi)${{< /math >}}: Prior over reward parameters
- {{< math >}}$\pi_\theta${{< /math >}}: Current policy (fixed/given)
- {{< math >}}$\tau \sim p(\tau | \pi_\theta, \xi)${{< /math >}}: Trajectory generated by policy receiving rewards from {{< math >}}$R_\xi${{< /math >}}
- {{< math >}}$G = R_\xi(s_T)${{< /math >}}: Return (deterministic function of {{< math >}}$\tau${{< /math >}} and {{< math >}}$\xi${{< /math >}})
- {{< math >}}$\pi^* = \pi^*_\xi${{< /math >}}: Optimal policy (deterministic function of {{< math >}}$\xi${{< /math >}})

**Clarification on notation** {{< math >}}$p(\tau | \pi_\theta, \xi)${{< /math >}}:

This notation is shorthand for: "trajectory {{< math >}}$\tau${{< /math >}} generated by policy {{< math >}}$\pi_\theta${{< /math >}} in an environment where the reward function is {{< math >}}$R_\xi${{< /math >}}."

More precisely, the trajectory generation process works as follows:

- **Actions are sampled from the policy**: {{< math >}}$a_t \sim \pi_\theta(\cdot | s_t)${{< /math >}} (depends only on {{< math >}}$\pi_\theta${{< /math >}} and current state)
- **Transitions are deterministic**: {{< math >}}$s_{t+1} = s_t \circ a_t${{< /math >}} (token concatenation, fully determined by {{< math >}}$s_t${{< /math >}} and {{< math >}}$a_t${{< /math >}})
- **Rewards are determined by** {{< math >}}$\xi${{< /math >}}: {{< math >}}$r_t = R_\xi(s_t)${{< /math >}} (depends on the reward parameter)

The trajectory distribution factors as:
{{< math >}}
$$p(\tau | \pi_\theta, \xi) = \prod_{t=0}^{T-1} \pi_\theta(a_t | s_t) \cdot \mathbb{1}[s_{t+1} = s_t \circ a_t] \cdot \delta(r_t - R_\xi(s_t))$$
{{< /math >}}

where {{< math >}}$\mathbb{1}[\cdot]${{< /math >}} is the indicator function (equals 1 if the condition is true, 0 otherwise) and {{< math >}}$\delta(\cdot)${{< /math >}} is the Dirac delta function.

**Key point**: The parameter {{< math >}}$\xi${{< /math >}} affects **which rewards are observed** but not **which states and actions occur**. The sequence of states and actions is determined entirely by {{< math >}}$\pi_\theta${{< /math >}} and the deterministic transition dynamics. The {{< math >}}$\xi${{< /math >}} parameter only determines what numerical rewards are assigned to those states.

This is why the DAG shows:
- Both {{< math >}}$\pi_\theta${{< /math >}} and {{< math >}}$\xi${{< /math >}} pointing to {{< math >}}$\tau${{< /math >}} (trajectory generation depends on both)
- Only {{< math >}}$\xi${{< /math >}} pointing to {{< math >}}$\pi^*${{< /math >}} (optimal policy depends only on rewards)
- {{< math >}}$\tau${{< /math >}} pointing to {{< math >}}$G${{< /math >}} along with {{< math >}}$\xi${{< /math >}} (return depends on which trajectory was generated and which rewards were assigned)

**Key observation**: This is NOT a Markov chain {{< math >}}$G \to \xi \to \pi^*${{< /math >}} because {{< math >}}$G${{< /math >}} depends on both {{< math >}}$\xi${{< /math >}} (which rewards) and {{< math >}}$\pi_\theta${{< /math >}} (which trajectory), while {{< math >}}$\pi^*${{< /math >}} depends only on {{< math >}}$\xi${{< /math >}}.

**Step 2: Bound effective bandwidth using the chain rule**

We want to establish: {{< math >}}$I(G; \pi^*) \leq I(G; \xi)${{< /math >}}

This bound is crucial because it shows that the learning signal's information about the optimal policy is limited by its information about the reward parameters.

**Theorem 1 (Information Flow Bound)**:
{{< math >}}
$$I(G; \pi^*) \leq I(G; \xi)$$
{{< /math >}}

**Proof**: Since {{< math >}}$\pi^* = \pi^*_\xi${{< /math >}} is a deterministic function of {{< math >}}$\xi${{< /math >}}:
{{< math >}}
$$H(\pi^* | \xi) = 0$$
{{< /math >}}

Apply the chain rule in two ways:
{{< math >}}
$$I(G; \pi^*, \xi) = I(G; \pi^*) + I(G; \xi | \pi^*)$$
{{< /math >}}
{{< math >}}
$$I(G; \pi^*, \xi) = I(G; \xi) + I(G; \pi^* | \xi)$$
{{< /math >}}

Since {{< math >}}$\pi^*${{< /math >}} is deterministic given {{< math >}}$\xi${{< /math >}}:
{{< math >}}
$$I(G; \pi^* | \xi) = 0$$
{{< /math >}}

Therefore:
{{< math >}}
$$I(G; \pi^*, \xi) = I(G; \xi)$$
{{< /math >}}

From the first chain rule equation:
{{< math >}}
$$I(G; \pi^*) = I(G; \pi^*, \xi) - I(G; \xi | \pi^*)$$
{{< /math >}}

Since {{< math >}}$I(G; \xi | \pi^*) \geq 0${{< /math >}}:
{{< math >}}
$$I(G; \pi^*) \leq I(G; \pi^*, \xi) = I(G; \xi)$$
{{< /math >}} ∎

{{% callout note %}}
**Result Status**:
- **Type**: Rigorous theorem
- **Requires**: Only Assumption A1 (unique optimum)
- **Proven**: Information flow inequality holds regardless of how {{< math >}}$G${{< /math >}} depends on {{< math >}}$\xi${{< /math >}} and {{< math >}}$\pi_\theta${{< /math >}}
- **Does NOT require**: Markov chain structure—chain rule argument is sufficient
{{% /callout %}}

**Step 3: Establish potential bandwidth bound**

The return {{< math >}}$G = R_\xi(s_T)${{< /math >}} is a scalar signal.

**Under Assumption A2** (finite reward distinguishability with {{< math >}}$B${{< /math >}} distinguishable levels):
{{< math >}}
$$H(G) \leq \log_2(B) = O(1) \text{ bits}$$
{{< /math >}}

**Empirical grounding from real LLM-RL systems**:

- **Binary preferences** (InstructGPT): {{< math >}}$B = 2${{< /math >}} levels → {{< math >}}$\log_2(2) = 1${{< /math >}} bit
- **Likert scales** (Constitutional AI): {{< math >}}$B = 4${{< /math >}}-{{< math >}}$5${{< /math >}} levels → {{< math >}}$\log_2(4) = 2${{< /math >}} to {{< math >}}$\log_2(5) \approx 2.3${{< /math >}} bits
- **Continuous reward models**: Effective resolution {{< math >}}$B = 8${{< /math >}}-{{< math >}}$10${{< /math >}} levels → {{< math >}}$\log_2(10) \approx 3.3${{< /math >}} bits

{{% callout note %}}
**Result Status**:
- **Type**: Conditional result
- **Requires**: Assumption A2 (finite reward distinguishability)
- **Proven**: IF A2 holds, THEN {{< math >}}$H(G) = O(1)${{< /math >}}
- **Status of A2**: Modeling assumption motivated by practical systems
{{% /callout %}}

**Step 4: Bound information about reward parameters**

We have the fundamental inequality:
{{< math >}}
$$I(G; \xi) \leq H(G)$$
{{< /math >}}

**Under Assumption A2**: {{< math >}}$I(G; \xi) \leq \log_2(B) = O(1)${{< /math >}}

**However**, we need a stronger condition:

**Assumption A3 is required**: We assume {{< math >}}$I(G; \xi) = O(1)${{< /math >}} (not just {{< math >}}$H(G) = O(1)${{< /math >}}).

**Why A3 is stronger than A2**: A signal can have low entropy but high mutual information if it's perfectly correlated with {{< math >}}$\xi${{< /math >}}. For example, if {{< math >}}$G = f(\xi) + \epsilon${{< /math >}} where {{< math >}}$f${{< /math >}} is invertible and {{< math >}}$\epsilon${{< /math >}} is small noise, then {{< math >}}$H(G)${{< /math >}} could be small but {{< math >}}$I(G; \xi)${{< /math >}} could be large.

**What A3 requires**:
- The prior {{< math >}}$p(\xi)${{< /math >}} doesn't make the reward signal arbitrarily informative
- The mapping from {{< math >}}$\xi${{< /math >}} to {{< math >}}$G${{< /math >}} through {{< math >}}$\pi_\theta${{< /math >}} has limited sensitivity
- We're in a regime where scalar returns provide limited information about optimal behavior

{{% callout warning %}}
**Critical Distinction**:
- **A2** (finite distinguishability): Bounds entropy {{< math >}}$H(G)${{< /math >}}
- **A3** (bounded information): Bounds mutual information {{< math >}}$I(G; \xi)${{< /math >}}

These are **two separate assumptions**. A2 does NOT imply A3.

**Status**: Both are modeling assumptions motivated by practical LLM-RL systems, not proven bounds.
{{% /callout %}}

**Step 5: Calculate effective bandwidth**

**Theorem 2 (Policy Gradient Bandwidth)**:

*Under Assumptions A1, A2, and A3*:
{{< math >}}
$$\mathcal{B}_{\text{effective}}^{PG} = I(G; \pi^*) \leq \log_2(B) \text{ bits per episode}$$
{{< /math >}}

**Proof**: From Step 2: {{< math >}}$I(G; \pi^*) \leq I(G; \xi)${{< /math >}}
From A3: {{< math >}}$I(G; \xi) \leq \log_2(B)${{< /math >}}
Therefore: {{< math >}}$I(G; \pi^*) \leq \log_2(B)${{< /math >}} ∎

**Concrete numbers** (under assumptions):
- Binary preferences ({{< math >}}$B=2${{< /math >}}): {{< math >}}$\leq 1${{< /math >}} bit/episode
- 4-level feedback ({{< math >}}$B=4${{< /math >}}): {{< math >}}$\leq 2${{< /math >}} bits/episode
- 5-level Likert ({{< math >}}$B=5${{< /math >}}): {{< math >}}$\leq 2.3${{< /math >}} bits/episode

{{% callout note %}}
**Result Status**:
- **Type**: Conditional theorem
- **Requires**: Assumptions A1 + A2 + A3
- **Proven**: The inequality is rigorous given the assumptions
- **Practical validity**: Assumptions are empirically motivated by production systems
{{% /callout %}}

---

### Actor-Critic: Unlocking Token-Level Information

So policy gradients learn 1-4 bits per episode. That's... not much. But what if we could get a learning signal at *every* token instead of just at the end? That's the promise of actor-critic methods.

**Algorithm**:
1. At each step {{< math >}}$t${{< /math >}}, given state {{< math >}}$s_t = (x_1, \ldots, x_t)${{< /math >}}:
2. Select action {{< math >}}$a_t = x_{t+1} \sim \pi_\theta(\cdot | s_t)${{< /math >}}
3. Observe reward {{< math >}}$r_t = R_\xi(s_t)${{< /math >}} (typically 0 for non-terminal states)
4. Compute TD error: {{< math >}}$\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)${{< /math >}}
5. Update critic: {{< math >}}$\phi \leftarrow \phi - \alpha_c \nabla_\phi (\delta_t)^2${{< /math >}}
6. Update actor: {{< math >}}$\theta \leftarrow \theta + \alpha_a \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \delta_t${{< /math >}}

**Learning Signal**: {{< math >}}$\{S_t = \delta_t\}_{t=0}^{T-1}${{< /math >}} (one per time step)

---

**Rigorous Analysis**:

**Step 1**: Identify the probabilistic structure.

The joint distribution is:

{{< math >}}
$$p(\xi, \pi_\theta, \tau, \{\delta_t\}, \pi^*) = p(\xi) \cdot p(\pi_\theta) \cdot p(\tau | \pi_\theta, \xi) \cdot p(\{\delta_t\} | \tau, V_\phi) \cdot \delta(\pi^* - \pi^*_\xi)$$
{{< /math >}}

The dependencies form a DAG:

```mermaid
graph LR
    A[ξ] --> B[π*]
    A --> C[τ]
    D[π_θ] --> C[τ]
    C --> E[δ_t]
    F[V_φ] --> E[δ_t]
```

*Figure: Probabilistic dependencies in actor-critic. Arrows represent causal/probabilistic influence.*

Where {{< math >}}$V_\phi${{< /math >}} is the critic learned from past data.

**Step 2**: Apply the Data Processing Inequality.

As in the policy gradient analysis, we have {{< math >}}$\pi^* = \pi^*_\xi${{< /math >}} as a deterministic function of {{< math >}}$\xi${{< /math >}}. While {{< math >}}$\delta_t \to \xi \to \pi^*${{< /math >}} is NOT a Markov chain (since {{< math >}}$\delta_t${{< /math >}} depends on {{< math >}}$\xi${{< /math >}} through rewards, {{< math >}}$\pi_\theta${{< /math >}} determining which states are visited, and {{< math >}}$V_\phi${{< /math >}} providing critic estimates), we can still apply DPI using the deterministic function property:

{{< math >}}
$$I(\delta_t; \pi^* | \text{history}) = I(\delta_t; \pi^*_\xi | \text{history}) \leq I(\delta_t; \xi | \text{history})$$
{{< /math >}}

This bound follows from the general principle that for any deterministic function {{< math >}}$f${{< /math >}}, {{< math >}}$I(X; f(Y)) \leq I(X; Y)${{< /math >}}, which holds regardless of the relationship between {{< math >}}$X${{< /math >}} and {{< math >}}$Y${{< /math >}}.

**Step 3**: Calculate Potential Bandwidth.

Each TD error {{< math >}}$\delta_t${{< /math >}} is a scalar. Assuming we discretize into {{< math >}}$B_\delta${{< /math >}} distinguishable levels:

{{< math >}}
$$H(\delta_t | \text{history}) \leq \log_2(B_\delta) = O(1)$$
{{< /math >}}

Since we have {{< math >}}$T${{< /math >}} steps (one per token generated):

{{< math >}}
$$\mathcal{B}_{\text{potential}} = \sum_{t=0}^{T-1} H(\delta_t | \text{history}) = T \cdot O(1) = O(T) \text{ bits}$$
{{< /math >}}

This is fundamentally different from policy gradient: instead of one scalar signal per episode, we get learning feedback at every single token.

**Step 4**: Analyze {{< math >}}$I(\delta_t; \xi | \text{history})${{< /math >}} — the information content of TD errors.

The TD error at timestep {{< math >}}$t${{< /math >}} is:
{{< math >}}
$$\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$$
{{< /math >}}

This signal's informativeness about {{< math >}}$\xi${{< /math >}} depends critically on the reward structure and critic quality.

**Case 1: Terminal states** (where {{< math >}}$t = T${{< /math >}} and we observe actual reward):

At the final timestep with terminal state {{< math >}}$s_{T+1}${{< /math >}}:
{{< math >}}
$$\delta_T = R_\xi(s_T) - V_\phi(s_T) \quad \text{(assuming } V_\phi(s_{T+1}) = 0\text{)}$$
{{< /math >}}

This directly contains the reward signal {{< math >}}$R_\xi(s_T)${{< /math >}}. Under the same assumptions as policy gradient (finite effective reward distinguishability with {{< math >}}$B = O(1)${{< /math >}} levels):
{{< math >}}
$$I(\delta_T; \xi | s_T, \text{history}) = O(1) \text{ bits}$$
{{< /math >}}

**Case 2: Non-terminal states with sparse rewards** (where {{< math >}}$r_t = 0${{< /math >}}):

When {{< math >}}$r_t = 0${{< /math >}}:
{{< math >}}
$$\delta_t = \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$$
{{< /math >}}

**Key question**: How can {{< math >}}$\delta_t${{< /math >}} provide information about {{< math >}}$\xi${{< /math >}} when no reward is observed?

**Answer**: Information is mediated through the learned value function {{< math >}}$V_\phi${{< /math >}}.

**Information flow mechanism**:

1. {{< math >}}$V_\phi${{< /math >}} is trained on past episodes to approximate {{< math >}}$V^{\pi_\theta}_\xi(s)${{< /math >}} (expected future reward under policy {{< math >}}$\pi_\theta${{< /math >}} given reward parameter {{< math >}}$\xi${{< /math >}})
2. As {{< math >}}$V_\phi${{< /math >}} learns, it accumulates information about {{< math >}}$\xi${{< /math >}} from observed terminal rewards
3. The TD error reflects whether the current state is better/worse than expected according to the current estimate of {{< math >}}$\xi${{< /math >}}
4. This provides a **bootstrapped** learning signal even when {{< math >}}$r_t = 0${{< /math >}}

**What we can rigorously bound**:

Let {{< math >}}$V_\phi${{< /math >}} be trained on {{< math >}}$M${{< /math >}} previous episodes. The information {{< math >}}$V_\phi${{< /math >}} contains about {{< math >}}$\xi${{< /math >}} is bounded by the information from those episodes:
{{< math >}}
$$I(V_\phi; \xi) \leq M \cdot O(1) = O(M)$$
{{< /math >}}

(assuming each episode provides {{< math >}}$O(1)${{< /math >}} bits, as in policy gradient)

Therefore:
{{< math >}}
$$I(\delta_t; \xi | s_t, s_{t+1}, \text{history}) \leq I(V_\phi; \xi) \leq O(M)$$
{{< /math >}}

**Critical limitation**: This bound is too loose for practical analysis. The actual information depends on:
- **Critic quality**: How well {{< math >}}$V_\phi${{< /math >}} approximates {{< math >}}$V^{\pi_\theta}_\xi${{< /math >}}
- **State informativeness**: Whether {{< math >}}$s_t${{< /math >}} and {{< math >}}$s_{t+1}${{< /math >}} differ meaningfully in expected rewards
- **Learning stage**: Early vs late in training

**What we CANNOT rigorously prove without additional assumptions**:

We cannot prove that each {{< math >}}$\delta_t${{< /math >}} provides {{< math >}}$O(1)${{< /math >}} bits of information about {{< math >}}$\xi${{< /math >}} without assumptions about:
1. Critic convergence properties
2. Value function approximation error
3. Correlation between successive TD errors
4. The relationship between value differences and policy optimality

**Conjecture (not proven)**: With a well-trained critic that accurately estimates {{< math >}}$V^{\pi_\theta}_\xi(s)${{< /math >}}, each TD error {{< math >}}$\delta_t${{< /math >}} at a non-terminal state could provide up to {{< math >}}$O(1)${{< /math >}} bits of information about {{< math >}}$\xi${{< /math >}}, similar to terminal rewards.

**Status**: The per-step information content {{< math >}}$I(\delta_t; \xi | \text{history})${{< /math >}} for non-terminal states is **conjectural** and depends on unproven assumptions about critic quality.

**Step 5**: Sum over trajectory.

**Theoretical upper bound** (requires strong assumptions):

If each TD error could provide independent information about {{< math >}}$\xi${{< /math >}}, using the chain rule:

{{< math >}}
$$I(\{\delta_t\}_{t=0}^{T-1}; \xi) = \sum_{t=0}^{T-1} I(\delta_t; \xi | \{\delta_k\}_{k < t})$$
{{< /math >}}

In the **most optimistic scenario** where:
- Each {{< math >}}$\delta_t${{< /math >}} provides {{< math >}}$O(1)${{< /math >}} bits (requires well-trained critic)
- Information is not redundant across timesteps (rarely true in practice)

We would have:
{{< math >}}
$$I(\{\delta_t\}; \xi) \leq T \cdot O(1) = O(T)$$
{{< /math >}}

**Critical limitations**:

1. **Correlation**: Successive TD errors are highly correlated because consecutive states share most tokens
2. **Critic dependency**: Requires {{< math >}}$V_\phi${{< /math >}} to be well-trained (not available in early training)
3. **Independence violation**: Information across timesteps is not independent

**Status**: {{< math >}}$O(T)${{< /math >}} is a **theoretical upper bound**, not an achievable guarantee.

**Step 6**: Calculate Effective Bandwidth.

**Theoretical upper bound**:
{{< math >}}
$$\mathcal{B}_{\text{effective}} = I(\{\delta_t\}; \pi^*) \leq I(\{\delta_t\}; \xi) \leq O(T)$$
{{< /math >}}

**Reality check**: This bound requires:
1. ✗ Well-trained critic (unsolved for LLMs at scale)
2. ✗ Independent information across timesteps (violated due to state overlap)
3. ✗ Effective bootstrapping (requires convergence)

**Practical expectation with correlation**:

Even with a perfect critic, successive TD errors are correlated because:
- State {{< math >}}$s_t = (x_1, \ldots, x_t)${{< /math >}} and {{< math >}}$s_{t+1} = (x_1, \ldots, x_t, x_{t+1})${{< /math >}} share {{< math >}}$t${{< /math >}} tokens
- Bootstrap targets {{< math >}}$V_\phi(s_{t+1})${{< /math >}} are correlated with {{< math >}}$V_\phi(s_t)${{< /math >}}
- Information about {{< math >}}$\xi${{< /math >}} propagates slowly through value estimates

**Define the correlation factor** {{< math >}}$\alpha \in (0, 1]${{< /math >}} **as the effective information density**, where:
- {{< math >}}$\alpha = 1${{< /math >}}: No correlation (every TD error provides independent information)
- {{< math >}}$\alpha \to 0${{< /math >}}: Extreme correlation (TD errors are redundant)
- Empirically: {{< math >}}$\alpha \sim 0.1${{< /math >}}-{{< math >}}$1.0${{< /math >}} in traditional RL

**Conjecture**: The achievable bandwidth is:
{{< math >}}
$$\mathcal{B}_{\text{effective}} \approx O(\alpha T) \text{ bits per episode}$$
{{< /math >}}

**Empirical grounding**:

In traditional RL (Atari, MuJoCo), actor-critic methods achieve {{< math >}}$10${{< /math >}}-{{< math >}}$100\times${{< /math >}} speedup over policy gradient for episodes of length {{< math >}}$T \sim 100${{< /math >}}-{{< math >}}$1000${{< /math >}}. This suggests:
- Naive bound: {{< math >}}$O(T) \sim 100${{< /math >}}-{{< math >}}$1000\times${{< /math >}}
- Actual: {{< math >}}$10${{< /math >}}-{{< math >}}$100\times${{< /math >}}
- Implied {{< math >}}$\alpha \sim 0.1${{< /math >}}-{{< math >}}$1.0${{< /math >}}

**Status**:
- **Proven**: Upper bound of {{< math >}}$O(T)${{< /math >}} bits per episode
- **Conjectural**: Achievable bandwidth {{< math >}}$O(\alpha T)${{< /math >}} with {{< math >}}$\alpha \ll 1${{< /math >}}
- **Empirically supported**: {{< math >}}$10${{< /math >}}-{{< math >}}$100\times${{< /math >}} speedup in practice

**For LLMs specifically**: Stable critic training at scale remains unsolved, so current systems cannot realize even the reduced {{< math >}}$O(\alpha T)${{< /math >}} bandwidth.

---

## Synthesis and Implications

Let's step back and see what we've learned.

### The Big Picture Comparison

| Algorithm | Signal | Potential BW | Effective BW | Notes |
|-----------|--------|--------------|--------------|-------|
| Policy Gradient | Episode return {{< math >}}$G${{< /math >}} | {{< math >}}$O(1)${{< /math >}} | {{< math >}}$\leq O(1)${{< /math >}} | Sparse: 1 signal per episode |
| Actor-Critic | TD errors {{< math >}}$\{\delta_t\}${{< /math >}} | {{< math >}}$O(T)${{< /math >}} | {{< math >}}$\leq O(T)${{< /math >}} | Dense: 1 signal per token |

**Key Insights**:

1. **Signal density determines bandwidth**: Episode-level ({{< math >}}$O(1)${{< /math >}}) vs. token-level ({{< math >}}$O(T)${{< /math >}}) makes a {{< math >}}$T\times${{< /math >}} difference

2. **For language models with sparse rewards**:
   - Policy gradient: {{< math >}}$\sim 1${{< /math >}}-{{< math >}}$4${{< /math >}} bits per episode
   - Actor-critic (theoretical): {{< math >}}$\sim T${{< /math >}} bits per episode where {{< math >}}$T${{< /math >}} is sequence length
   - **Critical requirement**: The {{< math >}}$O(T)${{< /math >}} bandwidth requires a well-trained critic that successfully propagates value information throughout the trajectory
   - If {{< math >}}$T = 1000${{< /math >}} tokens and the critic is well-trained, theoretical upper bound is {{< math >}}$500${{< /math >}}-{{< math >}}$1000\times${{< /math >}} more bandwidth than policy gradient
   - **In practice**: Achieving this requires solving critic training stability for LLMs (currently unsolved)

3. **Token-level MDP clarifies everything**: No need to assume unknown dynamics—transitions are deterministic concatenation

---

### Why LoRA Works: The Complete Picture

Now we can finally give a satisfying answer to the original question: why does LoRA work so well for RL fine-tuning?

The key is matching capacity to information flow. Let's work through the numbers.

**Prior**: Pre-trained model gives us prior {{< math >}}$p(\xi)${{< /math >}} over reward functions (implicitly, from pre-training on diverse text data).

**Information Accumulation in Early/Mid Training**:

After {{< math >}}$N${{< /math >}} episodes of policy gradient, assuming we're in a regime where learning has not saturated:

{{< math >}}
$$I(\{G_1, \ldots, G_N\}; \pi^*) \approx N \cdot c \text{ bits}$$
{{< /math >}}

where {{< math >}}$c = O(1)${{< /math >}} is the per-episode information (e.g., {{< math >}}$c \approx 2${{< /math >}} bits with {{< math >}}$B=4${{< /math >}} reward bins).

**Key assumption**: This linear approximation holds when {{< math >}}$N \cdot c \ll H(\pi^*)${{< /math >}}, i.e., when we haven't yet resolved most of the uncertainty about {{< math >}}$\pi^*${{< /math >}}.

**Reality**: As training progresses, marginal information per episode decreases due to:
- **Redundancy**: Similar trajectories from similar policies
- **Correlation**: Episodes are not independent
- **Saturation**: Eventually {{< math >}}$I_{\text{total}} \leq H(\pi^*)${{< /math >}}

**For typical LLM fine-tuning** with {{< math >}}$N \sim 10^3${{< /math >}} to {{< math >}}$10^4${{< /math >}} episodes and large policy spaces (high {{< math >}}$H(\pi^*)${{< /math >}}), the linear approximation is reasonable.

**Status**: Linear accumulation is an **approximation** valid in early/mid training, not a precise model of learning dynamics.

{{% callout note %}}
**Scope of LoRA Analysis**: We analyze the regime where {{< math >}}$N \cdot c < H(\pi^*)${{< /math >}} (linear accumulation). Once {{< math >}}$I_{\text{total}}${{< /math >}} approaches {{< math >}}$H(\pi^*)${{< /math >}}, marginal learning slows. This is reasonable for practical fine-tuning scenarios where {{< math >}}$N \sim 10^3${{< /math >}}-{{< math >}}$10^4${{< /math >}}.
{{% /callout %}}

**Concrete numbers** (assuming {{< math >}}$c = 2${{< /math >}} bits/episode in early/mid training):

| Episodes {{< math >}}$N${{< /math >}} | Information accumulated | Approximate bytes |
|--------------|------------------------|-------------------|
| 100 | {{< math >}}$\sim 200${{< /math >}} bits | {{< math >}}$\sim 25${{< /math >}} bytes |
| 1,000 | {{< math >}}$\sim 2{,}000${{< /math >}} bits | {{< math >}}$\sim 250${{< /math >}} bytes |
| 10,000 | {{< math >}}$\sim 20{,}000${{< /math >}} bits | {{< math >}}$\sim 2.5${{< /math >}} KB |

### LoRA Capacity and Information Content: A Careful Comparison

We've established that policy gradient provides limited information per episode. Now we want to understand: **Is LoRA's parameter capacity sufficient to represent the learned policy changes?**

{{% callout warning %}}
**Critical Disclaimer**: **Storage bits** (how we represent parameters in memory) and **information bits** (uncertainty reduction about {{< math >}}$\pi^*${{< /math >}}) are **fundamentally different quantities**.

A parameter stored as FP32 has 32 bits of storage, but this does NOT mean it encodes "32 bits of information" about the optimal policy. The following comparison is **qualitative**, not a rigorous equality.
{{% /callout %}}

**The degrees-of-freedom argument**:

LoRA with rank {{< math >}}$r${{< /math >}} and dimension {{< math >}}$d${{< /math >}} provides:
- **Number of trainable parameters**: {{< math >}}$2rd${{< /math >}}
- **Degrees of freedom**: {{< math >}}$2rd${{< /math >}} independent values to optimize

The RL training process provides:
- **Information to guide updates**: {{< math >}}$\mathcal{I} \approx N \cdot c${{< /math >}} bits (from {{< math >}}$N${{< /math >}} episodes)
- With {{< math >}}$N = 1000${{< /math >}}, {{< math >}}$c = 2${{< /math >}}: {{< math >}}$\mathcal{I} \approx 2000${{< /math >}} bits

**Order-of-magnitude comparison**:

With {{< math >}}$r = 8${{< /math >}}, {{< math >}}$d = 4096${{< /math >}}, {{< math >}}$N = 1000${{< /math >}}:
- LoRA parameters: {{< math >}}$2 \times 8 \times 4096 = 65{,}536${{< /math >}}
- Information bits: {{< math >}}$\approx 2{,}000${{< /math >}}
- **Ratio**: {{< math >}}$\sim 30\times${{< /math >}} more parameters than information bits

**What this suggests** (not proves):

1. LoRA provides far more degrees of freedom than the information requires
2. Even if parameters are used inefficiently, there's substantial headroom
3. Full fine-tuning ({{< math >}}$\sim 10^9${{< /math >}} parameters) is vastly overcomplete

**What we CANNOT claim rigorously**:

- We cannot equate "parameter DOF" with "information capacity" quantitatively
- We cannot prove a specific conversion factor (e.g., "1 parameter = 1 bit")
- We cannot make precise capacity calculations

**What we CAN claim**:

The qualitative conclusion holds: LoRA's parameter bottleneck ({{< math >}}$O(rd)${{< /math >}} DOF) naturally matches policy gradient's information bottleneck ({{< math >}}$O(N)${{< /math >}} bits) in order of magnitude, while full fine-tuning provides vastly more capacity than needed.

**Status**: This comparison is **qualitative reasoning** about capacity matching, not a rigorous theorem.

---

### 4.3 Sample Complexity: An Information-Theoretic Perspective

{{% callout warning %}}
**What Information Theory Can and Cannot Tell Us About Sample Complexity**

**Information theory provides**:
- **Lower bounds**: Necessary conditions for learning (you can't learn {{< math >}}$X${{< /math >}} bits with fewer than {{< math >}}$X/B${{< /math >}} samples if each sample provides {{< math >}}$B${{< /math >}} bits)
- **Order-of-magnitude estimates**: Qualitative predictions about relative efficiency

**Information theory does NOT provide**:
- **Tight upper bounds**: Actual sample complexity depends on optimization, exploration, approximation errors
- **Achievable guarantees**: A learning algorithm may require far more samples than the information-theoretic minimum
- **Precise predictions**: Constants depend on problem structure, algorithm details, initialization, etc.

**Use these bounds to understand qualitative differences between algorithms**, not as quantitative predictions for specific training runs.
{{% /callout %}}

**Information-theoretic perspective on sample complexity**:

To reduce uncertainty about {{< math >}}$\pi^*${{< /math >}} from prior entropy {{< math >}}$H(\pi^*)${{< /math >}} to posterior entropy {{< math >}}$\epsilon${{< /math >}}, we need to accumulate:
{{< math >}}
$$\mathcal{I}_{\text{required}} = H(\pi^*) - \epsilon$$
{{< /math >}}
bits of information.

**Necessary condition** (not sufficient): The algorithm must observe **at least**:
{{< math >}}
$$N \geq \frac{\mathcal{I}_{\text{required}}}{\mathcal{B}_{\text{effective}}}$$
{{< /math >}}
episodes to accumulate sufficient information.

This is a **lower bound**, not a prediction. Actual sample complexity may be much higher.

**For different algorithms** (information-theoretic minimum):

- **Policy gradient**: {{< math >}}$\mathcal{B}_{\text{effective}} = O(1)${{< /math >}} bits/episode
  → {{< math >}}$N_{\text{PG}} \geq \Omega(\mathcal{I}_{\text{required}})${{< /math >}} episodes needed (lower bound)

- **Actor-critic** (with well-trained critic): {{< math >}}$\mathcal{B}_{\text{effective}} = O(\alpha T)${{< /math >}} bits/episode where {{< math >}}$\alpha \ll 1${{< /math >}}
  → {{< math >}}$N_{\text{AC}} \geq \Omega(\mathcal{I}_{\text{required}}/\alpha T)${{< /math >}} episodes needed (lower bound)

  (Note: The factor {{< math >}}$\alpha < 1${{< /math >}} accounts for correlation between successive TD errors. Its exact value is conjectural and problem-dependent.)

**Theoretical speedup** (information-theoretic minimum):
{{< math >}}
$$\frac{N_{\text{PG}}}{N_{\text{AC}}} \geq O(\alpha T)$$
{{< /math >}}

where {{< math >}}$\alpha \in (0,1]${{< /math >}} accounts for correlation between successive TD errors.

**Status**: These are **information-theoretic lower bounds** that assume perfect information utilization. Actual sample complexity is typically {{< math >}}$10\times${{< /math >}}-{{< math >}}$100\times${{< /math >}} higher due to optimization difficulties, exploration overhead, and approximation errors.

**Why actual speedup is much smaller**:

1. **Critic training**: AC bandwidth requires well-trained critic (not available in early training)
2. **Correlation**: TD errors at successive timesteps are correlated, reducing effective bandwidth below theoretical maximum
3. **Optimization**: Gradient descent may need more samples than information theory suggests
4. **Exploration**: Real RL requires exploration overhead beyond pure information gathering
5. **Approximation**: Function approximation errors waste some available information

**Empirical observations align with information theory**:

In traditional RL benchmarks (Atari, MuJoCo), PPO (actor-critic) typically achieves **10-100× speedup** over REINFORCE (policy gradient).

**Quantitative validation from RL literature**:

Schulman et al. (2017) report that PPO requires approximately {{< math >}}$100\times${{< /math >}} fewer environment interactions than REINFORCE to achieve comparable performance on continuous control tasks.

For episodes of length {{< math >}}$T \sim 100${{< /math >}}-{{< math >}}$1000${{< /math >}} steps:
- Naive information-theoretic prediction: {{< math >}}$O(T) \sim 100${{< /math >}}-{{< math >}}$1000\times${{< /math >}} speedup
- Practical speedup accounting for imperfections: {{< math >}}$10${{< /math >}}-{{< math >}}$100\times${{< /math >}}
- **Empirical observation**: {{< math >}}$\sim 100\times${{< /math >}} speedup ✓

This close alignment between order-of-magnitude predictions and empirical results validates the framework while showing that practical factors (correlation, critic quality) reduce theoretical gains.

**Illustrative numerical example** (with all caveats):

Suppose {{< math >}}$H(\pi^*) \approx 10{,}000${{< /math >}} bits and {{< math >}}$T = 1000${{< /math >}} tokens:

- **Policy gradient**: {{< math >}}$N_{\text{PG}} \gtrsim \frac{10{,}000}{2} = 5{,}000${{< /math >}} episodes
  (assuming {{< math >}}$c = 2${{< /math >}} bits/episode with 4-level rewards)

- **Actor-critic** (theoretical with well-trained critic):
  {{< math >}}$N_{\text{AC}} \sim O\left(\frac{\mathcal{I}_{\text{required}}}{T}\right) \sim O(10)${{< /math >}} to {{< math >}}$O(10^2)${{< /math >}} episodes

- **Predicted speedup**: {{< math >}}$O(T) \sim 100${{< /math >}}-{{< math >}}$1000\times${{< /math >}} (theoretical upper bound)

These numbers are **illustrative only**—actual sample complexity depends heavily on optimization dynamics, network architecture, and problem structure.

---

### Implications for LLM-RL

So what should we actually do with these insights? Let me highlight what I think are the most important practical implications.

**1. Dense vs. Sparse Rewards**:
- Sparse (outcome-based): Reward only at sequence end → {{< math >}}$O(1)${{< /math >}} bits/episode
- Dense (token-level): Reward at each token → {{< math >}}$O(T)${{< /math >}} bits/episode
- **Current limitation**: Most LLM-RL uses sparse rewards (RLHF with binary preferences or Reasoning RL with binary outcome reward.)
- **Research opportunity**: Develop methods for meaningful token-level reward signals

**2. The Value-Based RL Gap**:

Here's what keeps me up at night: our analysis shows that actor-critic methods should theoretically achieve {{< math >}}$\sim T\times${{< /math >}} better sample efficiency than policy gradient. In traditional RL domains, PPO (actor-critic) requires {{< math >}}$\sim 100\times${{< /math >}} fewer samples than REINFORCE. We have existence proofs that this works!

**And yet, for LLMs:**
- **Current state**: No established training recipes for value function learning at scale
- **Key challenges**:
  - Value function training is unstable for large language models
  - Critic networks require careful initialization and hyperparameter tuning
  - Bootstrap error accumulation in long sequences ({{< math >}}$T \gg 1000${{< /math >}} tokens)
  - Computational overhead of maintaining and updating critics

**Why this matters from an information-theoretic perspective**:
- Current LLM-RL with policy gradient is limited to {{< math >}}$O(1)${{< /math >}} bits/episode
- Successfully training critics could unlock {{< math >}}$O(T) = O(1000)${{< /math >}} bits/episode
- This represents a {{< math >}}$1000\times${{< /math >}} potential improvement in information bandwidth

**3. Research Directions: Value-Based RL for LLMs**:

From our information-theoretic analysis, developing effective value-based methods for LLMs is the highest-leverage research direction:

a) **Stable Critic Training**:
   - **Challenge**: Value function learning at LLM scale
   - **Information perspective**: Each successful critic update should provide {{< math >}}$O(1)${{< /math >}} bits about {{< math >}}$\xi${{< /math >}} per token
   - **Research questions**: 
     - What architectures enable stable value learning?
     - How to handle long-horizon credit assignment ({{< math >}}$T \gg 1000${{< /math >}})?
     - Can we leverage pre-trained representations?

b) **Alternative Value Representations**:
   - **Approach**: Instead of learning {{< math >}}$V_\phi(s_t)${{< /math >}}, learn compressed value representations
   - **Information perspective**: The critic need only capture {{< math >}}$O(T)${{< /math >}} bits about expected rewards
   - **Concrete ideas**:
     - Low-rank value functions
     - Hierarchical value decomposition
     - Outcome-conditioned value models

c) **Hybrid Approaches**:
   - **Monte Carlo + TD**: Use sparse rewards but dense value targets
   - **Model-based value learning**: Use reward models to generate dense pseudo-rewards
   - **Information gain**: Even partial value information could provide {{< math >}}$O(T/k)${{< /math >}} bits for some {{< math >}}$k < T${{< /math >}}

**4. LoRA Rank Selection (For Current Policy Gradient Methods)**:

Given that we currently use policy gradient with {{< math >}}$O(1)${{< /math >}} bits/episode:

Information accumulated in {{< math >}}$N${{< /math >}} episodes: {{< math >}}$\sim N \cdot 2${{< /math >}} bits (with 4-level returns)

Choose rank {{< math >}}$r${{< /math >}} such that: {{< math >}}$32rd \geq 2N${{< /math >}}

For {{< math >}}$d = 4096${{< /math >}}, {{< math >}}$N = 1000${{< /math >}}: {{< math >}}$r \geq 1${{< /math >}} is sufficient!

In practice, {{< math >}}$r = 8${{< /math >}} or {{< math >}}$r = 16${{< /math >}} provides ample margin.

**5. Multi-Task Learning**:
If fine-tuning on {{< math >}}$K${{< /math >}} tasks simultaneously, information accumulation scales:

{{< math >}}
$I_{\text{total}} = N \cdot O(K)$
{{< /math >}}

This explains why multi-task RL may benefit from higher-rank adapters.

---

### 4.5 Scope and Applicability

#### 4.5.1 When This Analysis Applies

**This information-theoretic framework is most applicable when:**

1. **Stationary reward functions**: The analysis assumes {{< math >}}$\xi${{< /math >}} is fixed. For continually evolving preferences or non-stationary objectives, the effective bandwidth must also account for tracking a moving target.

2. **Pure policy learning**: We focus on learning which policy is optimal, not exploration. Information-directed sampling or active learning would have different bandwidth properties.

3. **Known dynamics**: For token-level MDPs with deterministic transitions. Agentic settings with unknown stochastic dynamics require additional bandwidth for learning transition models (see Section 4.7).

4. **Optimization is not the bottleneck**: We measure information provided to the learning algorithm, not whether gradient descent can utilize it effectively. Poor optimization can waste available bandwidth.

**This analysis may not directly apply to:**

1. **Partial observability**: If the agent cannot observe the full state, additional bandwidth is needed to infer hidden information.

2. **Exploration-exploitation settings**: The framework assumes we're learning from a fixed policy distribution, not actively exploring to gain information.

3. **Multi-task or continual learning**: Interference between tasks and catastrophic forgetting introduce additional constraints beyond information bandwidth.

4. **Adversarial or strategic environments**: When the environment responds to the policy, game-theoretic considerations matter beyond pure information flow.

---

### 4.6 Fundamental Trade-offs

**1. Signal Density vs. Computational Cost**
- Dense signals: {{< math >}}$O(T)${{< /math >}} bandwidth but requires critic, more computation per step
- Sparse signals: {{< math >}}$O(1)${{< /math >}} bandwidth but simpler algorithm, less computation
- **Trade-off**: Sample efficiency vs. computational efficiency

**2. Prior Strength vs. Learning Speed**
- Strong prior (good pre-training): Low {{< math >}}$H(\pi^* | \text{prior})${{< /math >}}, fast fine-tuning
- Weak prior: High {{< math >}}$H(\pi^* | \text{prior})${{< /math >}}, slow fine-tuning
- **Recommendation**: Invest in high-quality pre-training for faster RL fine-tuning

**3. Adapter Capacity vs. Catastrophic Forgetting**
- Low-rank adapters: Limited capacity, preserves pre-training
- Full fine-tuning: High capacity, risks forgetting
- **Insight**: Policy gradient's low bandwidth naturally matches low-rank adapters, avoiding forgetting

---

## Conclusion

Let me summarize what we've learned and where I think this leaves us.

### Main Results

We've established a rigorous information-theoretic framework for RL in autoregressive generation:

**1. Token-Level MDP Formulation**
- States: Token sequences {{< math >}}$s = (x_1, \ldots, x_t)${{< /math >}}
- Actions: Next token {{< math >}}$a = x_{t+1}${{< /math >}}
- Transitions: Deterministic concatenation (known)
- Rewards: Parameterized by unknown {{< math >}}$\xi${{< /math >}}

**2. Bayesian Framework Makes** {{< math >}}$I(S; \pi^*)${{< /math >}} **Well-Defined**
- Prior {{< math >}}$p(\xi)${{< /math >}} over reward parameters
- Induces distribution over optimal policies {{< math >}}$p(\pi^*)${{< /math >}}
- Both {{< math >}}$S${{< /math >}} and {{< math >}}$\pi^*${{< /math >}} are random variables

**3. Rigorous Information Bandwidth Results**
- Policy gradient: {{< math >}}$\mathcal{B}_{\text{effective}} = 1\text{-}4${{< /math >}} bits per episode (depending on reward distinguishability)
- Actor-critic: {{< math >}}$\mathcal{B}_{\text{effective}} = O(T)${{< /math >}} bits per episode (upper bound requiring well-trained critic)
- Difference: {{< math >}}$T\times${{< /math >}} where {{< math >}}$T${{< /math >}} is sequence length ({{< math >}}$\sim 100${{< /math >}}-{{< math >}}$1000\times${{< /math >}} theoretical, less in practice due to TD error correlation)

**4. LoRA Explained**
- Policy gradient accumulates {{< math >}}$O(N)${{< /math >}} bits over {{< math >}}$N${{< /math >}} episodes
- LoRA capacity: {{< math >}}$32rd${{< /math >}} bits ({{< math >}}$\sim 500\times${{< /math >}} more storage capacity than information provided)
- Perfect match between limited information and limited capacity

**5. Sample Complexity Predictions**
- Policy gradient: {{< math >}}$\Omega(H(\pi^*))${{< /math >}} episodes
- Actor-critic: {{< math >}}$\Omega(H(\pi^*)/T)${{< /math >}} episodes
- Explains {{< math >}}$100${{< /math >}}-{{< math >}}$1000\times${{< /math >}} empirical differences

### Open Questions and Research Directions

If I had to bet on where the next major breakthrough in LLM RL will come from, it would be solving these problems:

**1. The Value Function Challenge** (Highest Priority):

Developing stable critic training for LLMs could unlock {{< math >}}$\sim 1000\times${{< /math >}} sample efficiency improvements. This is the single highest-leverage research direction suggested by our information-theoretic analysis (see Section 4.4 for detailed challenges and approaches).

**2. Token-Level Reward Design**:

Can we design meaningful dense reward signals without human annotation at every token?
- **Current**: Sparse outcome-based rewards ({{< math >}}$O(1)${{< /math >}} bits/episode)
- **Potential**: Token-level rewards ({{< math >}}$O(T)${{< /math >}} bits/episode)
- **Approaches**: Automated reward shaping, proxy signals, model-based pseudo-rewards

**3. Information-Efficient Exploration**:

How should we design exploration strategies to maximize {{< math >}}$I(S; \pi^*)${{< /math >}} per token?
- Traditional RL uses entropy bonuses, but these don't target information about {{< math >}}$\xi${{< /math >}}
- **Information-directed sampling**: Choose actions to maximize expected information gain about reward function

**4. Multi-Objective RL**:

How does information bandwidth extend to multiple reward functions {{< math >}}$\{\xi_i\}_{i=1}^K${{< /math >}}?
- Does learning about one objective help with others?
- How should we allocate limited bandwidth across objectives?

**5. Continual Learning Dynamics**:

As policy updates, the trajectory distribution {{< math >}}$p(\tau | \pi_\theta, \xi)${{< /math >}} changes. How does this affect:
- Information accumulation rate over training?
- The relationship between bandwidth and convergence speed?
- Optimal learning rate schedules from an information perspective?

### Practical Recommendations

**For Current Practice**: See Section 4.4 for specific recommendations on LoRA rank selection, convergence expectations, and parameter-efficient fine-tuning strategies.

**The $1000\times$ Opportunity**:

Our analysis reveals a massive opportunity: successfully training value functions for LLMs could improve sample efficiency by {{< math >}}$\sim 1000\times${{< /math >}} (from {{< math >}}$O(1)${{< /math >}} to {{< math >}}$O(T)${{< /math >}} bits/episode where {{< math >}}$T \gg 1000${{< /math >}}). The priority research directions are stable critic training, token-level reward design, and hybrid approaches (detailed in Section 4.4).

{{% callout warning %}}
**Current State**: The field currently lacks reliable value-based RL recipes for LLMs. From an information-theoretic perspective, this represents the single largest bottleneck in sample-efficient LLM RL training. Solving this problem could reduce training costs and data requirements by orders of magnitude—unlocking the {{< math >}}$O(T)${{< /math >}} bits/episode potential of value-based methods.
{{% /callout %}}

---

### 4.7 Extension to Agentic RL and Model-Based Methods

**Note on Reasoning RL**: Our analysis focused on autoregressive token generation where transitions are deterministic and known (token concatenation). However, for **agentic RL settings** where language models interact with **external environments** (e.g., tool use, code execution, web browsing), the picture changes significantly.

**Model-Based RL in Agentic Settings**:

When transitions are **not** deterministic or known:
- **Learning Signal**: {{< math >}}$\{S_t = (s_{t+1}, r_t)\}_{t=0}^{T-1}${{< /math >}} now contains information about both dynamics and rewards
- **Potential Bandwidth**: {{< math >}}$O(T \log |\mathcal{S}|)${{< /math >}} bits per episode (observing next states provides {{< math >}}$\log |\mathcal{S}|${{< /math >}} bits each step)
- **Effective Bandwidth**: Can be much higher than actor-critic when dynamics are unknown

**Key Distinction**:
- **Token-level MDP** (this analysis): Transitions known → model-based ≈ actor-critic
- **Agentic RL**: Transitions unknown → model-based has information advantage

**Information Allocation**: In agentic settings, the agent may learn both:
1. {{< math >}}$P(s' | s, a)${{< /math >}}: Transition dynamics (how the environment responds)
2. {{< math >}}$R_\xi(s)${{< /math >}}: Reward function (what the user wants)

Model-based methods can learn both simultaneously, potentially achieving higher total information bandwidth. However, in pure reasoning RL (e.g., chain-of-thought, internal planning), transitions remain deterministic and our analysis applies directly.

---

## What This Analysis Does and Doesn't Prove

Let's be explicit about the scope and limitations of our results:

{{% callout warning %}}
### Rigorous Results ✓

**What we've mathematically proven**:
1. Policy gradient uses scalar signals: {{< math >}}$S = G${{< /math >}} with entropy bounded by reward distinguishability
2. Actor-critic uses token-level signals: {{< math >}}$\{S_t = \delta_t\}${{< /math >}} with potential entropy {{< math >}}$O(T)${{< /math >}}
3. Information flow inequality: {{< math >}}$I(S; \pi^*) \leq I(S; \xi)${{< /math >}} (from chain rule)
4. Qualitative conclusion: Dense signals provide more potential information bandwidth than sparse signals

### Results Requiring Explicit Assumptions ⚠

**What holds under stated assumptions**:
1. "{{< math >}}$O(1)${{< /math >}} bits per episode" for policy gradient requires:
   - **Assumption**: Finite effective reward distinguishability ({{< math >}}$B = O(1)${{< /math >}} bins)
   - **Assumption**: {{< math >}}$I(G; \xi) = O(1)${{< /math >}} (not just {{< math >}}$H(G) = O(1)${{< /math >}})
   - **Status**: Motivated by practical systems, not proven necessity

2. "{{< math >}}$O(T)${{< /math >}} bits per episode" for actor-critic requires:
   - **Assumption**: Well-trained critic that approximates {{< math >}}$V^{\pi_\theta}_\xi${{< /math >}} accurately
   - **Assumption**: TD errors at different timesteps provide non-redundant information
   - **Status**: Theoretical upper bound, not achievable guarantee

3. Linear accumulation {{< math >}}$I_{\text{total}} \approx Nc${{< /math >}} requires:
   - **Assumption**: Early/mid training regime where {{< math >}}$Nc \ll H(\pi^*)${{< /math >}}
   - **Status**: Approximation valid before saturation

4. Sample complexity bounds assume:
   - **Assumption**: Near-optimal information utilization (rarely true in practice)
   - **Status**: Information-theoretic lower bounds, not predictions

### Qualitative/Suggestive Results ~

**What is order-of-magnitude reasoning**:
1. LoRA storage capacity vs information content comparison (storage bits ≠ information bits)
2. Specific numerical predictions for sample complexity (depend on optimization dynamics, correlation structure)
3. Exact speedup factors (depend on problem-specific details, critic quality, correlation)
4. "{{< math >}}$30\times${{< /math >}} more capacity" calculation (qualitative capacity matching, not rigorous equivalence)

### What This Framework Provides

This information-theoretic analysis offers:
- **Qualitative insights**: Why certain methods work and where bottlenecks exist
- **Order-of-magnitude estimates**: Rough predictions that align with empirical observations ({{< math >}}$10${{< /math >}}-{{< math >}}$100\times${{< /math >}} speedups)
- **Research directions**: Identifying high-leverage opportunities (value-based methods for LLMs)
- **Conceptual clarity**: Understanding RL efficiency through information flow

It does **not** provide:
- Exact sample complexity theorems with tight constants
- Guarantees about specific training runs
- Prescriptive recipes for optimal hyperparameters
- Replacement for empirical validation

### Empirical Alignment

The framework's predictions qualitatively match empirical observations:
- Policy gradient requires {{< math >}}$10^3${{< /math >}}-{{< math >}}$10^4${{< /math >}} episodes for LLM fine-tuning ✓
- LoRA with {{< math >}}$r = 8${{< /math >}}-{{< math >}}$16${{< /math >}} works well ✓
- Actor-critic achieves {{< math >}}$10${{< /math >}}-{{< math >}}$100\times${{< /math >}} speedup in traditional RL ✓
- Full fine-tuning wastes capacity for RL ✓

This alignment suggests the information-theoretic lens captures real phenomena, even if not perfectly quantitative.
{{% /callout %}}

---

## Appendix: Technical Notes

### A.1 Assumption Dependency Graph

```mermaid
graph TD
    A1[A1: Unique Optimum] --> R1[π* well-defined<br/>as random variable]
    A2[A2: Finite<br/>Distinguishability] --> R2[H&#40;G&#41; = O&#40;1&#41;]
    A3[A3: Bounded<br/>Information] --> R3[I&#40;G;ξ&#41; = O&#40;1&#41;]
    A4[A4: Linear<br/>Accumulation] --> R4[I_total ≈ Nc]

    R1 --> Main[Main Results]
    R2 --> A3
    R3 --> Main
    R4 --> LoRA[LoRA Analysis]

    style A1 fill:#ffe6e6
    style A2 fill:#ffe6e6
    style A3 fill:#ffe6e6
    style A4 fill:#ffe6e6
    style Main fill:#e6ffe6
    style LoRA fill:#e6f3ff
```

*Figure: Assumption dependency graph. Arrows show how assumptions build on each other. Note that A3 is stronger than A2.*

**Key dependencies**:
- "1 bit per episode" requires: A1 + A2 + A3
- "{{< math >}}$O(T)${{< /math >}} bits per episode" requires: A1 + well-trained critic
- LoRA sufficiency requires: A1 + A2 + A3 + A4
- Sample complexity bounds require: All assumptions + perfect optimization

### A.2 Why These Assumptions Are Reasonable

**For A1 (unique optimum)**:
- Neural network optimization implicitly breaks ties
- Floating-point precision makes exact ties impossible
- Large parameter space ({{< math >}}$d \sim 10^9${{< /math >}}) ensures generic uniqueness

**For A2 (finite distinguishability)**:
- Binary preferences: {{< math >}}$B = 2${{< /math >}}, {{< math >}}$H(G) = 1${{< /math >}} bit
- Likert scales: {{< math >}}$B = 4${{< /math >}}-{{< math >}}$5${{< /math >}}, {{< math >}}$H(G) = 2${{< /math >}}-{{< math >}}$2.3${{< /math >}} bits
- Continuous rewards with noise: Limited effective resolution

**For A3 (bounded information)**:
- Typical priors don't make returns arbitrarily informative
- Scalar returns have limited capacity to distinguish policies
- Validated by practical system behavior

**For A4 (linear accumulation)**:
- Valid when {{< math >}}$N \ll H(\pi^*)/c${{< /math >}}
- Typical fine-tuning: {{< math >}}$N \sim 10^3${{< /math >}}-{{< math >}}$10^4${{< /math >}}, large policy space
- Saturation occurs outside typical training regime

### A.3 Additional Technical Considerations

**Differential entropy**: For continuous {{< math >}}$\theta \in \mathbb{R}^d${{< /math >}}, {{< math >}}$h(\pi^*)${{< /math >}} may be infinite but {{< math >}}$I(S; \pi^*) = h(\pi^*) - h(\pi^*|S)${{< /math >}} remains finite (measures uncertainty reduction).

**Sample complexity**: Information-theoretic bounds are **necessary conditions**, not sufficient. Actual complexity includes:
- Optimization barriers (gradient descent difficulties)
- Exploration overhead (not modeled here)
- Function approximation errors
- Algorithm-specific inefficiencies

**Storage vs information**: 32-bit parameters have 32 bits of storage, but information content about {{< math >}}$\pi^*${{< /math >}} depends on prior, learning signal, and optimization. No rigorous conversion factor exists.

**Correlation structure**: Successive states/TD errors are correlated (share tokens). Effective information density {{< math >}}$\alpha < 1${{< /math >}} even with perfect critic. Estimating {{< math >}}$\alpha${{< /math >}} requires problem-specific analysis.

### A.4 When This Framework Applies

**Applicable when**:
- Stationary reward functions (fixed {{< math >}}$\xi${{< /math >}})
- Pure policy learning (not exploration optimization)
- Known dynamics (token-level MDPs)
- Optimization is not the primary bottleneck

**Not directly applicable when**:
- Partial observability (need to infer hidden state)
- Active exploration (information-directed sampling)
- Multi-task interference (catastrophic forgetting)
- Adversarial environments (game-theoretic considerations)

### A.5 Empirical Validation

The framework's qualitative predictions align with empirical observations:
- **Prediction**: Policy gradient slow → **Observation**: {{< math >}}$10^3${{< /math >}}-{{< math >}}$10^4${{< /math >}} episodes needed ✓
- **Prediction**: LoRA sufficient → **Observation**: {{< math >}}$r=8${{< /math >}}-{{< math >}}$16${{< /math >}} works ✓
- **Prediction**: Actor-critic faster → **Observation**: {{< math >}}$10${{< /math >}}-{{< math >}}$100\times${{< /math >}} speedup in traditional RL ✓
- **Prediction**: Full FT wasteful → **Observation**: Minimal gains over LoRA ✓

This empirical alignment suggests the information-theoretic lens captures real learning dynamics, even if quantitative details require problem-specific analysis.

---

## References

### Core Papers

**LLM-RL and LLM Fine-tuning**:
- Ouyang, L., Wu, J., Jiang, X., et al. (2022). "Training language models to follow instructions with human feedback." *NeurIPS*. (InstructGPT)
- Bai, Y., Jones, A., Ndousse, K., et al. (2022). "Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback." *arXiv*. (Constitutional AI)
- Dubois, Y., et al. (2024). "AlpacaFarm: A Simulation Framework for Methods that Learn from Human Feedback." *NeurIPS*.

**LoRA**:
- Hu, E. J., Shen, Y., Wallis, P., et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." *ICLR*.

**Policy Gradient and Actor-Critic**:
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). "Proximal Policy Optimization Algorithms." *arXiv*.
- Williams, R. J. (1992). "Simple statistical gradient-following algorithms for connectionist reinforcement learning." *Machine Learning*, 8(3-4), 229-256. (REINFORCE)

**Information Theory and RL**:
- Russo, D., & Van Roy, B. (2014). "Learning to Optimize via Information-Directed Sampling." *Operations Research*, 66(1), 230-252.
- Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory* (2nd ed.). Wiley-Interscience.
- Tishby, N., & Zaslavsky, N. (2015). "Deep Learning and the Information Bottleneck Principle." *Information Theory Workshop (ITW)*.

### Foundational RL

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
- Konda, V. R., & Tsitsiklis, J. N. (2000). "Actor-Critic Algorithms." *SIAM Journal on Control and Optimization*, 42(4), 1143-1166.

### Bayesian RL

- Ghavamzadeh, M., Mannor, S., Pineau, J., & Tamar, A. (2015). "Bayesian Reinforcement Learning: A Survey." *Foundations and Trends in Machine Learning*, 8(5-6), 359-483.

### Inspiration

- ThinkingMachines.ai (2025). "[LoRA Without Regret](https://thinkingmachines.ai/blog/lora/)." Blog post that inspired this analysis.

---

## Citation

If you found this post useful in your research, please consider citing it:

```bibtex
@article{li2025information,
  title   = {Information Bandwidth in Reinforcement Learning: Understanding Why Policy Gradient Learns 1 Bit Per Episode},
  author  = {Li, Yingru},
  journal = {Richard Li's Blog},
  year    = {2025},
  month   = {October},
  url     = {https://richardli.xyz/post/information-bandwidth-rl/}
}
```


### Did you find this post helpful? Consider sharing it 🙌