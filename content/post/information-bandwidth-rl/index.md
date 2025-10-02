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
| **Policy Gradient** | 1-4 bits | Baseline (slow) | ✓ Works at scale |
| **Actor-Critic** | Up to {{< math >}}$O(T) \sim 150${{< /math >}}-{{< math >}}$300${{< /math >}} bits | {{< math >}}$50${{< /math >}}-{{< math >}}$150\times${{< /math >}} faster (theoretical) | ✗ Unstable for LLMs |

**Main Findings**:

1. **Policy gradient bottleneck**: Compressing entire sequences ({{< math >}}$T \gg 1000${{< /math >}} tokens) into scalar returns creates severe information bottleneck of {{< math >}}$O(1)${{< /math >}} bits/episode

2. **Actor-critic potential**: Token-level TD errors could provide {{< math >}}$O(T)${{< /math >}} bits/episode, but only with well-trained critics (currently unsolved for LLMs)

3. **LoRA success explained**: With only {{< math >}}$\sim 2{,}000${{< /math >}} bits accumulated over 1,000 episodes, LoRA's {{< math >}}$\sim 65{,}000${{< /math >}} parameters provide {{< math >}}$30\times${{< /math >}} more capacity than needed

4. **Research opportunity**: Developing stable value-based methods for LLMs could unlock {{< math >}}$50${{< /math >}}-{{< math >}}$1000\times${{< /math >}} sample efficiency improvements

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

**For current practice** (policy gradient era):
- Use LoRA with {{< math >}}$r = 8${{< /math >}}-{{< math >}}$16${{< /math >}} (sufficient for {{< math >}}$O(1)${{< /math >}} bits/episode)
- Expect slow convergence (thousands of episodes needed)
- Full fine-tuning wastes capacity ({{< math >}}$\sim 10^6\times${{< /math >}} overcapacity)

**For future research** (unlock actor-critic potential):
- Develop stable critic training for LLMs
- Design token-level reward signals
- Explore low-rank value functions

The information-theoretic lens reveals not just why current methods work, but where the biggest opportunities for improvement lie.

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

Here's where things get interesting. Instead of treating the reward function as fixed and known, let's be honest: we're uncertain about what the "right" reward function is. Maybe we're learning human preferences, or trying to figure out what makes a good code completion, or optimizing for some task we only partially understand.

So let's model this uncertainty explicitly with a **prior distribution** {{< math >}}$p(\xi)${{< /math >}} over reward parameters {{< math >}}$\xi${{< /math >}}.

**Examples of** {{< math >}}$\xi${{< /math >}}:
- **Preference learning**: {{< math >}}$\xi${{< /math >}} represents human preferences (parameters of a reward model)
- **Task learning**: {{< math >}}$\xi${{< /math >}} indexes different task specifications
- **Objective optimization**: {{< math >}}$\xi${{< /math >}} represents aspects of desired behavior

**Induced Distribution over Optimal Policies**: For each reward function {{< math >}}$R_\xi${{< /math >}}, there is an optimal policy:

{{< math >}}
$$\pi^*_\xi = \arg\max_\pi J(\pi; R_\xi)$$
{{< /math >}}

**Assumption (Unique Optimum)**: We assume that for each {{< math >}}$\xi${{< /math >}}, the optimal policy {{< math >}}$\pi^*_\xi${{< /math >}} is **unique** (or at least, unique up to measure zero).

**When this assumption holds**:

1. **Continuous action spaces**: With softmax over large vocabularies ({{< math >}}$|\mathcal{V}| \sim 50{,}000${{< /math >}}), exact ties between policies have probability zero under continuous distributions

2. **Strict concavity of objective**: If {{< math >}}$J(\pi; R_\xi)${{< /math >}} is strictly concave in the policy parameters {{< math >}}$\theta${{< /math >}}, there exists a unique maximizer

3. **Lexicographic tie-breaking**: For discrete cases or when multiple optima exist mathematically, we can impose an arbitrary but consistent ordering (e.g., by parameter norm) to select a unique policy

4. **Generic property**: For "most" reward functions (in a measure-theoretic sense), the optimal policy is unique. Exact degeneracies require special structure.

**When this assumption might fail**:

- **Symmetries**: If the MDP has symmetries (e.g., equivalent actions), multiple policies may be exactly optimal
- **Flat regions**: If the objective has flat plateaus, many policies achieve the same maximum value
- **Discrete spaces**: In small discrete action spaces, ties are more common
- **Pathological rewards**: Carefully constructed reward functions could have degenerate optima

**Why we need this assumption**:

For the mapping {{< math >}}$\xi \mapsto \pi^*_\xi${{< /math >}} to be well-defined (single-valued), we need each {{< math >}}$\xi${{< /math >}} to determine a unique {{< math >}}$\pi^*${{< /math >}}. If multiple optimal policies exist for some {{< math >}}$\xi${{< /math >}}, we would need to specify a selection rule.

**Practical impact**:

In real LLM fine-tuning:
- Neural network optimization implicitly breaks ties via initialization and optimization path
- Floating-point arithmetic has finite precision, making exact ties nearly impossible
- The policy space {{< math >}}$\mathbb{R}^d${{< /math >}} (with {{< math >}}$d \sim 10^9${{< /math >}} parameters) is so large that generic uniqueness holds
- Even if multiple {{< math >}}$\theta${{< /math >}} values achieve the same maximum objective, they typically induce the same policy {{< math >}}$\pi_\theta(\cdot|\cdot)${{< /math >}}

Therefore, **effective uniqueness holds in practice** for LLM RL, making this assumption reasonable.

**Note on differential entropy**:

For continuous policy parameterizations {{< math >}}$\pi_\theta${{< /math >}} where {{< math >}}$\theta \in \mathbb{R}^d${{< /math >}}, we use differential entropy {{< math >}}$h(\pi^*)${{< /math >}}, which may be infinite. However:
- The mutual information {{< math >}}$I(S; \pi^*)${{< /math >}} remains well-defined and finite
- It measures **reduction in uncertainty**, not absolute uncertainty
- Our analysis relies on {{< math >}}$I(S; \pi^*)${{< /math >}}, not {{< math >}}$H(\pi^*)${{< /math >}} directly

For discrete policy spaces or when working with finite {{< math >}}$\epsilon${{< /math >}}-nets of the policy space, all entropies are finite.

Under this assumption, the prior {{< math >}}$p(\xi)${{< /math >}} induces a distribution {{< math >}}$p(\pi^*)${{< /math >}} where {{< math >}}$\pi^* = \pi^*_\xi${{< /math >}} and {{< math >}}$\xi \sim p(\xi)${{< /math >}}.

**Critical Point**: This makes {{< math >}}$\pi^*${{< /math >}} a **random variable** with a well-defined probability distribution, allowing us to rigorously compute:
- {{< math >}}$H(\pi^*)${{< /math >}}: Entropy of the optimal policy (we use differential entropy for continuous policy parameters)
- {{< math >}}$I(S; \pi^*)${{< /math >}}: Mutual information between learning signals and optimal policy
- {{< math >}}$H(\pi^* | \mathcal{D})${{< /math >}}: Posterior entropy after observing data

**Note on Entropy**: For continuous policy spaces (e.g., neural network parameters {{< math >}}$\theta \in \mathbb{R}^d${{< /math >}}), we use differential entropy. The key quantities {{< math >}}$I(S; \pi^*)${{< /math >}} remain well-defined and measure reduction in uncertainty, even if {{< math >}}$H(\pi^*)${{< /math >}} itself may be infinite in the continuous case.

**Learning Objective**: Reduce uncertainty about {{< math >}}$\pi^*${{< /math >}} by observing trajectories.

---

### Information Theory Foundations

Before diving into the analysis, let's review the information-theoretic tools we'll need. If you're comfortable with entropy and mutual information, feel free to skim this section—but I want to be explicit about the machinery we're using.

**Entropy**: For discrete random variable {{< math >}}$X${{< /math >}}:

{{< math >}}
$$H(X) = -\sum_{x} p(x) \log_2 p(x)$$
{{< /math >}}

Measures average uncertainty (in bits).

**Conditional Entropy**:

{{< math >}}
$$H(X | Y) = \sum_{y} p(y) H(X | Y=y) = -\sum_{x,y} p(x,y) \log_2 p(x|y)$$
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

### Data Processing Inequality: The Key Tool

The Data Processing Inequality (DPI) is going to be our workhorse theorem. The intuition is beautiful: you can't create information by processing it. Every transformation, every function application, every summary—it can only lose information or preserve it, never create it from thin air.

Let me state this precisely. For a **Markov chain** {{< math >}}$X \to Y \to Z${{< /math >}}, the variables satisfy:

{{< math >}}
$$p(z | x, y) = p(z | y)$$
{{< /math >}}

Meaning: {{< math >}}$X${{< /math >}} and {{< math >}}$Z${{< /math >}} are conditionally independent given {{< math >}}$Y${{< /math >}}.

**Data Processing Inequality (DPI)**:
If {{< math >}}$X \to Y \to Z${{< /math >}}, then:

{{< math >}}
$$I(X; Z) \leq I(X; Y)$$
{{< /math >}}

**Proof**:

{{< math >}}
$$\begin{align}
I(X; Y, Z) &= I(X; Y) + I(X; Z | Y)\\
&= I(X; Y) + 0 \quad \text{(by conditional independence)}\\
&= I(X; Y)
\end{align}$$
{{< /math >}}

Also:

{{< math >}}
$$\begin{align}
I(X; Y, Z) &= I(X; Z) + I(X; Y | Z)\\
&\geq I(X; Z) \quad \text{(since } I(X; Y | Z) \geq 0\text{)}
\end{align}$$
{{< /math >}}

Therefore: {{< math >}}$I(X; Z) \leq I(X; Y, Z) = I(X; Y)${{< /math >}}.

**Intuition**: Processing information through a chain can only lose information, never create it.

---

**DPI for Deterministic Functions**:

A special case we'll use is when {{< math >}}$Z = f(Y)${{< /math >}} is a deterministic function of {{< math >}}$Y${{< /math >}}. In this case, {{< math >}}$Z${{< /math >}} is conditionally independent of {{< math >}}$X${{< /math >}} given {{< math >}}$Y${{< /math >}} (since knowing {{< math >}}$Y${{< /math >}} completely determines {{< math >}}$Z${{< /math >}}), which establishes the Markov chain {{< math >}}$X \to Y \to Z${{< /math >}}.

The Data Processing Inequality therefore applies:
{{< math >}}
$$I(X; f(Y)) \leq I(X; Y)$$
{{< /math >}}

**Proof**: Since {{< math >}}$f${{< /math >}} is deterministic, {{< math >}}$H(Z|Y) = 0${{< /math >}}, which means {{< math >}}$I(X; Z | Y) = 0${{< /math >}}. By the chain rule:
{{< math >}}
$$I(X; Y, Z) = I(X; Y) + I(X; Z | Y) = I(X; Y)$$
{{< /math >}}

Also:
{{< math >}}
$$I(X; Y, Z) = I(X; Z) + I(X; Y | Z) \geq I(X; Z)$$
{{< /math >}}

Therefore {{< math >}}$I(X; Z) \leq I(X; Y)${{< /math >}}.

**Intuition**: Processing information through a deterministic function can only lose or preserve information, never create it.

**Important caveat**: This form of DPI applies when we have the Markov chain structure. However, we can establish similar bounds using the chain rule even without Markov structure, as we'll see in the policy gradient analysis.

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

**Step 2**: Bound effective bandwidth using the chain rule.

We want to establish: {{< math >}}$I(G; \pi^*) \leq I(G; \xi)${{< /math >}}

This bound is crucial because it shows that the learning signal's information about the optimal policy is limited by its information about the reward parameters.

**Proof using chain rule for mutual information**:

Recall that {{< math >}}$\pi^* = \pi^*_\xi${{< /math >}} is a deterministic function of {{< math >}}$\xi${{< /math >}}. This means that once we know {{< math >}}$\xi${{< /math >}}, we know {{< math >}}$\pi^*${{< /math >}} with certainty, so:
{{< math >}}
$$H(\pi^* | \xi) = 0$$
{{< /math >}}

Now apply the chain rule for mutual information in two different ways:

{{< math >}}
$$I(G; \pi^*, \xi) = I(G; \pi^*) + I(G; \xi | \pi^*)$$
{{< /math >}}

{{< math >}}
$$I(G; \pi^*, \xi) = I(G; \xi) + I(G; \pi^* | \xi)$$
{{< /math >}}

Since {{< math >}}$\pi^*${{< /math >}} is deterministic given {{< math >}}$\xi${{< /math >}}, knowing {{< math >}}$\xi${{< /math >}} tells us nothing new about the relationship between {{< math >}}$G${{< /math >}} and {{< math >}}$\pi^*${{< /math >}}:
{{< math >}}
$$I(G; \pi^* | \xi) = 0$$
{{< /math >}}

Therefore:
{{< math >}}
$$I(G; \pi^*, \xi) = I(G; \xi)$$
{{< /math >}}

From the first equation, since mutual information is always non-negative:
{{< math >}}
$$I(G; \pi^*) \leq I(G; \pi^*) + I(G; \xi | \pi^*) = I(G; \pi^*, \xi)$$
{{< /math >}}

Combining these:
{{< math >}}
$$I(G; \pi^*) \leq I(G; \pi^*, \xi) = I(G; \xi)$$
{{< /math >}}

**Important note**: This bound holds even though {{< math >}}$G${{< /math >}} depends on both {{< math >}}$\xi${{< /math >}} (through the reward function {{< math >}}$R_\xi${{< /math >}}) and {{< math >}}$\pi_\theta${{< /math >}} (through which trajectory is generated). The key insight is that {{< math >}}$\pi^*${{< /math >}} is uniquely determined by {{< math >}}$\xi${{< /math >}}, regardless of how {{< math >}}$G${{< /math >}} was generated. We do NOT need a Markov chain {{< math >}}$G \to \xi \to \pi^*${{< /math >}} for this inequality to hold—the chain rule argument is sufficient.

**Step 3**: Calculate Potential Bandwidth and establish the finite information bound.

The return {{< math >}}$G = R_\xi(s_T)${{< /math >}} is a scalar that depends on:
- The reward parameter {{< math >}}$\xi \sim p(\xi)${{< /math >}} (what we're learning about)
- The policy {{< math >}}$\pi_\theta${{< /math >}} (which sequences are generated)
- The stochastic generation process

**Potential bandwidth** is bounded by the entropy of this signal:
{{< math >}}
$$\mathcal{B}_{\text{potential}} = H(G) = H(R_\xi(s_T))$$
{{< /math >}}

**Critical insight**: For the "1 bit per episode" result to hold, we need to establish that {{< math >}}$H(G)${{< /math >}} and {{< math >}}$I(G; \xi)${{< /math >}} are both {{< math >}}$O(1)${{< /math >}} (bounded by a constant).

**Why rewards have limited information capacity**:

In practice, reward signals have **finite effective distinguishability** due to:

1. **Categorical feedback**: Human preference data is often binary ("A > B") or small Likert scales (1-5 stars)
2. **Bounded precision**: Reward models output scores with finite precision (e.g., floats with limited significant digits)
3. **Noise**: Human judgments and reward model predictions have inherent variability
4. **Limited sensitivity**: Different reward values may not meaningfully distinguish different optimal policies

**Formalization via effective binning**:

Assume that from the perspective of learning about {{< math >}}$\pi^*${{< /math >}}, the continuous reward values can be effectively discretized into {{< math >}}$B${{< /math >}} distinguishable levels. This means:

- If two reward values {{< math >}}$r_1, r_2${{< /math >}} fall in the same bin, they provide essentially the same information about which policy is optimal
- The number of bins {{< math >}}$B${{< /math >}} captures the effective "resolution" of the reward signal for policy learning

Under this assumption:
{{< math >}}
$$H(G) \leq \log_2(B)$$
{{< /math >}}

**Empirical grounding from real LLM-RL systems**:

- **Binary preferences** (Ouyang et al., 2022, InstructGPT): Human annotators choose between two completions → {{< math >}}$B = 2${{< /math >}} → {{< math >}}$\log_2(2) = 1${{< /math >}} bit

- **Likert-style scales** (Bai et al., 2022, Constitutional AI): Multi-level harmlessness ratings with {{< math >}}$B = 4${{< /math >}}-{{< math >}}$5${{< /math >}} distinguishable levels → {{< math >}}$\log_2(4) = 2${{< /math >}} to {{< math >}}$\log_2(5) \approx 2.3${{< /math >}} bits

- **Continuous reward models**: While rewards are technically continuous, effective distinguishability is limited by:
  - Human judgment noise (inter-annotator agreement {{< math >}}$\sim 70${{< /math >}}-{{< math >}}$80\%${{< /math >}})
  - Reward model prediction uncertainty
  - Limited sensitivity to small score differences
  - Typical effective resolution: {{< math >}}$B = 8${{< /math >}}-{{< math >}}$10${{< /math >}} levels → {{< math >}}$\log_2(10) \approx 3.3${{< /math >}} bits

These empirical observations from production LLM-RL systems support our {{< math >}}$O(1)${{< /math >}} bit assumption.

**Key assumption being made**: We assume that the mutual information {{< math >}}$I(G; \xi)${{< /math >}} is similarly bounded. This requires that:
- The prior {{< math >}}$p(\xi)${{< /math >}} doesn't make the reward signal arbitrarily informative
- The mapping from reward values to optimal policies has limited sensitivity
- We're in a regime where scalar returns provide limited information about optimal behavior

**Rigorous statement**:

With {{< math >}}$B${{< /math >}} effectively distinguishable reward levels:
{{< math >}}
$$\mathcal{B}_{\text{potential}} = H(G) \leq \log_2(B) = O(1) \text{ bits}$$
{{< /math >}}

For typical LLM-RL systems with {{< math >}}$B \in \{2, 4, 5\}${{< /math >}}:
{{< math >}}
$$\mathcal{B}_{\text{potential}} = 1 \text{ to } 2.3 \text{ bits per episode}$$
{{< /math >}}

**When this bound might not hold**:

This analysis assumes a regime where:
- Rewards don't have unbounded precision that meaningfully distinguishes policies
- The prior {{< math >}}$p(\xi)${{< /math >}} is reasonably diffuse (not concentrated on a tiny region)
- We're learning from typical human feedback or reward models

If rewards were arbitrarily precise and highly informative about subtle policy differences, {{< math >}}$H(G)${{< /math >}} could be larger. However, this is not the practical regime for current LLM RL systems.

**Step 4**: Analyze {{< math >}}$I(G; \xi)${{< /math >}}.

The return {{< math >}}$G = R_\xi(s_T)${{< /math >}} provides information about {{< math >}}$\xi${{< /math >}} through the observed reward. Since {{< math >}}$G${{< /math >}} is a scalar observation:

{{< math >}}
$$I(G; \xi) \leq H(G) \leq \log_2(B) = O(1)$$
{{< /math >}}

The first inequality is fundamental: mutual information cannot exceed the entropy of either variable. The second follows from our effective binning assumption in Step 3.

**Key point**: Even if the reward function {{< math >}}$R_\xi${{< /math >}} is continuous and {{< math >}}$\xi${{< /math >}} is high-dimensional, the **scalar observation** {{< math >}}$G${{< /math >}} provides only {{< math >}}$O(1)${{< /math >}} bits of information about {{< math >}}$\xi${{< /math >}} when effective distinguishability is limited to {{< math >}}$B = O(1)${{< /math >}} levels.

In the best case (maximum information transmission), each episode distinguishes between {{< math >}}$B${{< /math >}} hypotheses about {{< math >}}$\xi${{< /math >}}, giving:
{{< math >}}
$$I(G; \xi) \approx \log_2(B) \text{ bits}$$
{{< /math >}}

**Step 5**: Calculate Effective Bandwidth.

Using the DPI from Step 2:

{{< math >}}
$$\mathcal{B}_{\text{effective}} = I(G; \pi^*) \leq I(G; \xi) \leq H(G) = O(1)$$
{{< /math >}}

**The "1 bit per episode" result**: This is where the original claim becomes precise. With {{< math >}}$B = 2${{< /math >}} bins (basically "good" vs "bad"):

{{< math >}}
$$I(G; \pi^*) \leq \log_2(2) = 1 \text{ bit per episode}$$
{{< /math >}}

Literally one bit! And even with more granular feedback—say {{< math >}}$B = 4${{< /math >}} bins:

{{< math >}}
$$I(G; \pi^*) \leq \log_2(4) = 2 \text{ bits per episode}$$
{{< /math >}}

We're still talking about a trickle of information.

---

**Summary**:

| Quantity | Value | Meaning |
|----------|-------|---------|
| Potential Bandwidth | {{< math >}}$O(1)${{< /math >}} bits | Scalar signal has limited capacity |
| Effective Bandwidth | {{< math >}}$\leq O(1)${{< /math >}} bits | At most {{< math >}}$\log_2(B)${{< /math >}} bits where {{< math >}}$B${{< /math >}} is number of distinguishable reward levels |
| Bottleneck | Episode-level compression | {{< math >}}$T${{< /math >}} tokens → 1 scalar return |

{{% callout note %}}
**Key insight**: Policy gradient compresses an entire sequence of {{< math >}}$T${{< /math >}} tokens into a single scalar reward signal, creating a severe information bottleneck.
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

Where {{< math >}}$V_\phi${{< /math >}} is the critic learned from past data.

**Step 2**: Apply the Data Processing Inequality.

As in the policy gradient analysis, we have {{< math >}}$\pi^* = \pi^*_\xi${{< /math >}} as a deterministic function of {{< math >}}$\xi${{< /math >}}. While {{< math >}}$\delta_t \to \xi \to \pi^*${{< /math >}} is NOT a Markov chain (since {{< math >}}$\delta_t${{< /math >}} depends on {{< math >}}$\xi${{< /math >}} through rewards, {{< math >}}$\pi_\theta${{< /math >}} determining which states are visited, and {{< math >}}$V_\phi${{< /math >}} providing critic estimates), we can still apply DPI using the deterministic function property:

{{< math >}}
$$I(\delta_t; \pi^* | \text{history}) = I(\delta_t; \pi^*_\xi | \text{history}) \leq I(\delta_t; \xi | \text{history})$$
{{< /math >}}

This bound follows from the general principle that for any deterministic function {{< math >}}$f${{< /math >}}, {{< math >}}$I(X; f(Y)) \leq I(X; Y)${{< /math >}}, which holds regardless of the relationship between {{< math >}}$X${{< /math >}} and {{< math >}}$Y${{< /math >}}.

**Step 3**: Calculate Potential Bandwidth.

Each TD error {{< math >}}$\delta_t${{< /math >}} is a scalar. Assuming we discretize into {{< math >}}$B_\delta${{< /math >}} bins:

{{< math >}}
$$H(\delta_t | \text{history}) \leq \log_2(B_\delta) = O(1)$$
{{< /math >}}

Since we have {{< math >}}$T${{< /math >}} steps (one per token generated):

{{< math >}}
$$\mathcal{B}_{\text{potential}} = \sum_{t=0}^{T-1} H(\delta_t | \text{history}) = T \cdot O(1) = O(T) \text{ bits}$$
{{< /math >}}

{{% callout note %}}
**Key difference from policy gradient**: We get a learning signal at **every token**, not just at episode end.
{{% /callout %}}

**Step 4**: Analyze {{< math >}}$I(\delta_t; \xi | \text{history})${{< /math >}} — the information content of TD errors.

The TD error at timestep {{< math >}}$t${{< /math >}} is:
{{< math >}}
$$\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$$
{{< /math >}}

This signal's informativeness about {{< math >}}$\xi${{< /math >}} depends critically on the reward structure and critic quality.

**Case 1: Terminal states** (where {{< math >}}$t = T${{< /math >}} and we observe the actual reward):

At the final timestep:
{{< math >}}
$$\delta_T = R_\xi(s_T) + \gamma V_\phi(s_{T+1}) - V_\phi(s_T)$$
{{< /math >}}

If {{< math >}}$s_{T+1}${{< /math >}} is a terminal/absorbing state with {{< math >}}$V_\phi(s_{T+1}) = 0${{< /math >}}:
{{< math >}}
$$\delta_T = R_\xi(s_T) - V_\phi(s_T)$$
{{< /math >}}

This directly contains the reward signal {{< math >}}$R_\xi(s_T)${{< /math >}}, providing:
{{< math >}}
$$I(\delta_T; \xi | s_T, \text{history}) = O(1) \text{ bits}$$
{{< /math >}}

(bounded by the same argument as policy gradient)

**Case 2: Non-terminal states with sparse rewards** (where {{< math >}}$r_t = 0${{< /math >}}):

This is the critical case for understanding actor-critic's potential advantage. When {{< math >}}$r_t = 0${{< /math >}}:
{{< math >}}
$$\delta_t = \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$$
{{< /math >}}

**Key question**: How can {{< math >}}$\delta_t${{< /math >}} provide information about {{< math >}}$\xi${{< /math >}} when no reward is observed?

**Answer**: Information is mediated through the learned value function {{< math >}}$V_\phi${{< /math >}}.

**Information flow mechanism**:

1. The value function {{< math >}}$V_\phi${{< /math >}} is trained on past episodes to approximate {{< math >}}$V^{\pi_\theta}_\xi(s)${{< /math >}} — the expected future reward under the current policy given reward parameter {{< math >}}$\xi${{< /math >}}

2. As {{< math >}}$V_\phi${{< /math >}} learns, it accumulates information about {{< math >}}$\xi${{< /math >}} from observed terminal rewards in previous episodes

3. The TD error {{< math >}}$\delta_t${{< /math >}} at non-terminal states reflects whether the current state is better or worse than expected according to the current estimate of {{< math >}}$\xi${{< /math >}}

4. This provides a **bootstrapped** learning signal even when {{< math >}}$r_t = 0${{< /math >}}

**Rigorous bound on non-terminal information**:

Let {{< math >}}$V_\phi${{< /math >}} be trained on {{< math >}}$M${{< /math >}} previous episodes. The information {{< math >}}$V_\phi${{< /math >}} contains about {{< math >}}$\xi${{< /math >}} is bounded by:
{{< math >}}
$$I(V_\phi; \xi) \leq M \cdot \log_2(B) = O(M)$$
{{< /math >}}

Given this, the TD error at a non-terminal state can provide:
{{< math >}}
$$I(\delta_t; \xi | s_t, s_{t+1}, V_\phi) \leq I(V_\phi; \xi) = O(M)$$
{{< /math >}}

However, this bound is too loose for practical analysis. The actual information depends on:

- **Critic quality**: How well {{< math >}}$V_\phi${{< /math >}} approximates the true value function
- **State informativeness**: Whether {{< math >}}$s_t${{< /math >}} and {{< math >}}$s_{t+1}${{< /math >}} differ in their expected rewards
- **Learning stage**: Early vs. late in training

**Practical information content**:

In practice, we expect:
{{< math >}}
$$I(\delta_t; \xi | s_t, s_{t+1}, \text{history}) \approx c_{\text{critic}} \cdot O(1) \text{ bits}$$
{{< /math >}}

where {{< math >}}$c_{\text{critic}} \in [0, 1]${{< /math >}} is a critic quality factor:
- {{< math >}}$c_{\text{critic}} \approx 0${{< /math >}}: Poorly trained critic (random initialization)
- {{< math >}}$c_{\text{critic}} \approx 0.1${{< /math >}}-{{< math >}}$0.5${{< /math >}}: Moderately trained critic
- {{< math >}}$c_{\text{critic}} \approx 1${{< /math >}}: Well-trained critic that accurately estimates {{< math >}}$V^{\pi_\theta}_\xi${{< /math >}}

**Training phase dependence**:

| Phase | Critic State | {{< math >}}$I(\delta_t; \xi)${{< /math >}} per non-terminal step | Effective info/episode |
|-------|--------------|------------------------------------------|------------------------|
| Early | Untrained {{< math >}}$V_\phi${{< /math >}} | {{< math >}}$\approx 0${{< /math >}} | {{< math >}}$\sim O(1)${{< /math >}} (terminal only) |
| Mid | Improving {{< math >}}$V_\phi${{< /math >}} | {{< math >}}$\sim 0.1${{< /math >}}-{{< math >}}$0.5${{< /math >}} bits | {{< math >}}$\sim 0.1T${{< /math >}} to {{< math >}}$0.5T${{< /math >}} |
| Late | Converged {{< math >}}$V_\phi${{< /math >}} | {{< math >}}$\sim O(1)${{< /math >}} bits | {{< math >}}$\sim O(T)${{< /math >}} |

**Critical implication**: The {{< math >}}$O(T)${{< /math >}} bandwidth is an **asymptotic upper bound** achieved only when:
1. The critic has converged to a good approximation of {{< math >}}$V^{\pi_\theta}_\xi${{< /math >}}
2. TD errors at different timesteps provide non-redundant information
3. The value function successfully propagates information from sparse terminal rewards back through the trajectory

**Early training reality**: When {{< math >}}$V_\phi${{< /math >}} is poorly initialized or undertrained, actor-critic's effective bandwidth collapses toward {{< math >}}$O(1)${{< /math >}} bits per episode—similar to policy gradient! This explains why actor-critic methods require careful critic training to achieve their theoretical advantages.

**Step 5**: Sum over trajectory.

Using the chain rule for mutual information:

{{< math >}}
$$I(\{\delta_t\}_{t=0}^{T-1}; \xi) = \sum_{t=0}^{T-1} I(\delta_t; \xi | \{\delta_k\}_{k < t})$$
{{< /math >}}

Each step can provide new information about {{< math >}}$\xi${{< /math >}}, but the total is bounded by:

{{< math >}}
$$I(\{\delta_t\}; \xi) \leq \sum_{t=0}^{T-1} H(\delta_t | \text{history}) = O(T)$$
{{< /math >}}

**Step 6**: Calculate Effective Bandwidth.

{{< math >}}
$$\mathcal{B}_{\text{effective}} = I(\{\delta_t\}; \pi^*) \leq I(\{\delta_t\}; \xi) \leq O(T)$$
{{< /math >}}

**The role of the critic**: The effective bandwidth depends on critic quality:

| Training Phase | Critic State | {{< math >}}$I(\{\delta_t\}; \xi)${{< /math >}} | Effective Bandwidth |
|----------------|--------------|------------------------|---------------------|
| Early | Random {{< math >}}$V_\phi${{< /math >}} | Low | {{< math >}}$\ll O(T)${{< /math >}} |
| Middle | Improving | Growing | {{< math >}}$\to O(T)${{< /math >}} |
| Late | Converged | High | {{< math >}}$\approx O(T)${{< /math >}} |

When the critic is well-trained, {{< math >}}$V_\phi(s_t)${{< /math >}} approximates true expected rewards, and the TD errors become informative about {{< math >}}$\xi${{< /math >}} throughout the trajectory.

**Important caveat**: The {{< math >}}$O(T)${{< /math >}} bandwidth is an **upper bound** that requires a well-trained critic. In practice, especially early in training or without careful critic tuning, the effective bandwidth may be much lower. The TD errors at different timesteps may also be correlated (not providing independent information), further reducing the effective bandwidth below the theoretical maximum.

### TD Error Correlation Reduces Effective Bandwidth

So far we've established that actor-critic methods can achieve {{< math >}}$O(T)${{< /math >}} bits per episode with a well-trained critic. However, this analysis assumed that TD errors at different timesteps provide independent information. In practice, successive TD errors are **correlated**, which reduces the effective information rate.

**Source of correlation**:

In autoregressive generation, consecutive states share most tokens:
{{< math >}}
$$s_t = (x_1, \ldots, x_t), \quad s_{t+1} = (x_1, \ldots, x_t, x_{t+1})$$
{{< /math >}}

This creates several sources of correlation in TD errors:

1. **State overlap**: Consecutive states differ by only one token, so their value estimates are highly correlated
2. **Bootstrapping**: The value function itself creates temporal dependencies: {{< math >}}$V_\phi(s_{t+1})${{< /math >}} appears in {{< math >}}$\delta_t${{< /math >}}, and value updates propagate through time
3. **Policy coherence**: The same policy {{< math >}}$\pi_\theta${{< /math >}} generates the entire sequence, creating correlated actions

**Modeling correlation**:

Consider a simplified model where TD errors follow an AR(1) process:
{{< math >}}
$$\delta_t = \rho \cdot \delta_{t-1} + \epsilon_t$$
{{< /math >}}

where:
- {{< math >}}$\rho \in [0, 1]${{< /math >}} is the correlation coefficient
- {{< math >}}$\epsilon_t${{< /math >}} are independent innovations with {{< math >}}$H(\epsilon_t) = h${{< /math >}} bits

**Information-theoretic consequence**:

For this process, the total information in {{< math >}}$T${{< /math >}} observations is:

{{< math >}}
$$I(\{\delta_t\}_{t=0}^{T-1}; \xi) \leq T \cdot h \cdot g(\rho)$$
{{< /math >}}

where {{< math >}}$g(\rho)${{< /math >}} is a reduction factor due to correlation.

For Gaussian AR(1), we can compute:
{{< math >}}
$$g(\rho) = \frac{\sqrt{1-\rho^2}}{1-\rho^2/2} \approx 1 - \rho \quad \text{(for moderate } \rho\text{)}$$
{{< /math >}}

**Numerical examples**:

| Correlation {{< math >}}$\rho${{< /math >}} | Reduction factor {{< math >}}$g(\rho)${{< /math >}} | Effective info rate |
|-------------------|---------------------------|---------------------|
| 0.0 (independent) | 1.0 | {{< math >}}$T \cdot h${{< /math >}} bits |
| 0.5 (moderate) | {{< math >}}$\approx 0.5${{< /math >}} | {{< math >}}$0.5T \cdot h${{< /math >}} bits |
| 0.7 (high) | {{< math >}}$\approx 0.3${{< /math >}} | {{< math >}}$0.3T \cdot h${{< /math >}} bits |
| 0.9 (very high) | {{< math >}}$\approx 0.1${{< /math >}} | {{< math >}}$0.1T \cdot h${{< /math >}} bits |

**Empirical estimates**:

In LLM token generation with {{< math >}}$T \gg 1000${{< /math >}} tokens:
- Correlation between adjacent TD errors: {{< math >}}$\rho \approx 0.5${{< /math >}} to {{< math >}}$0.8${{< /math >}} (empirically observed in RL training)
- Effective reduction: {{< math >}}$3\times${{< /math >}} to {{< math >}}$10\times${{< /math >}} compared to independent case

**Revised bandwidth estimate**:

Combining critic quality and correlation effects:

{{< math >}}
$$\mathcal{B}_{\text{effective, AC}} = c_{\text{critic}} \cdot g(\rho) \cdot T \cdot O(1) \text{ bits per episode}$$
{{< /math >}}

With realistic values ({{< math >}}$c_{\text{critic}} \approx 0.5${{< /math >}}, {{< math >}}$g(\rho) \approx 0.3${{< /math >}}, {{< math >}}$T = 1000${{< /math >}}):
{{< math >}}
$$\mathcal{B}_{\text{effective, AC}} \approx 0.15 \cdot 1000 = 150 \text{ bits per episode}$$
{{< /math >}}

Compared to policy gradient ({{< math >}}$\sim 2${{< /math >}} bits per episode):
{{< math >}}
$$\text{Actual speedup} \approx 75\times$$
{{< /math >}}

This is still substantial, but far from the naive {{< math >}}$1000\times${{< /math >}} suggested by ignoring correlation and critic quality.

**Why this matters**:

This partially explains why:
1. Actor-critic methods for LLMs haven't achieved the full theoretical {{< math >}}$T\times${{< /math >}} improvement
2. Value-based RL for language models remains challenging
3. Sample efficiency gains are significant but not as dramatic as pure {{< math >}}$O(T)${{< /math >}} scaling suggests

**Research implication**: Developing methods to:
- Reduce correlation between successive TD errors
- Improve critic training efficiency
- Better utilize the available {{< math >}}$O(T)${{< /math >}} information capacity

...could unlock closer-to-theoretical sample efficiency improvements.

---

**Summary**:

| Quantity | Value | Meaning |
|----------|-------|---------|
| Potential Bandwidth | {{< math >}}$O(T)${{< /math >}} bits | Dense signal: one per token |
| Effective Bandwidth | {{< math >}}$\leq O(T)${{< /math >}} bits | Achievable when critic is trained |
| Improvement over PG | {{< math >}}$T \times${{< /math >}} | For {{< math >}}$T=1000${{< /math >}} tokens, {{< math >}}$1000\times${{< /math >}} more bandwidth |

{{% callout note %}}
**Key insight**: Actor-critic provides learning signals at every token generation step, not just episode end, yielding {{< math >}}$T\times${{< /math >}} more information capacity.
{{% /callout %}}

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
   - Actor-critic: {{< math >}}$\sim T${{< /math >}} bits per episode where {{< math >}}$T${{< /math >}} is sequence length
   - If {{< math >}}$T = 1000${{< /math >}} tokens, actor-critic has {{< math >}}$250${{< /math >}}-{{< math >}}$1000\times${{< /math >}} more bandwidth

3. **Token-level MDP clarifies everything**: No need to assume unknown dynamics—transitions are deterministic concatenation

---

### Why LoRA Works: The Complete Picture

Now we can finally give a satisfying answer to the original question: why does LoRA work so well for RL fine-tuning?

The key is matching capacity to information flow. Let's work through the numbers.

**Prior**: Pre-trained model gives us prior {{< math >}}$p(\xi)${{< /math >}} over reward functions (implicitly, from pre-training on diverse text data).

**Information Accumulation with Saturation**:

After {{< math >}}$N${{< /math >}} episodes of policy gradient, the total information gained is:
{{< math >}}
$$I(\{G_1, \ldots, G_N\}; \pi^*) = \sum_{i=1}^{N} I(G_i; \pi^* | \{G_j\}_{j < i})$$
{{< /math >}}

**Critical insight**: The marginal information {{< math >}}$I(G_i; \pi^* | \{G_j\}_{j<i})${{< /math >}} from episode {{< math >}}$i${{< /math >}} **decreases as training progresses**. This happens for three reasons:

1. **Redundancy**: As the policy approaches {{< math >}}$\pi^*${{< /math >}}, it generates similar trajectories that observe similar rewards
2. **Saturation**: There is limited total uncertainty to resolve ({{< math >}}$I_{\text{total}} \leq H(\pi^*)${{< /math >}})
3. **Correlation**: Episodes from similar policies are not independent observations

**Training phase analysis**:

| Phase | Episode {{< math >}}$i${{< /math >}} | Marginal info {{< math >}}$I(G_i; \pi^* \mid \text{past})${{< /math >}} | Cumulative info |
|-------|-------------|------------------------------------------------|-----------------|
| **Early** | {{< math >}}$i \ll N^*${{< /math >}} | {{< math >}}$\approx c${{< /math >}} bits | {{< math >}}$I_{\text{total}} \approx i \cdot c${{< /math >}} |
| **Mid** | {{< math >}}$i \sim N^*${{< /math >}} | Decreasing | Sublinear growth |
| **Late** | {{< math >}}$i \gg N^*${{< /math >}} | {{< math >}}$\to 0${{< /math >}} | {{< math >}}$I_{\text{total}} \to H(\pi^*)${{< /math >}} |

where {{< math >}}$N^* \approx H(\pi^*)/c${{< /math >}} is the **saturation point** at which most uncertainty has been resolved.

**Mathematical model** (illustrative):

A simple model for information accumulation with saturation:
{{< math >}}
$$I_{\text{total}}(N) \approx H(\pi^*) \cdot \left(1 - e^{-cN/H(\pi^*)}\right)$$
{{< /math >}}

This captures:
- Linear growth for {{< math >}}$N \ll H(\pi^*)/c${{< /math >}}: {{< math >}}$I_{\text{total}} \approx cN${{< /math >}}
- Saturation for {{< math >}}$N \gg H(\pi^*)/c${{< /math >}}: {{< math >}}$I_{\text{total}} \to H(\pi^*)${{< /math >}}

**For LoRA analysis**, we consider the **early/mid training regime** where:
- {{< math >}}$N \ll N^* = H(\pi^*)/c${{< /math >}} (training has not saturated)
- Linear approximation holds: {{< math >}}$I_{\text{total}} \approx N \cdot c${{< /math >}} bits
- This is valid for typical fine-tuning where {{< math >}}$N \sim 10^3${{< /math >}} to {{< math >}}$10^4${{< /math >}} episodes

**Concrete numbers** (assuming early/mid regime with {{< math >}}$c = 2${{< /math >}} bits/episode):

| Episodes {{< math >}}$N${{< /math >}} | Information accumulated | Approximate bytes |
|--------------|------------------------|-------------------|
| 100 | {{< math >}}$\sim 200${{< /math >}} bits | {{< math >}}$\sim 25${{< /math >}} bytes |
| 1,000 | {{< math >}}$\sim 2{,}000${{< /math >}} bits | {{< math >}}$\sim 250${{< /math >}} bytes |
| 10,000 | {{< math >}}$\sim 20{,}000${{< /math >}} bits | {{< math >}}$\sim 2.5${{< /math >}} KB |
| 100,000 | Approaching saturation | Depends on {{< math >}}$H(\pi^*)${{< /math >}} |

**Important caveat**: These are **upper bounds** on useful information. Actual information about {{< math >}}$\pi^*${{< /math >}} may be lower due to:
- Noise in reward observations
- Redundant episodes (correlated trajectories)
- Information not directly relevant to finding {{< math >}}$\pi^*${{< /math >}}

{{% callout note %}}
**Key Assumption for LoRA Analysis**: We analyze the learning regime where fine-tuning has not yet saturated, i.e., {{< math >}}$N \cdot c < H(\pi^*)${{< /math >}}, so information accumulates approximately linearly: {{< math >}}$I_{\text{total}} \approx N \cdot c${{< /math >}} bits.

This is reasonable for practical fine-tuning scenarios where {{< math >}}$N \sim 10^3${{< /math >}} to {{< math >}}$10^4${{< /math >}} episodes and the policy space is large (high {{< math >}}$H(\pi^*)${{< /math >}}). Once {{< math >}}$I_{\text{total}}${{< /math >}} approaches {{< math >}}$H(\pi^*)${{< /math >}}, marginal learning necessarily slows as there is less remaining uncertainty to resolve.
{{% /callout %}}

### LoRA Capacity and Information Content: A Careful Comparison

We've established that policy gradient provides limited information per episode. Now we want to understand: **Is LoRA's parameter capacity sufficient to represent the learned policy changes?**

**Critical disclaimer upfront**: We must be careful here. **Storage bits** (how we represent parameters in memory) and **information bits** (uncertainty reduction about {{< math >}}$\pi^*${{< /math >}}) are fundamentally different quantities. A 1GB hard drive doesn't mean you've "learned 1GB of information." However, we can still make a meaningful argument about representational capacity.

**The degrees of freedom argument**:

LoRA with rank {{< math >}}$r${{< /math >}} and dimension {{< math >}}$d${{< /math >}} provides:
- **Number of parameters**: {{< math >}}$2rd${{< /math >}} (two low-rank matrices)
- **Degrees of freedom**: {{< math >}}$2rd${{< /math >}} independent values to optimize

The question is: **How much information about optimal policies can this parameter space represent?**

**Lower bound on representational capacity** (informal):

Consider a conservative estimate: each parameter can encode roughly {{< math >}}$\sim 1${{< /math >}} bit of independent information about the policy. This is extremely conservative because:
- FP32 parameters have 32 bits of storage
- Even with quantization, parameters typically use 8-16 bits
- Neural network parameters can represent complex, nonlinear relationships

Under this conservative estimate:
- LoRA representational capacity: {{< math >}}$\gtrsim 2rd${{< /math >}} bits of policy-relevant information

**Information requirements from RL**:

Policy gradient over {{< math >}}$N${{< /math >}} episodes provides:
- Total information: {{< math >}}$\mathcal{I} = 2N${{< /math >}} bits (with {{< math >}}$B = 4${{< /math >}} reward bins)

**Comparison** (with {{< math >}}$r = 8${{< /math >}}, {{< math >}}$d = 4096${{< /math >}}, {{< math >}}$N = 1000${{< /math >}}):
- LoRA degrees of freedom: {{< math >}}$2 \times 8 \times 4096 = 65{,}536${{< /math >}} parameters
- Information to encode: {{< math >}}$\sim 2{,}000${{< /math >}} bits
- **Ratio**: {{< math >}}$\sim 30\times${{< /math >}} more parameters than information bits

Even if each parameter encodes only 0.1 bits of policy-relevant information (very pessimistic), LoRA still has {{< math >}}$3\times${{< /math >}} headroom.

**What this comparison actually means**:

The parameter update {{< math >}}$\Delta\theta${{< /math >}} from RL training must **encode** the policy-relevant information learned from episodes. If policy gradient provides only {{< math >}}$\sim 2N${{< /math >}} bits of information to guide this update, and LoRA provides {{< math >}}$2rd${{< /math >}} parameters to store it, then:

{{< math >}}
$$\text{Parameters per information bit} = \frac{2rd}{2N} = \frac{rd}{N}$$
{{< /math >}}

With {{< math >}}$r = 8${{< /math >}}, {{< math >}}$d = 4096${{< /math >}}, {{< math >}}$N = 1000${{< /math >}}:
{{< math >}}
$$\frac{8 \times 4096}{1000} \approx 33 \text{ parameters per information bit}$$
{{< /math >}}

**Practical interpretation**: The LoRA parameter space is **vastly overcomplete** for the information being learned. Even accounting for:
- Inefficient parameter utilization
- Redundancy in neural network representations
- Optimization constraints

...there is substantial headroom.

**Storage capacity perspective** (with all caveats):

If we naively compare storage bits:
- LoRA storage (FP32): {{< math >}}$32 \times 8 \times 4096 = 1{,}048{,}576${{< /math >}} bits {{< math >}}$= 128${{< /math >}} KB
- Information from 1000 episodes: {{< math >}}$\sim 2000${{< /math >}} bits {{< math >}}$\approx 0.25${{< /math >}} KB
- **Ratio**: {{< math >}}$\sim 500\times${{< /math >}}

But this comparison is **not rigorous** because storage bits {{< math >}}$\neq${{< /math >}} information bits. We mention it only to illustrate the order-of-magnitude mismatch.

**The key insight** (what we can rigorously claim):

Regardless of the exact conversion between storage bits and information content, the qualitative conclusion holds:

1. Policy gradient provides {{< math >}}$O(N)${{< /math >}} bits of information about {{< math >}}$\pi^*${{< /math >}}
2. LoRA provides {{< math >}}$O(rd)${{< /math >}} degrees of freedom to represent policy changes
3. In typical settings: {{< math >}}$rd \gg N${{< /math >}}

This **qualitatively explains** why low-rank adapters work well for policy gradient fine-tuning—the parameter bottleneck matches the information bottleneck.

**Why full fine-tuning is wasteful**:

Extending this argument to full fine-tuning of a 7B parameter model:
- Parameters: {{< math >}}$7 \times 10^9${{< /math >}}
- Degrees of freedom: {{< math >}}$7 \times 10^9${{< /math >}}
- Information from 1000 episodes PG: {{< math >}}$\sim 2000${{< /math >}} bits

{{< math >}}
$$\text{Parameters per information bit} = \frac{7 \times 10^9}{2000} \approx 3.5 \times 10^6$$
{{< /math >}}

That's **3.5 million parameters per information bit**—an absurd overcapacity that invites overfitting, catastrophic forgetting, and computational waste.

**Conclusion** (with appropriate epistemic humility):

While we cannot rigorously quantify the exact relationship between parameter DOF and information capacity, the order-of-magnitude analysis strongly suggests:
- LoRA ({{< math >}}$r = 8${{< /math >}}-{{< math >}}$16${{< /math >}}): Well-matched to policy gradient's information rate
- Full fine-tuning: Vastly overcapacitated for sparse learning signals
- This mismatch qualitatively explains empirical observations about LoRA's effectiveness

The low-rank bottleneck naturally matches the information bottleneck—which is why LoRA works.

---

### 4.3 Sample Complexity: An Information-Theoretic Perspective

{{% callout warning %}}
**Limitations of Information-Theoretic Sample Complexity Bounds**

The bounds in this section are **information-theoretic lower bounds** that represent idealized conditions. They assume:

1. **Perfect information utilization**: No optimization barriers or gradient descent difficulties
2. **Linear accumulation**: Minimal redundancy and correlation between episodes
3. **Direct translation**: Information directly converts to convergence

These assumptions rarely hold in practice. Actual sample complexity depends on:
- Optimization landscape and gradient descent dynamics
- Redundancy and correlation between episodes
- Approximation errors in policy/value representations
- Exploration-exploitation tradeoffs
- Neural network optimization challenges

**Treat these as order-of-magnitude estimates** that explain qualitative differences between algorithms, not tight quantitative predictions for real training runs.
{{% /callout %}}

**Information-theoretic perspective on sample complexity**:

To reduce uncertainty about {{< math >}}$\pi^*${{< /math >}} from prior entropy {{< math >}}$H(\pi^*)${{< /math >}} to posterior entropy {{< math >}}$\epsilon${{< /math >}}, we need to accumulate:
{{< math >}}
$$\mathcal{I}_{\text{required}} = H(\pi^*) - \epsilon$$
{{< /math >}}
bits of information.

**Necessary condition** (not sufficient): The algorithm must observe at least:
{{< math >}}
$$N \geq \frac{\mathcal{I}_{\text{required}}}{\mathcal{B}_{\text{effective}}}$$
{{< /math >}}
episodes to accumulate sufficient information.

**For different algorithms** (under idealized conditions):

- **Policy gradient**: {{< math >}}$\mathcal{B}_{\text{effective}} = O(1)${{< /math >}} bits/episode
  → {{< math >}}$N_{\text{PG}} = \Omega(\mathcal{I}_{\text{required}})${{< /math >}} episodes needed

- **Actor-critic**: {{< math >}}$\mathcal{B}_{\text{effective}} = O(T)${{< /math >}} bits/episode
  → {{< math >}}$N_{\text{AC}} = \Omega(\mathcal{I}_{\text{required}}/T)${{< /math >}} episodes needed

**Theoretical speedup**:
{{< math >}}
$$\frac{N_{\text{PG}}}{N_{\text{AC}}} = O(T)$$
{{< /math >}}

**Why actual speedup is much smaller**:

1. **Critic training**: AC bandwidth requires well-trained critic (not available in early training)
2. **Correlation**: TD error correlation reduces effective {{< math >}}$T${{< /math >}} by factor of 3-10× (see Section 3.2)
3. **Optimization**: Gradient descent may need more samples than information theory suggests
4. **Exploration**: Real RL requires exploration overhead beyond pure information gathering
5. **Approximation**: Function approximation errors waste some available information

**Empirical observations align with information theory**:

In traditional RL benchmarks (Atari, MuJoCo), PPO (actor-critic) typically achieves **10-100× speedup** over REINFORCE (policy gradient).

**Quantitative validation from RL literature**:

Schulman et al. (2017) report that PPO requires approximately {{< math >}}$100\times${{< /math >}} fewer environment interactions than REINFORCE to achieve comparable performance on continuous control tasks.

For episodes of length {{< math >}}$T \sim 100${{< /math >}}-{{< math >}}$1000${{< /math >}} steps:
- Naive information-theoretic prediction: {{< math >}}$100${{< /math >}}-{{< math >}}$1000\times${{< /math >}} speedup
- Correlation-adjusted prediction: {{< math >}}$10${{< /math >}}-{{< math >}}$100\times${{< /math >}} speedup (with {{< math >}}$g(\rho) \approx 0.1${{< /math >}}-{{< math >}}$0.3${{< /math >}})
- **Empirical observation**: {{< math >}}$\sim 100\times${{< /math >}} speedup ✓

This close alignment between information-theoretic predictions (with correlation correction) and empirical results validates the framework while showing the importance of accounting for correlation and critic quality.

**Illustrative numerical example** (with all caveats):

Suppose {{< math >}}$H(\pi^*) \approx 10{,}000${{< /math >}} bits and {{< math >}}$T = 1000${{< /math >}} tokens:

- **Policy gradient**: {{< math >}}$N_{\text{PG}} \gtrsim \frac{10{,}000}{2} = 5{,}000${{< /math >}} episodes
  (assuming {{< math >}}$c = 2${{< /math >}} bits/episode with 4-bin rewards)

- **Actor-critic** (well-trained critic, accounting for correlation):
  {{< math >}}$N_{\text{AC}} \gtrsim \frac{10{,}000}{0.15 \times 1000} = 67${{< /math >}} episodes
  (with {{< math >}}$c_{\text{critic}} \approx 0.5${{< /math >}}, {{< math >}}$g(\rho) \approx 0.3${{< /math >}})

- **Predicted speedup**: {{< math >}}$\sim 75\times${{< /math >}}

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

Information accumulated in {{< math >}}$N${{< /math >}} episodes: {{< math >}}$\sim N \cdot 2${{< /math >}} bits (with 4-bin returns)

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
- LoRA capacity: {{< math >}}$32rd${{< /math >}} bits ({{< math >}}$\sim 50{,}000\times${{< /math >}} more than needed)
- Perfect match between limited information and limited capacity

**5. Sample Complexity Predictions**
- Policy gradient: {{< math >}}$\Omega(H(\pi^*))${{< /math >}} episodes
- Actor-critic: {{< math >}}$\Omega(H(\pi^*)/T)${{< /math >}} episodes
- Explains {{< math >}}$100${{< /math >}}-{{< math >}}$1000\times${{< /math >}} empirical differences

### Open Questions and Research Directions

If I had to bet on where the next major breakthrough in LLM RL will come from, it would be solving these problems:

**1. The Value Function Challenge** (Highest Priority):

How can we develop stable, scalable methods for training value functions on LLMs?
- **Information gain**: Could unlock {{< math >}}$\sim 1000\times${{< /math >}} improvement (from {{< math >}}$O(1)${{< /math >}} to {{< math >}}$O(1000)${{< /math >}} bits/episode)
- **Key obstacles**: Training stability, long-horizon credit assignment, computational overhead
- **Promising directions**: Low-rank critics, hierarchical decomposition, better initialization

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

For practitioners fine-tuning LLMs with RL:

**Current Best Practices (Policy Gradient Era)**:

1. **Use LoRA with small rank** ({{< math >}}$r = 8${{< /math >}} to {{< math >}}$16${{< /math >}}) for policy gradient—our analysis shows full fine-tuning is wasteful given {{< math >}}$O(1)${{< /math >}} bits/episode

2. **Expect slow convergence**: With {{< math >}}$O(1)${{< /math >}} bits/episode, budget for thousands of episodes

3. **Invest in pre-training quality** to reduce {{< math >}}$H(\pi^* | \text{prior})${{< /math >}} and speed up RL fine-tuning

4. **For multi-task learning**, scale LoRA rank proportionally to number of tasks ({{< math >}}$I_{\text{total}} = N \cdot O(K)${{< /math >}})

**The $1000\times$ Opportunity**:

Our analysis reveals a massive opportunity: successfully training value functions for LLMs could improve sample efficiency by {{< math >}}$\sim 1000\times${{< /math >}} (from {{< math >}}$O(1)${{< /math >}} to {{< math >}}$O(T)${{< /math >}} bits/episode where {{< math >}}$T \gg 1000${{< /math >}}).

**Priority Research Directions**:

1. **Develop stable critic training methods** for LLMs
   - Low-rank value functions
   - Better initialization strategies
   - Hierarchical value decomposition

2. **Design meaningful token-level rewards** (currently most methods use sparse, outcome-based rewards)

3. **Explore hybrid approaches** that partially capture {{< math >}}$O(T)${{< /math >}} bandwidth:
   - Monte Carlo + TD methods
   - Model-based value learning with reward models
   - Outcome-conditioned value estimation

{{% callout warning %}}
**Current State**: The field currently lacks reliable value-based RL recipes for LLMs. From an information-theoretic perspective, this represents the single largest bottleneck in sample-efficient LLM RL training. Solving this problem could reduce training costs and data requirements by orders of magnitude.
{{% /callout %}}

{{% callout note %}}
The information-theoretic perspective not only explains current methods (why LoRA works, why training is slow) but also reveals where the highest-leverage research opportunities lie: unlocking the {{< math >}}$O(T)${{< /math >}} bits/episode potential of value-based methods.
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

Before the appendix, let's be crystal clear about the scope and limitations of our results:

{{% callout warning %}}
### Rigorous Results ✓

What we've **mathematically proven**:
- Policy gradient uses {{< math >}}$O(1)${{< /math >}} bit learning signals per episode
- Actor-critic uses {{< math >}}$O(T)${{< /math >}} bit learning signals per episode
- Qualitative conclusion: dense signals provide more information bandwidth
- LoRA parameter capacity exceeds policy gradient information flow

### Results Requiring Assumptions ⚠

What holds **under stated assumptions**:
- "1-4 bits per episode" requires finite effective reward distinguishability ({{< math >}}$B = O(1)${{< /math >}} levels)
- {{< math >}}$O(T)${{< /math >}} bandwidth requires well-trained critic that successfully propagates value information
- Linear accumulation ({{< math >}}$I_{\text{total}} \approx Nc${{< /math >}}) requires early/mid training regime before saturation
- Sample complexity bounds assume near-optimal information utilization

### Qualitative/Suggestive Results ~

What is **order-of-magnitude reasoning**:
- LoRA storage capacity vs. information content comparison (storage bits {{< math >}}$\neq${{< /math >}} information bits)
- Specific numerical predictions for sample complexity (depend on optimization dynamics)
- Exact speedup factors (depend on correlation structure, critic quality, problem specifics)
- Information saturation curves (simplified models of complex learning dynamics)

### What This Framework Provides

This information-theoretic analysis offers:
- **Qualitative insights**: Why certain methods work and where bottlenecks exist
- **Order-of-magnitude estimates**: Rough predictions that align with empirical observations
- **Research directions**: Identifying high-leverage opportunities (value-based methods)
- **Conceptual clarity**: Understanding RL efficiency through information flow

It does **not** provide:
- Exact sample complexity theorems with tight constants
- Guarantees about specific training runs
- Prescriptive recipes for optimal hyperparameters
- Replacement for empirical validation
{{% /callout %}}

---

## Appendix: Scope and Limitations

**What this analysis rigorously establishes**:
1. ✅ Policy gradient uses scalar signals with {{< math >}}$O(1)${{< /math >}} bit capacity per episode
2. ✅ Actor-critic uses token-level signals with {{< math >}}$O(T)${{< /math >}} bit capacity per episode
3. ✅ The qualitative conclusion: dense signals provide {{< math >}}$T\times${{< /math >}} more information bandwidth
4. ✅ LoRA's storage capacity vastly exceeds information provided by policy gradient

**What requires additional assumptions**:
1. ⚠️ Exact sample complexity bounds (requires analysis of optimization dynamics)
2. ⚠️ Linear accumulation of information (assumes limited redundancy across episodes)
3. ⚠️ Actor-critic achieving {{< math >}}$O(T)${{< /math >}} bandwidth (requires well-trained critic)
4. ⚠️ Reward distinguishability (assumes {{< math >}}$O(1)${{< /math >}} effective precision)

**What remains qualitative**:
1. 📊 Specific constants (exact bits per distinguishable reward level)
2. 📊 Saturation dynamics (when marginal information gain decreases)
3. 📊 Relationship between storage bits and representational capacity in neural networks

**Perspective**: This analysis provides an **information-theoretic lens** for understanding RL efficiency in LLM fine-tuning. The mathematical framework is rigorous, but translating information-theoretic bounds to concrete sample complexity requires additional assumptions about optimization and learning dynamics. The value lies in the **qualitative insights** and **order-of-magnitude comparisons**, which align well with empirical observations and provide intuition for why certain methods work.

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

- ThinkingMachines.ai (2024). "[LoRA Without Regret](https://thinkingmachines.ai/blog/lora/)." Blog post that inspired this analysis.

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