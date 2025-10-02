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

**Assumption (Unique Optimum)**: We assume that for each {{< math >}}$\xi${{< /math >}}, the optimal policy {{< math >}}$\pi^*_\xi${{< /math >}} is unique. This holds generically when:
- The action space is continuous (as in LLMs with softmax over large vocabularies)
- We use lexicographic tie-breaking in discrete cases
- The reward function has sufficient structure to avoid exact ties

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

A special case that we'll use frequently is when {{< math >}}$Z = f(Y)${{< /math >}} is a deterministic function of {{< math >}}$Y${{< /math >}}. This implies the Markov chain {{< math >}}$X \to Y \to Z${{< /math >}}, because knowing {{< math >}}$Y${{< /math >}} fully determines {{< math >}}$Z${{< /math >}}, making {{< math >}}$Z${{< /math >}} conditionally independent of {{< math >}}$X${{< /math >}}. The Data Processing Inequality therefore applies directly:

{{< math >}}
$$I(X; f(Y)) \leq I(X; Y)$$
{{< /math >}}

**Intuition**: Applying a deterministic function {{< math >}}$f${{< /math >}} can only reduce or preserve information, never create it.

**Application to RL**: Since the optimal policy {{< math >}}$\pi^* = \pi^*_\xi${{< /math >}} is a deterministic function of the reward parameter {{< math >}}$\xi${{< /math >}}, we can establish:

{{< math >}}
$$I(S; \pi^*) = I(S; \pi^*_\xi) \leq I(S; \xi)$$
{{< /math >}}

**Proof**: The equality {{< math >}}$I(S; \pi^*) = I(S; \pi^*_\xi)${{< /math >}} holds by definition since {{< math >}}$\pi^* = \pi^*_\xi${{< /math >}}. For the inequality, since {{< math >}}$\pi^*_\xi${{< /math >}} is a deterministic function of {{< math >}}$\xi${{< /math >}}, we have {{< math >}}$I(S; \pi^* | \xi) = 0${{< /math >}} (knowing {{< math >}}$\xi${{< /math >}} fully determines {{< math >}}$\pi^*${{< /math >}}). By the chain rule:

{{< math >}}
$$I(S; \pi^*, \xi) = I(S; \xi) + I(S; \pi^* | \xi) = I(S; \xi)$$
{{< /math >}}

Also, {{< math >}}$I(S; \pi^*, \xi) \geq I(S; \pi^*)${{< /math >}} since observing more variables cannot decrease mutual information. Therefore {{< math >}}$I(S; \pi^*) \leq I(S; \xi)${{< /math >}}.

This bound holds despite {{< math >}}$S${{< /math >}} depending on both {{< math >}}$\xi${{< /math >}} (through rewards) and {{< math >}}$\pi_\theta${{< /math >}} (through trajectory generation)—it relies only on {{< math >}}$\pi^*${{< /math >}} being uniquely determined by {{< math >}}$\xi${{< /math >}}.

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

**Key observation**: This is NOT a Markov chain {{< math >}}$G \to \xi \to \pi^*${{< /math >}} because {{< math >}}$G${{< /math >}} depends on both {{< math >}}$\xi${{< /math >}} (which rewards) and {{< math >}}$\pi_\theta${{< /math >}} (which trajectory), while {{< math >}}$\pi^*${{< /math >}} depends only on {{< math >}}$\xi${{< /math >}}.

**Step 2**: Apply the correct form of the Data Processing Inequality.

Since {{< math >}}$\pi^* = \pi^*_\xi${{< /math >}} is a deterministic function of {{< math >}}$\xi${{< /math >}}, we can apply the DPI:

For any random variables {{< math >}}$X${{< /math >}} and {{< math >}}$Y${{< /math >}}, and deterministic function {{< math >}}$f${{< /math >}}:

{{< math >}}
$$I(X; f(Y)) \leq I(X; Y)$$
{{< /math >}}

**Proof**: Since {{< math >}}$f${{< /math >}} is deterministic, {{< math >}}$f(Y)${{< /math >}} is conditionally independent of {{< math >}}$X${{< /math >}} given {{< math >}}$Y${{< /math >}}: {{< math >}}$p(f(Y)|X,Y) = p(f(Y)|Y)${{< /math >}}. This establishes the Markov chain {{< math >}}$X \to Y \to f(Y)${{< /math >}}. By the Data Processing Inequality proven above (Section 2.4):

{{< math >}}
$$I(X; f(Y)) \leq I(X; Y)$$
{{< /math >}}

Applied to our setting with {{< math >}}$X = G${{< /math >}} and {{< math >}}$Y = \xi${{< /math >}}:

{{< math >}}
$$I(G; \pi^*) = I(G; \pi^*_\xi) \leq I(G; \xi)$$
{{< /math >}}

This bound is rigorous and does not require a Markov chain assumption.

**Step 3**: Calculate Potential Bandwidth.

Here's the critical observation: the return {{< math >}}$G${{< /math >}} is just a single scalar! Your language model generates maybe 1000 tokens, forming a rich trajectory through sequence space, and at the end you compress all of that down to one number. How much information can possibly survive that brutal compression?

The return depends on:
- The prior {{< math >}}$\xi \sim p(\xi)${{< /math >}}
- The policy {{< math >}}$\pi_\theta${{< /math >}} (determines which sequences are generated)
- The generated sequence {{< math >}}$s_T${{< /math >}}

The entropy of this scalar signal is:

{{< math >}}
$$H(G) = H(R_\xi(s_T))$$
{{< /math >}}

where the randomness comes from both {{< math >}}$\xi${{< /math >}} and the sequence generation.

**Practical bound on return entropy**: In practice, reward signals have **limited effective precision** due to:
- Human feedback is often categorical (e.g., binary preferences, Likert scales)
- Reward models output bounded scores with finite precision
- Noise and stochasticity in reward observations

**Empirical observation**: Most RL systems use rewards that can be meaningfully distinguished into {{< math >}}$B = 2${{< /math >}} to {{< math >}}$B = 10${{< /math >}} levels. For example, with {{< math >}}$B = 5${{< /math >}} bins (very negative, negative, neutral, positive, very positive):

{{< math >}}
$$H(G) \lesssim \log_2(5) \approx 2.32 \text{ bits}$$
{{< /math >}}

More rigorously, even if rewards are continuous, the **mutual information** {{< math >}}$I(G; \xi)${{< /math >}} is bounded by the effective distinguishability of reward signals. With {{< math >}}$B${{< /math >}} effectively distinguishable reward levels:

{{< math >}}
$$I(G; \xi) \leq \log_2(B) = O(1)$$
{{< /math >}}

This bound captures the practical information content of scalar reward signals: with {{< math >}}$B = 2${{< /math >}} to {{< math >}}$10${{< /math >}} distinguishable levels:

{{< math >}}
$$\mathcal{B}_{\text{potential}} = H(G) = 1\text{-}3.3 \text{ bits per episode}$$
{{< /math >}}

**Step 4**: Analyze {{< math >}}$I(G; \xi)${{< /math >}}.

The return {{< math >}}$G = R_\xi(s_T)${{< /math >}} is a single scalar observation that depends on:
- The reward function {{< math >}}$R_\xi${{< /math >}} (what we're learning about)
- The generated sequence {{< math >}}$s_T${{< /math >}} (determined by policy)

This is a many-to-one mapping: the entire sequence of {{< math >}}$T${{< /math >}} tokens is compressed into one scalar value.

By the data processing inequality (information can only decrease through compression):

{{< math >}}
$$I(G; \xi) \leq H(G)$$
{{< /math >}}

In fact, they are equal when the return is an invertible function of {{< math >}}$\xi${{< /math >}} given {{< math >}}$s_T${{< /math >}}. But the effective constraint is:

{{< math >}}
$$I(G; \xi) \leq H(G) = O(1)$$
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

**Step 4**: Analyze {{< math >}}$I(\delta_t; \xi | \text{history})${{< /math >}}.

The TD error contains:

{{< math >}}
$$\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$$
{{< /math >}}

In the sparse reward case (common for language models), {{< math >}}$r_t = 0${{< /math >}} except at terminal states. But even when {{< math >}}$r_t = 0${{< /math >}}, the TD error provides information about {{< math >}}$\xi${{< /math >}} through the value function updates.

**For terminal states** ({{< math >}}$t = T${{< /math >}}):

{{< math >}}
$$\delta_T = R_\xi(s_T) - V_\phi(s_T)$$
{{< /math >}}

This directly observes the reward, giving:

{{< math >}}
$$I(\delta_T; \xi | s_T) = O(1)$$
{{< /math >}}

**For non-terminal states**: The value function {{< math >}}$V_\phi(s_t)${{< /math >}} estimates expected future rewards under the current policy. As the critic learns, it accumulates information about {{< math >}}$\xi${{< /math >}} from past episodes.

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

**Correlation between TD errors**: In practice, TD errors at successive timesteps are often correlated, particularly in long sequences. This correlation reduces the effective information rate compared to the {{< math >}}$T \cdot O(1)${{< /math >}} upper bound.

**Rigorous bound**: Even with correlation, the information still scales as {{< math >}}$O(T)${{< /math >}}, but with a reduced constant. If {{< math >}}$\delta_t = \rho \delta_{t-1} + \epsilon_t${{< /math >}} where innovations {{< math >}}$\epsilon_t${{< /math >}} are independent with {{< math >}}$H(\epsilon_t) = c${{< /math >}} bits, then:

{{< math >}}
$$I(\{\delta_t\}; \xi) \leq T \cdot c \cdot f(\rho)$$
{{< /math >}}

where {{< math >}}$f(\rho) < 1${{< /math >}} is a decreasing function of correlation strength. For example, with {{< math >}}$\rho = 0.7${{< /math >}}, the effective constant might be reduced by a factor of {{< math >}}$2${{< /math >}}-{{< math >}}$5\times${{< /math >}} compared to independent observations.

This correlation arises because:
- Consecutive states share most tokens (only one token differs)
- Value estimates propagate through bootstrapping
- Policy changes affect multiple timesteps similarly

While the asymptotic scaling remains {{< math >}}$O(T)${{< /math >}}, the reduced constant factor partially explains why actor-critic methods for LLMs haven't achieved the full theoretical improvement over policy gradient. Additionally, **optimization difficulties** in training the critic further reduce practical gains.

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

**Information Accumulation**:

After {{< math >}}$N${{< /math >}} episodes of policy gradient, total information gained:

{{< math >}}
$$I(\{G_1, \ldots, G_N\}; \pi^*) = \sum_{i=1}^{N} I(G_i; \pi^* | \{G_j\}_{j < i})$$
{{< /math >}}

**Key insight**: Each term {{< math >}}$I(G_i; \pi^* | \{G_j\}_{j < i})${{< /math >}} represents the **marginal information** from episode {{< math >}}$i${{< /math >}}. This decreases as training progresses due to:
1. **Redundancy**: Later episodes may visit similar states and observe similar rewards
2. **Saturation**: As {{< math >}}$\pi_\theta${{< /math >}} approaches {{< math >}}$\pi^*${{< /math >}}, less new information is gained
3. **Correlation**: Episodes generated by the same (or similar) policy are not independent

**Bounding total information**: We have two constraints:

1. **Per-episode bound**: Each return provides at most {{< math >}}$O(1)${{< /math >}} bits about {{< math >}}$\pi^*${{< /math >}}
2. **Total uncertainty bound**: {{< math >}}$I_{\text{total}} \leq H(\pi^*)${{< /math >}} (cannot learn more than initial uncertainty)

**Practical regime**: In early/mid training where {{< math >}}$N \cdot O(1) < H(\pi^*)${{< /math >}}, marginal information per episode remains roughly {{< math >}}$O(1)${{< /math >}}, giving:

{{< math >}}
$$I_{\text{total}} \approx N \cdot c \text{ bits}$$
{{< /math >}}

where {{< math >}}$c \in [1, 4]${{< /math >}} depends on reward distinguishability. As training saturates, marginal gains decrease.

However, we also have the constraint that we cannot learn more than the initial uncertainty:

{{< math >}}
$$I_{\text{total}} \leq H(\pi^*)$$
{{< /math >}}

{{% callout note %}}
**Key Assumption for LoRA Analysis**: We analyze the learning regime where fine-tuning has not yet fully converged to {{< math >}}$\pi^*${{< /math >}}, i.e., {{< math >}}$$N \cdot O(1) < H(\pi^*)$${{< /math >}}, so the information accumulated is approximately {{< math >}}$N \cdot O(1)${{< /math >}} bits.

This is reasonable for practical fine-tuning scenarios where {{< math >}}$N \sim 10^3$ to $10^4${{< /math >}} episodes and the policy space is large. Once {{< math >}}$I_{\text{total}}${{< /math >}} approaches {{< math >}}$H(\pi^*)${{< /math >}}, the learning rate necessarily slows as there is less remaining uncertainty to resolve.
{{% /callout %}}

**Concrete numbers**: With {{< math >}}$c = 2${{< /math >}} bits per episode (4-level rewards):
- After {{< math >}}$N = 1000${{< /math >}} episodes: {{< math >}}$\approx 2000${{< /math >}} bits {{< math >}}$\approx 250${{< /math >}} bytes
- After {{< math >}}$N = 10000${{< /math >}} episodes: {{< math >}}$\approx 20000${{< /math >}} bits {{< math >}}$\approx 2.5${{< /math >}} KB

**Important caveat**: These are **upper bounds** on useful information. Actual information about {{< math >}}$\pi^*${{< /math >}} may be less due to noise and redundancy in episodes.

**LoRA Capacity Analysis**:

Rank-{{< math >}}$r${{< /math >}} adapter for dimension-{{< math >}}$d${{< /math >}} layer:
- Parameters: {{< math >}}$2rd${{< /math >}} (two low-rank matrices)
- **Storage capacity**: {{< math >}}$64rd${{< /math >}} bits (FP32 representation)

**Example**: {{< math >}}$r = 8${{< /math >}}, {{< math >}}$d = 4096${{< /math >}} (typical for large LLMs):
- Parameters: {{< math >}}$2 \times 8 \times 4096 = 65{,}536${{< /math >}}
- Storage: {{< math >}}$64 \times 8 \times 4096 = 2{,}097{,}152${{< /math >}} bits {{< math >}}$= 256${{< /math >}} KB

**Capacity comparison**:

{{< math >}}
$$\frac{\text{LoRA storage capacity}}{\text{Information from RL}} = \frac{256 \text{ KB}}{2.5 \text{ KB}} \approx 100\times$$
{{< /math >}}

LoRA has **100 times** more capacity than the information we're trying to store! And even after 10,000 episodes ({{< math >}}$\sim 20${{< /math >}} KB of information), LoRA still has {{< math >}}$\sim 10\times${{< /math >}} headroom.

**Why this comparison is meaningful**: Sure, storage bits and information bits are different concepts—a 1GB hard drive doesn't mean you've learned 1GB of information. But the argument still holds: the **change** in the LoRA parameters from their initial state must *encode* the policy-relevant information learned through RL. If policy gradient provides only {{< math >}}$\sim 2N${{< /math >}} bits of information to guide this change, and LoRA provides a much larger parameter space ({{< math >}}$64rd${{< /math >}} bits) to store it, then the learning signal is the bottleneck, not the adapter's capacity.

{{% callout note %}}
**Conclusion**: LoRA provides orders of magnitude more representational capacity than policy gradient delivers information. This explains why such low-rank adapters suffice—the learning signal is the bottleneck, not the adapter capacity.
{{% /callout %}}

**Why full fine-tuning is wasteful**: And if you thought the LoRA numbers were striking, look at full fine-tuning. A 7B parameter model has storage capacity {{< math >}}$7\text{B} \times 32 = 224${{< /math >}} GB (in FP32):

{{< math >}}
$$\frac{7\text{B} \times 32 \text{ bits}}{2000 \text{ bits}} \approx 100{,}000{,}000\times \text{ excess capacity}$$
{{< /math >}}

That's **100 million times** more capacity than policy gradient can fill in a typical training run. No wonder people were overfitting!

---

### 4.3 Sample Complexity: An Information-Theoretic Perspective

**Information-theoretic lower bound**:

To reduce uncertainty about {{< math >}}$\pi^*${{< /math >}} from prior entropy {{< math >}}$H(\pi^*)${{< /math >}} to posterior entropy {{< math >}}$\epsilon${{< /math >}}:

{{< math >}}
$$I_{\text{required}} = H(\pi^*) - \epsilon$$
{{< /math >}}

**Translating to sample complexity**: If each episode provides {{< math >}}$I_{\text{per-episode}}${{< /math >}} bits:

{{< math >}}
$$N \geq \frac{I_{\text{required}}}{I_{\text{per-episode}}}$$
{{< /math >}}

**For different algorithms**:
- **Policy gradient**: {{< math >}}$I_{\text{per-episode}} = 1\text{-}4${{< /math >}} bits (with typical reward discretization) → {{< math >}}$N_{\text{PG}} = \Omega(H(\pi^*)/c)${{< /math >}} where {{< math >}}$c \in [1,4]${{< /math >}}
- **Actor-critic**: {{< math >}}$I_{\text{per-episode}} = O(T)${{< /math >}} → {{< math >}}$N_{\text{AC}} = \Omega(H(\pi^*)/T)${{< /math >}}

**Important caveats**:
1. Assumes information accumulates **additively** (ignores redundancy between episodes)
2. Real sample complexity depends on **optimization dynamics** (gradient descent may need more samples than information theory suggests)
3. Assumes **perfect utilization** of information (in practice, some information is "wasted")
4. Actor-critic requires **well-trained critic** to achieve {{< math >}}$O(T)${{< /math >}} bandwidth

**Example (illustrative)**: If {{< math >}}$H(\pi^*) \approx 10{,}000${{< /math >}} bits and {{< math >}}$T = 1000${{< /math >}} tokens:
- Policy gradient: {{< math >}}$\gtrsim 2{,}500${{< /math >}} to {{< math >}}$10{,}000${{< /math >}} episodes
- Actor-critic: {{< math >}}$\gtrsim 10${{< /math >}} to {{< math >}}$1{,}000${{< /math >}} episodes (depending on critic quality)
- **Gap**: {{< math >}}$10\times${{< /math >}} to {{< math >}}$1000\times${{< /math >}} improvement possible

These are **order-of-magnitude estimates** that explain qualitative differences observed empirically, not rigorous sample complexity theorems.

---

### Implications for LLM-RL

So what should we actually do with these insights? Let me highlight what I think are the most important practical implications.

**1. Dense vs. Sparse Rewards**:
- Sparse (outcome-based): Reward only at sequence end → {{< math >}}$O(1)${{< /math >}} bits/episode
- Dense (token-level): Reward at each token → {{< math >}}$O(T)${{< /math >}} bits/episode
- **Current limitation**: Most LLM-RL uses sparse rewards (RLHF with binary preferences)
- **Research opportunity**: Develop methods for meaningful token-level reward signals

**2. The Value-Based RL Gap**:

Here's what keeps me up at night: our analysis shows that actor-critic methods should theoretically achieve {{< math >}}$\sim T\times${{< /math >}} better sample efficiency than policy gradient. In traditional RL domains, PPO (actor-critic) requires {{< math >}}$\sim 100\times${{< /math >}} fewer samples than REINFORCE. We have existence proofs that this works!

**And yet, for LLMs:**
- **Current state**: No established training recipes for value function learning at scale
- **Key challenges**:
  - Value function training is unstable for large language models
  - Critic networks require careful initialization and hyperparameter tuning
  - Bootstrap error accumulation in long sequences ({{< math >}}$T \sim 1000${{< /math >}} tokens)
  - Computational overhead of maintaining and updating critics

**Why this matters from an information-theoretic perspective**:
- Current RLHF with policy gradient is limited to {{< math >}}$O(1)${{< /math >}} bits/episode
- Successfully training critics could unlock {{< math >}}$O(T) = O(1000)${{< /math >}} bits/episode
- This represents a {{< math >}}$1000\times${{< /math >}} potential improvement in information bandwidth

**3. Research Directions: Value-Based RL for LLMs**:

From our information-theoretic analysis, developing effective value-based methods for LLMs is the highest-leverage research direction:

a) **Stable Critic Training**:
   - **Challenge**: Value function learning at LLM scale
   - **Information perspective**: Each successful critic update should provide {{< math >}}$O(1)${{< /math >}} bits about {{< math >}}$\xi${{< /math >}} per token
   - **Research questions**: 
     - What architectures enable stable value learning?
     - How to handle long-horizon credit assignment ({{< math >}}$T \sim 1000${{< /math >}})?
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

Choose rank {{< math >}}$r${{< /math >}} such that: {{< math >}}$64rd \geq 2N${{< /math >}}

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
- LoRA capacity: {{< math >}}$64rd${{< /math >}} bits ({{< math >}}$\sim 100{,}000\times${{< /math >}} more than needed)
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

Our analysis reveals a massive opportunity: successfully training value functions for LLMs could improve sample efficiency by {{< math >}}$\sim 1000\times${{< /math >}} (from {{< math >}}$O(1)${{< /math >}} to {{< math >}}$O(T)${{< /math >}} bits/episode where {{< math >}}$T \sim 1000${{< /math >}}).

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

## Further Reading

- Russo & Van Roy (2014), "Learning to Optimize via Information-Directed Sampling"
- Ouyang et al. (2022), "Training language models to follow instructions with human feedback" (InstructGPT)
- Hu et al. (2021), "LoRA: Low-Rank Adaptation of Large Language Models"
- Cover & Thomas (2006), "Elements of Information Theory"

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