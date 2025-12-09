---
title: 'Language as a Universal Interface for Reinforcement Learning Agents'
subtitle: 'A rigorous mathematical framework for understanding agent-environment interaction and the fundamental challenges in building autonomous language agents'
summary: 'This post establishes a formal mathematical framework for language agents, deriving fundamental challenges from first principles and providing concrete design guidelines with real-world examples from SWE-Bench.'
authors:
  - admin
tags:
  - Language Agents
  - Reinforcement Learning
  - Agent Architecture
  - Software Engineering
categories:
  - Research
  - Theory
  - Engineering
date: '2025-11-07T00:00:00Z'
lastmod: '2025-11-07T00:00:00Z'
featured: true
draft: false
math: true
toc: true

# Featured image
image:
  caption: 'Language Agent Architecture'
  focal_point: ''
  preview_only: false

# Projects (optional)
projects: []
---

## Citation

```bibtex
@article{li2025languageagent,
  title   = {Language as a Universal Interface for Reinforcement Learning Agents},
  author  = {Li, Yingru},
  journal = {Richard Li's Blog},
  year    = {2025},
  url     = {https://richardli.xyz/post/language-rl-agent/}
}
```

---

## Introduction

**Why do some agents succeed while others fail?** While frameworks like ReAct (Yao et al., 2023) show that interleaving reasoning and acting works, they don't explain *why* it works or when it fails. We lack a principled understanding of what makes language agents fundamentally different from traditional RL agents.

This post establishes a mathematical framework revealing that **language serves as a universal interface** for RL agents, providing fundamental capabilities absent in traditional RL:

1. **Active vocabulary management** ({{< math >}}$\mathcal{V}${{< /math >}}): Agents design their own "language" for expressing actions, not constrained to fixed action spaces
2. **Active context management** ({{< math >}}$f_{\text{agent}}${{< /math >}}): Agents control what information to retain through learned state compression (memory management)
3. **Two-layer decision structure**: Macro policy {{< math >}}$\pi(A_k | X_k)${{< /math >}} implemented through micro token generation {{< math >}}$p_{\theta}(\mathbf{v}_k | X_k)${{< /math >}}, enabling Chain-of-Thought reasoning

These aren't implementation details—they're the fundamental architectural differences that determine agent capability.

**What we'll cover:**
- Why state compression is a physical necessity, not a design choice (§1.2)
- The two-layer decision structure that connects thought generation to action execution (§1.4)
- Trajectory probability modeling for multi-turn agent interactions (§1.5)
- Interface design separating environment dynamics from agent evaluation (§1.6)
- How SWE-Bench agents map to this formal framework (§1.7)

**The payoff:** A principled understanding of when agents will succeed, when they'll fail, and exactly which design choices matter.

---

## Formal Modeling

### 1.1 Agent-Environment Interface: The Building Blocks

Every agent-environment interaction reduces to a sequence of events. The key insight: there are two fundamentally different views of history.

#### Core Definitions

**Internal Thought** ({{< math >}}$th_k${{< /math >}}): At turn {{< math >}}$k${{< /math >}}, the agent generates free-form reasoning, planning, or intermediate text based on its internal state. This is the agent's private, internal information—the first step in its decision-making process.

**External Action** ({{< math >}}$A_k${{< /math >}}): The structured instruction extracted from {{< math >}}$th_k${{< /math >}} through a deterministic Parser function that will affect the external world:
{{< math >}}$$A_k = \text{Parser}(th_k)$${{< /math >}}

**External Observation** ({{< math >}}$O_k${{< /math >}}): After action {{< math >}}$A_{k-1}${{< /math >}} acts on the environment, the information returned to the agent.

**Vocabulary** ({{< math >}}$\mathcal{V}${{< /math >}}): The set of all tokens available to the language model for generating thoughts. Unlike traditional RL where action spaces are fixed, language agents can actively design and extend their vocabulary.

#### Two Types of History

**1. External History** ({{< math >}}$H_k^{\text{ext}}${{< /math >}}):

The objectively occurring, externally observable event sequence—the "ground truth" of world evolution:

{{< math >}}$$H_k^{\text{ext}} = (A_0, O_1, A_1, O_2, \ldots, A_{k-1}, O_k)$${{< /math >}}

**2. Agent-Centric History** ({{< math >}}$H_k^{\text{agent}}${{< /math >}}):

The complete information accessible to the agent during decision-making, including its internal thought process:

{{< math >}}$$H_k^{\text{agent}} = (\text{system\_prompt}, O_0, th_0, A_0, O_1, th_1, A_1, O_2, \ldots, th_{k-1}, A_{k-1}, O_k)$${{< /math >}}

This history is the complete information foundation for the agent's learning and construction of its internal mental model.

#### Environment

**Environment** ({{< math >}}$\mathcal{E}${{< /math >}}): The external environment's behavior is characterized by a probabilistic transition function {{< math >}}$\rho${{< /math >}}, which gives the probability of the next observation based on external history and the agent's action:

{{< math >}}$$\rho(O_{k+1} | H_k^{\text{ext}}, A_k)$${{< /math >}}

In our framework, {{< math >}}$\rho${{< /math >}} is externally given and typically unknown—a "black box."

#### Key Design Choices: Why Language Changes Everything

**Here's what traditional RL agents cannot do:** In classic RL, the action space {{< math >}}$\mathcal{A}${{< /math >}} is fixed. In Atari, you have {up, down, left, right, fire}. In chess, you have legal moves. The agent optimizes policy {{< math >}}$\pi(a|s)${{< /math >}} over this frozen set.

**Language agents break this constraint.** They have two degrees of freedom unavailable to traditional RL:

**1. Active Vocabulary Management ({{< math >}}$\mathcal{V}${{< /math >}})**

Unlike fixed action spaces in traditional RL, language agents can **actively design and manage their vocabulary**—the set of tokens they use to express thoughts and actions. This includes:

- **Domain-specific tokens**: Extending the vocabulary with task-relevant tokens (e.g., function names, domain concepts)
- **Structured output vocabularies**: Designing token sets that naturally express structured actions (e.g., JSON, XML tags)
- **Hierarchical vocabularies**: Multi-level token sets enabling both high-level planning and low-level execution

The vocabulary {{< math >}}$\mathcal{V}${{< /math >}} directly determines what can be expressed in {{< math >}}$th_k${{< /math >}}, which through the Parser determines the effective action space {{< math >}}$\mathcal{A}${{< /math >}}. This is a **learnable design choice**, not a fixed constraint.

**2. Active Context Management (via agent state)**

Language agents must **actively manage what information to retain** in their internal state at each turn. As we'll see in §1.2, agents compress the growing history {{< math >}}$H_k^{\text{agent}}${{< /math >}} into a finite state representation. This state management function is not just passive compression—it's an **active policy** for context management:

- **What to remember**: Selecting which past observations, thoughts, and actions to retain
- **What to forget**: Discarding irrelevant information to stay within context limits
- **How to compress**: Choosing representations (verbatim, summarized, structured)
- **When to retrieve**: Deciding when to access external memory vs. internal context

The design of this state management function is as important as the action policy itself—poor context management creates an information bottleneck that no amount of model capacity can overcome.

**Why this matters**: Both vocabulary and context are expressed in natural language—the same medium humans use. This makes language agents uniquely:
- **Interpretable**: You can read what the agent thinks and why
- **Debuggable**: You can pinpoint where reasoning fails
- **Jointly optimizable**: Vocabulary and context can be improved together

This is why language is a universal interface: it unifies expressiveness (vocabulary) and memory (context) in one coherent framework.

---

### 1.2 Why State Compression is Inevitable

#### The Dilemma

**In theory**, an agent could make perfect decisions using complete history: {{< math >}}$\pi(A_k | H_k^{\text{agent}})${{< /math >}}, where {{< math >}}$H_k^{\text{agent}}${{< /math >}} contains every thought, action, and observation from turn 0 to {{< math >}}$k${{< /math >}}.

**In practice**, this is impossible. As {{< math >}}$k${{< /math >}} grows, {{< math >}}$H_k^{\text{agent}}${{< /math >}} grows without bound. This isn't merely about computational cost (though Transformer's {{< math >}}$O(|H_k^{\text{agent}}|^2)${{< /math >}} scaling hurts). It's about **computability**: no finite device can process infinite input.

**The implication is stark**: State compression is not a design choice. It's a physical necessity.

#### Inevitable Conclusion: Agent State ({{< math >}}$X_k${{< /math >}})

To make decisions possible, the agent **must** compress the infinitely growing agent-centric history {{< math >}}$H_k^{\text{agent}}${{< /math >}} into a fixed-size internal representation. We call this representation the **agent state** {{< math >}}$X_k${{< /math >}}:

{{< math >}}$$X_k \approx \text{compress}(H_k^{\text{agent}})$${{< /math >}}

This state {{< math >}}$X_k${{< /math >}} is the agent's "mental model" or "working memory" of the world that it relies on for decision-making. From this point, the agent's policy is based on this computable state:

{{< math >}}$$\pi(A_k | X_k)$${{< /math >}}

#### The Critical Function: State Update ({{< math >}}$f_{\text{agent}}${{< /math >}})

State must evolve as new information arrives:

{{< math >}}$$X_{k+1} = f_{\text{agent}}(X_k, th_k, A_k, O_{k+1})$${{< /math >}}

**This function is the agent's memory policy**—deciding what to remember and what to forget. Its design is as crucial as the action policy {{< math >}}$\pi${{< /math >}} itself. In fact, optimizing {{< math >}}$f_{\text{agent}}${{< /math >}} is a **meta-learning problem**: learning a compression policy that preserves task-relevant information to maximize the primary policy's expected return.

**Why it matters**: Compression is lossy. {{< math >}}$X_k${{< /math >}}'s quality sets the performance ceiling. Even the world's most powerful LLM cannot compensate for bad memory management.

#### Practical Paradigms for {{< math >}}$f_{\text{agent}}${{< /math >}}

**Sliding Window**: The simplest approach—{{< math >}}$X_k${{< /math >}} only contains the most recent {{< math >}}$N${{< /math >}} turns of {{< math >}}$(th, A, O)${{< /math >}} tuples.

**Language Model-based Summarization**: Use language model calls to periodically "compress" old {{< math >}}$X_k${{< /math >}} and new {{< math >}}$(th_k, A_k, O_{k+1})${{< /math >}}.

**Structured Memory**: Extract information from {{< math >}}$H_k^{\text{agent}}${{< /math >}} and store it in an external vector database or knowledge graph. Here, {{< math >}}$X_k${{< /math >}} is a complex object containing dialogue summaries, entity lists, etc., and {{< math >}}$f_{\text{agent}}${{< /math >}} defines how to read and write this structured memory.

**Learnable Memory Modules**: Advanced approaches using neural architectures that jointly optimize memory selection with the policy itself.

---

### 1.3 Optimization Objective and Reward Formation

The agent's behavior is not random—it's driven by a clear objective: maximizing long-term cumulative reward.

#### Ultimate Goal: Maximize Return

The agent's ultimate goal is to maximize a long-term value called **return** ({{< math >}}$G_k${{< /math >}}), which is the cumulative sum of all future rewards considering time discount factor {{< math >}}$\gamma \in [0, 1]${{< /math >}}:

{{< math >}}$$G_k = \sum_{t=0}^{\infty} \gamma^t R_{k+t+1} = R_{k+1} + \gamma R_{k+2} + \gamma^2 R_{k+3} + \ldots$${{< /math >}}

The optimal policy {{< math >}}$\pi^*${{< /math >}} aims to maximize the expected value of this return:

{{< math >}}$$\pi^* = \arg\max_{\pi} \mathbb{E}[G_k | \pi]$${{< /math >}}

#### Reward Formulation

In any practically operational system, reward {{< math >}}$R_{k+1}${{< /math >}} calculation must rely on information the agent can access. We define reward formation as a reward function {{< math >}}$r${{< /math >}}, with the most general form:

{{< math >}}$$R_{k+1} = r(X_k, th_k, A_k, O_{k+1})$${{< /math >}}

**This definition is crucial** because it reveals the dual core role of state {{< math >}}$X_k${{< /math >}} and thought {{< math >}}$th_k${{< /math >}}: they are not only decision inputs but also evaluation (reward function {{< math >}}$r${{< /math >}}) inputs. A low-quality state {{< math >}}$X_k${{< /math >}} (poor compression of {{< math >}}$H_k^{\text{agent}}${{< /math >}}) means the reward {{< math >}}$R_{k+1} = r(X_k, th_k, A_k, O_{k+1})${{< /math >}} is computed from **partially observed information**. This creates a **non-Markovian reward signal**—the observed reward becomes a biased estimate of the true reward {{< math >}}$r(H_k^{\text{agent}}, th_k, A_k, O_{k+1})${{< /math >}} that would be computed from complete history. This is analogous to the classic POMDP problem, but applied to the reward function itself: poor state compression degrades not just the policy, but the learning signal that guides it.

---

### 1.4 The Two-Layer Decision Structure: Thought Before Action

**The key difference**: Traditional RL agents directly output actions. Language agents first generate thoughts (natural language reasoning), then parse actions from those thoughts.

This creates a two-layer structure with profound implications.

#### Layer 1: Macro Task Layer ({{< math >}}$M_{\text{turn}}${{< /math >}})

This layer completely inherits from the general framework—it's the level where the agent conducts meaningful interaction with the environment. Its state is macro state {{< math >}}$X_k${{< /math >}}, action is macro action {{< math >}}$A_k${{< /math >}}. Its ultimate goal is learning an optimal macro policy {{< math >}}$\pi^*(A_k | X_k)${{< /math >}} to maximize long-term return {{< math >}}$G_k${{< /math >}}.

#### Layer 2: Micro Generation Layer ({{< math >}}$M_{\text{micro}}${{< /math >}})

This layer's core function is to **implement** the macro policy {{< math >}}$\pi${{< /math >}}. It describes how the agent's "thought" is generated token by token by the LLM.

**Basic Units**: We must distinguish two concepts:

- **Token sequence** ({{< math >}}$\mathbf{v}_k${{< /math >}}): The fundamental data structure directly output by LLM policy {{< math >}}$p_{\theta}${{< /math >}}, a sequence composed of tokens: {{< math >}}$\mathbf{v}_k = (v_{k,1}, v_{k,2}, \ldots, v_{k,T_k})${{< /math >}}

- **Thought string** ({{< math >}}$th_k${{< /math >}}): The human-readable text string converted from token sequence {{< math >}}$\mathbf{v}_k${{< /math >}} through decode function: {{< math >}}$th_k = \text{Decode}(\mathbf{v}_k)${{< /math >}}

**Note**: When referring to {{< math >}}$th_k${{< /math >}}, without special indication, it can refer to either the thought string or the {{< math >}}$\mathbf{v}_k${{< /math >}} generated by the LLM at that time. The Decode function is typically deterministic (one token sequence maps to one string), but the inverse mapping (text to tokens) can be many-to-one due to different tokenization schemes.

**Generation Process**: This process is controlled by LLM parameters {{< math >}}$\theta${{< /math >}}, defining the probability of generating a specific token sequence {{< math >}}$\mathbf{v}_k${{< /math >}} given state {{< math >}}$X_k${{< /math >}}. For autoregressive models:

{{< math >}}$$p_{\theta}(\mathbf{v}_k | X_k) = \prod_{t=1}^{T_k} p_{\theta}(v_{k,t} | X_k, v_{k,1:t-1})$${{< /math >}}

where {{< math >}}$v_{k,1:t-1} = (v_{k,1}, \ldots, v_{k,t-1})${{< /math >}} denotes the token history up to position {{< math >}}$t-1${{< /math >}} in turn {{< math >}}$k${{< /math >}}.

#### Connecting Macro and Micro

The macro policy connects to micro generation through the following core equation. This equation is built on token sequence probabilities, precisely handling the characteristic that "one action can be implemented by multiple thoughts (multiple token sequences)":

{{< math >}}$$\pi(A_k | X_k) \equiv \sum_{\mathbf{v} \in \mathcal{V}^*} \mathbf{1} [ \text{Parser}(\text{Decode}(\mathbf{v})) = A_k ] \cdot p_{\theta}(\mathbf{v} | X_k)$${{< /math >}}

where {{< math >}}$\mathcal{V}^*${{< /math >}} represents all possible token sequence sets. This formula shows that a macro action's probability is the sum of probabilities of all "token sequences that can be decoded and parsed into that action."

**Core Learning Task**: The essence of agent training is using experience data obtained from environment interaction (i.e., sequences containing {{< math >}}$(X_k, th_k(\mathbf{v}_k), A_k, O_{k+1}, R_{k+1})${{< /math >}} information) to adjust micro generation layer parameters {{< math >}}$\theta${{< /math >}}, thereby optimizing macro layer policy {{< math >}}$\pi${{< /math >}}, ultimately achieving the goal of maximizing long-term return. The challenge: gradients must flow through the non-differentiable {{< math >}}$\text{Parser}${{< /math >}} function, requiring sampling-based RL methods like REINFORCE.

**Generality**: This two-layer definition has universality and can cover multiple generation model architectures:

- **Autoregressive models**: As defined above, generating {{< math >}}$\mathbf{v}_k${{< /math >}} by predicting tokens one by one
- **Diffusion models**: Generating entire token sequence {{< math >}}$\mathbf{v}_k${{< /math >}}'s representation through iterative denoising from noise
- **Other models**: Such as Tokenizer-free models, etc., where the core idea applies equally

#### Why This Structure Matters

**The opportunity**: Separating thought ({{< math >}}$th_k${{< /math >}}) from execution ({{< math >}}$A_k${{< /math >}}) unlocks Chain-of-Thought reasoning—complex planning without hardcoding logic into action space {{< math >}}$\mathcal{A}${{< /math >}}.

**The bottleneck**: The Parser is both bridge and weakness. Poor parsing wastes perfect reasoning. Robust Parser design is critical.

**The credit assignment nightmare**: When {{< math >}}$A_k${{< /math >}} fails, where's the blame?
- Was {{< math >}}$th_k${{< /math >}} wrong conceptually?
- Was {{< math >}}$th_k${{< /math >}} right but Parser-incompatible?
- Was {{< math >}}$A_k${{< /math >}} actually fine but environment-inappropriate?

This three-way ambiguity makes learning harder than single-layer RL. Technically, this manifests as a **high-variance gradient problem**: a single action {{< math >}}$A_k${{< /math >}} corresponds to many valid token sequences, but policy gradient methods only sample one, leading to high variance in gradient estimates.

**The silver lining**: Reverse parsing enables data augmentation. Given good action {{< math >}}$A_k${{< /math >}}, generate multiple thought chains {{< math >}}$th_k${{< /math >}} that lead to it. This creates rich {{< math >}}$(X_k, th_k, A_k)${{< /math >}} training data, teaching the model "how to think to act correctly."

---

### 1.5 Multi-turn Language Agent Trajectory Probability Modeling

A trajectory, typically denoted {{< math >}}$\tau${{< /math >}}, is a complete sequence of events produced by agent-environment interaction. In a complete trajectory, there are two core sources of randomness:

1. **Agent's decisions**: Under given state {{< math >}}$X_k${{< /math >}}, which "thought" {{< math >}}$\mathbf{v}_k${{< /math >}} the agent generates is a probabilistic event determined by its internal LLM policy {{< math >}}$p_{\theta}${{< /math >}}

2. **Environment's responses**: After the agent executes action {{< math >}}$A_k${{< /math >}}, which "observation" {{< math >}}$O_{k+1}${{< /math >}} the environment produces is a probabilistic event determined by the environment's transition function {{< math >}}$\rho${{< /math >}}

Other steps, such as parsing action {{< math >}}$A_k${{< /math >}} from thought {{< math >}}$\mathbf{v}_k${{< /math >}} (through `Parser` function), calculating reward {{< math >}}$R_{k+1}${{< /math >}} (through `r` function), and updating agent state {{< math >}}$X_{k+1}${{< /math >}} (through {{< math >}}$f_{\text{agent}}${{< /math >}} function, assumed deterministic) are deterministic.

#### Trajectory Definition (Agent-Centric History)

First, we define a trajectory {{< math >}}$\tau${{< /math >}} of length {{< math >}}$T${{< /math >}} turns as a sequence of core events starting from the initial state:

{{< math >}}$$\tau = (X_0, \mathbf{v}_0, A_0, O_1, X_1, \mathbf{v}_1, A_1, O_2, \ldots, X_T, \mathbf{v}_T, A_T, O_{T+1})$${{< /math >}}

where:
- {{< math >}}$X_k${{< /math >}} is the agent's state at turn {{< math >}}$k${{< /math >}} (compressed representation of history)
- {{< math >}}$\mathbf{v}_k${{< /math >}} is the token sequence generated by the LLM at turn {{< math >}}$k${{< /math >}}: {{< math >}}$\mathbf{v}_k = (v_{k,1}, \ldots, v_{k, |\mathbf{v}_k|})${{< /math >}}
- {{< math >}}$A_k${{< /math >}} is the action parsed from {{< math >}}$\mathbf{v}_k${{< /math >}}: {{< math >}}$A_k = \text{Parser}(\text{Decode}(\mathbf{v}_k))${{< /math >}}
- {{< math >}}$O_{k+1}${{< /math >}} is the environment's observation response to action {{< math >}}$A_k${{< /math >}}

#### Trajectory Probability

{{< math >}}$$P(\tau | \theta, \rho) = p(X_0) \prod_{k=0}^{T} \left[ \underbrace{p_{\theta}(\mathbf{v}_k | X_k)}_{\text{Agent's Policy}} \cdot \underbrace{\rho(O_{k+1} | H_k^{\text{ext}}, A_k)}_{\text{Environment's Dynamics}} \right]$${{< /math >}}

where:

{{< math >}}$$p_{\theta}(\mathbf{v}_k | X_k) = \prod_{t=1}^{|\mathbf{v}_k|}p_{\theta}(v_{k,t} | X_k, v_{k,1:t-1})$${{< /math >}}

#### Trajectory Probability Ratio

The trajectory probability ratio of two models {{< math >}}$\theta, \theta'${{< /math >}} is:

{{< math >}}$$\frac{P(\tau | \theta, \rho)}{P(\tau| \theta', \rho)} = \frac{p(X_0) \prod_{k=0}^{T} \left[ {p_{\theta}(\mathbf{v}_k | X_k)} \cdot {\rho(O_{k+1} | H_k^{\text{ext}}, A_k)} \right]} {p(X_0) \prod_{k=0}^{T} \left[ {p_{\theta'}(\mathbf{v}_k | X_k)} \cdot {\rho(O_{k+1} | H_k^{\text{ext}}, A_k)} \right]}$${{< /math >}}

After canceling common terms:

{{< math >}}$$\frac{P(\tau | \theta, \rho)}{P(\tau| \theta', \rho)} = \prod_{k = 0}^T \frac{p_{\theta}(\mathbf{v}_k | X_k)}{p_{\theta'}(\mathbf{v}_k | X_k)} = \prod_{k=0}^T \prod_{t = 1}^{|\mathbf{v}_k|} \frac{p_{\theta}(v_{k,t} | X_k, v_{k,1:t-1})}{p_{\theta'}(v_{k,t} | X_k, v_{k,1:t-1})}$${{< /math >}}

This factorization enables RL algorithms (PPO, GRPO, etc.) to optimize {{< math >}}$\theta${{< /math >}} by computing importance weights at either the turn level or token level.

---

### 1.6 Interface Abstraction: Connecting Theory and Code Implementation

To ground the above formalization theory into an extensible, trainable software system, our designed interface must reflect language agents' core characteristics. We adopt an "agent-driven evaluation" paradigm: the environment (Env) only simulates the world's "physical laws," while the agent (Agent) not only makes decisions but also actively evaluates the consequences of its decisions and generates reward signals for itself.

This paradigm cleanly separates two concepts:

- **World state transition**: Handled by the environment
- **Agent state transition and value judgment**: Handled by the agent

---

#### Environment Interface

The environment interface strictly follows its physical role: an interactive world simulator containing no subjective value judgments.

**step(action: A_k) → Tuple[O_{k+1}, bool, bool, Dict]**

- **Implements**: Encapsulates world laws {{< math >}}$\rho(O_{k+1} | H_k^{\text{ext}}, A_k)${{< /math >}}
- **Behavior**: Receives a macro action {{< math >}}$A_k${{< /math >}}, executes world state transition
- **Returns**:
  - `O_{k+1}` (Observation): Environment's next observation
  - `bool` (Terminated): Whether episode terminates due to task success/failure
  - `bool` (Truncated): Whether episode is cut short due to external limits (e.g., timeout)
  - `Dict` (Info): Additional information for debugging
- **Key**: This method does **not** return reward

**reset() → Tuple[O_0, Dict]**

- **Behavior**: Reset environment, start a new interaction episode. Returns initial observation

**action_space / observation_space**

- **Behavior**: Define legal action {{< math >}}$A_k${{< /math >}} and observation {{< math >}}$O_k${{< /math >}} structure and types, following gymnasium.Space specification

---

#### Agent Interface

The agent is the system's core, integrating five major functions: **perception, thinking, action, evaluation, and learning**.

**generate_thought_and_action(state: X_k) → Tuple[th_k, A_k]**

- **Implements**: Encapsulates the complete decision chain from state to action
- **Internal flow**:
  1. **Thought Generation**: Sample token sequence {{< math >}}$\mathbf{v}_k${{< /math >}} from micro generation layer {{< math >}}$p_{\theta}(\mathbf{v}_k | X_k)${{< /math >}}, then decode to thought text {{< math >}}$th_k = \text{Decode}(\mathbf{v}_k)${{< /math >}}
  2. **Action Parsing**: Call deterministic {{< math >}}$\text{Parser}(th_k)${{< /math >}} function to extract structured macro action {{< math >}}$A_k${{< /math >}} from thought
- **Returns**: {{< math >}}$(th_k, A_k)${{< /math >}} tuple—complete thought chain and final action for this decision

**evaluate_step(X_k, th_k, A_k, O_{k+1}) → R_{k+1}**

- **Implements**: Encapsulates reward function {{< math >}}$r(X_k, th_k, A_k, O_{k+1})${{< /math >}}. This is the core method actively called by the agent
- **Behavior**: The agent conducts **self-evaluation** based on its pre-decision state {{< math >}}$X_k${{< /math >}}, complete thought process {{< math >}}$th_k${{< /math >}}, executed action {{< math >}}$A_k${{< /math >}}, and environment-given consequence {{< math >}}$O_{k+1}${{< /math >}}, calculating reward value {{< math >}}$R_{k+1}${{< /math >}} for this step
- **Examples**:
  - A software engineering agent's `evaluate_step` might execute unit tests and calculate reward based on test pass rate
  - A dialogue agent's `evaluate_step` might call a sentiment analysis model to judge user satisfaction as reward

**learn(trajectory_batch: List[Tuple])**

- **Implements**: Connects to a specific reinforcement learning backend (e.g., [VeRL](https://github.com/volcengine/verl) - Volcano Engine Reinforcement Learning for LLMs)
- **Behavior**: Receives a batch of complete experience trajectories, where each trajectory point contains {{< math >}}$(X_k, th_k, A_k, R_{k+1}, X_{k+1}, ...)${{< /math >}}. It formats this data and passes it to the RL backend's optimizer (e.g., PPO, GRPO, ReMax, etc.) to perform updates to model parameters {{< math >}}$\theta${{< /math >}}

---

#### Main Interaction Loop Pseudocode

This interface design's main interaction loop clearly demonstrates how components collaborate:

```python
# Initialization
agent = Agent()
env = Env()
observation, info = env.reset()

# Build initial state from initial observation
agent.build_initial_state(observation)

for turn in range(MAX_TURNS):
    # 1. Agent gets current state
    current_state = agent.get_current_state()  # Get X_k

    # 2. Agent thinks and decides action
    thought, action = agent.generate_thought_and_action(current_state)  # Generate th_k, A_k

    # 3. Action acts on environment
    next_observation, terminated, truncated, info = env.step(action)  # Get O_{k+1}

    # 4. Agent actively evaluates previous step's result, generates reward
    reward = agent.evaluate_step(current_state, thought, action, next_observation)  # Calculate R_{k+1}

    # 5. Agent updates its internal state (memory)
    agent.update_state(thought, action, next_observation)  # f_agent -> X_{k+1}

    # 6. Store complete experience tuple in replay buffer
    agent.replay_buffer.add((current_state, thought, action, reward, agent.get_current_state()))

    # 7. (Optional) Perform one training learning iteration
    if len(agent.replay_buffer) > BATCH_SIZE:
        experience_batch = agent.replay_buffer.sample(BATCH_SIZE)
        agent.learn(experience_batch)  # Call VERL backend

    # Check if episode ends
    if terminated or truncated:
        break
```

---

### 1.7 Practical Analysis: SWE-Bench Agent Example

To connect the above abstract theoretical framework with real-world agent systems, we take an agent running on the Software Engineering Benchmark (SWE-Bench) as an example, analyzing in detail how its components correspond one-to-one with our formal definitions.

#### Environment ({{< math >}}$\mathcal{E}${{< /math >}})

In SWE-Benchmark settings, the environment {{< math >}}$\mathcal{E}${{< /math >}} is a highly isolated and standardized code repository.

- **Implementation**: Each task instance runs in a sandbox similar to bubble wrap, providing filesystem and network isolation. This ensures the agent's actions won't accidentally affect external systems and guarantees experiment reproducibility

- **World State Transition** ({{< math >}}$\rho(O_{k+1} | H_k^{\text{ext}}, A_k)${{< /math >}}): The environment's physical laws are defined by the underlying operating system (usually Linux) and pre-installed software (like git, python, pytest). When the agent executes an action {{< math >}}$A_k${{< /math >}} (e.g., a bash command), the environment undergoes state transition according to these laws (e.g., files are modified, processes are created) and captures stdout and stderr produced by that action as the next observation {{< math >}}$O_{k+1}${{< /math >}}

#### Agent-Environment Interface Correspondence

**Macro Action ({{< math >}}$A_k${{< /math >}}) and Action Space ({{< math >}}$\mathcal{A}${{< /math >}})**

The agent's macro actions {{< math >}}$A_k${{< /math >}} are a series of predefined, structured tool calls. Action space {{< math >}}$\mathcal{A}${{< /math >}} is the set of all these legal tool calls. Typical tools include:

- **bash**: Execute a shell command
  - Formal representation: {{< math >}}$A_k = (\text{tool: "bash"}, \{\text{command: string}\})${{< /math >}}
  - Example: {{< math >}}$A_k = (\text{bash}, \{\text{command: "ls -F /testbed"}\})${{< /math >}}

- **edit**: Modify files—itself is a composite tool
  - View file: {{< math >}}$A_k = (\text{edit}, \{\text{command: "view", path: string, view\_range: [int, int]}\})${{< /math >}}
  - String replacement: {{< math >}}$A_k = (\text{edit}, \{\text{command: "str\_replace", path: string, old\_str: string, new\_str: string}\})${{< /math >}}
  - Insert code: {{< math >}}$A_k = (\text{edit}, \{\text{command: "insert", path: string, new\_str: string, insert\_line: int}\})${{< /math >}}

- **submit**: Terminate task and submit final solution
  - Formal representation: {{< math >}}$A_k = (\text{submit}, \{\})${{< /math >}}

**Thought ({{< math >}}$th_k${{< /math >}}), Action Parsing (Parser), and Observation ({{< math >}}$O_k${{< /math >}})**

This is the core link connecting micro generation with macro interaction.

- **Internal Thought** ({{< math >}}$th_k${{< /math >}}): The complete text generated by the LLM given current state {{< math >}}$X_k${{< /math >}}. It usually contains reasoning process, analysis of current situation, and next step plan. Example:

```
Let's look at the utils module, which seems to handle parameter parsing:
<tool_call>
<function>edit</function>
<parameter name="command">view</parameter>
<parameter name="path">/testbed/spectree/utils.py</parameter>
</tool_call>
```

- **Action Parsing** ({{< math >}}$A_k = \text{Parser}(th_k)${{< /math >}}): Parser is a deterministic function responsible for extracting structured macro action {{< math >}}$A_k${{< /math >}} from free-form thought text {{< math >}}$th_k${{< /math >}}. In practice, this is typically implemented through regular expressions or XML/JSON parsing to identify and extract content in `<tool_call>` or similar tags. If parsing fails, it may produce a special "no-op" or "error" action

- **External Observation** ({{< math >}}$O_k${{< /math >}}): Information returned to the agent after the environment executes action {{< math >}}$A_{k-1}${{< /math >}}. In SWE-Benchmark, this is usually the combination of bash command stdout and stderr. To avoid information overload, returned observations are typically truncated or abbreviated. A well-designed tool should have clear success or failure return information so the agent can understand the consequences of its actions

#### State Construction and Update ({{< math >}}$X_k${{< /math >}} and {{< math >}}$f_{\text{agent}}${{< /math >}})

- **Agent-Centric History** ({{< math >}}$H_k^{\text{agent}}${{< /math >}}): This is the complete information source for agent decision-making. In SWE-Benchmark practice, it's usually organized as a dialogue-style record:

{{< math >}}$$H_k^{\text{agent}} = (\text{system\_prompt}, O_0, th_0, A_0, O_1, th_1, A_1, O_2, \ldots, th_{k-1}, A_{k-1}, O_k)$${{< /math >}}

where {{< math >}}$O_0${{< /math >}} contains the initial task description (Problem Statement) and environment introduction.

- **State Update** ({{< math >}}$X_{k+1} = f_{\text{agent}}(X_k, th_k, A_k, O_{k+1})${{< /math >}}): Due to the "history explosion" problem, complete {{< math >}}$H_k^{\text{agent}}${{< /math >}} cannot directly serve as LLM input. The agent's state {{< math >}}$X_k${{< /math >}} is actually the Prompt input to the LLM. {{< math >}}$f_{\text{agent}}${{< /math >}} is the strategy for constructing this Prompt from history—the memory and forgetting mechanism. Common implementations include:

  - **Sliding Window**: A simple strategy, e.g., SWE-agent-lm only retains the most recent 5 observations {{< math >}}$O_i${{< /math >}} and complete thought-action history {{< math >}}$(th, A)${{< /math >}} when constructing next state {{< math >}}$X_k${{< /math >}}

  - **Intelligent Compression**: More advanced methods, like Claude model—when history approaches context window limit, it retains about 30% of key historical steps and summarizes or compresses the remaining 70%

  - **Complete History**: In early SWE-agent or training stages, sometimes the entire {{< math >}}$H_k^{\text{agent}}${{< /math >}} is concatenated as {{< math >}}$X_k${{< /math >}}. While this is information-lossless, it's extremely costly and limited by model context length

#### Optimization Objective and Reward Formation ({{< math >}}$r${{< /math >}})

- **Reward Function** ({{< math >}}$R_{k+1} = r(X_k, th_k, A_k, O_{k+1})${{< /math >}}): In SWE-Benchmark, reward implementation is **sparse and delayed**.

  - For most intermediate steps (like bash, edit), reward {{< math >}}$R_{k+1}${{< /math >}} is constantly 0. The agent receives no explicit right/wrong signal during exploration
  - Only when the agent executes final action {{< math >}}$A_k = (\text{submit}, \{\})${{< /math >}} is a non-zero reward calculation triggered

- **evaluate_step Implementation**: This reward calculation process is implemented by the `agent.evaluate_step(...)` method in our interface abstraction. In SWE-Benchmark, this method executes an evaluation script (eval.sh) with the following specific flow:

  1. **Environment Reset**: Script first uses `git reset --hard` and `git clean -fd` to restore code repository to clean initial version
  2. **Apply Patch**: Apply all code modifications generated by the agent during interaction (in .patch file form) to the code repository
  3. **Run Tests**: Activate virtual environment, then use testing frameworks like pytest to run predefined test cases
  4. **Parse Results**: Script captures pytest output logs (eval log)
  5. **Calculate Reward**: By parsing logs, determine if tests passed. Final reward {{< math >}}$R_{\text{final}}${{< /math >}} is given based on this result, e.g.:
     - All tests pass: {{< math >}}$R_{\text{final}} = +1${{< /math >}}
     - Tests fail: {{< math >}}$R_{\text{final}} = -1${{< /math >}} (or other value less than 1)

This process perfectly interprets our definition: reward {{< math >}}$R_{k+1}${{< /math >}} is generated by the agent (its evaluation module) after action {{< math >}}$A_k${{< /math >}} acts on the environment and produces observation {{< math >}}$O_{k+1}${{< /math >}} (here referring to test results). This sparse reward characteristic also brings enormous credit assignment challenges to reinforcement learning algorithms (like PPO, ReMax).

**A prescriptive insight from our framework:** This sparse reward design is suboptimal. Our framework suggests a more effective agent could leverage its `evaluate_step` capability to **generate denser, intermediate rewards**. For example, after an `edit` action, the agent could self-evaluate by running a linter, static type-checker, or unit tests on modified functions—generating internal reward signals {{< math >}}$R_{k+1} > 0${{< /math >}} for syntactically correct code or passing local tests, even before the final submission. This demonstrates how our framework provides not just a descriptive model, but a blueprint for designing more sample-efficient agents.

---

## Notation Summary

The table below summarizes the key notation introduced throughout this post:

| Symbol | Type | Definition | First Use |
|--------|------|------------|-----------|
| {{< math >}}$k${{< /math >}} | Index | Turn/step index in the interaction | §1.1 |
| {{< math >}}$th_k${{< /math >}} | Text | Internal thought at turn {{< math >}}$k${{< /math >}} (free-form reasoning) | §1.1 |
| {{< math >}}$A_k${{< /math >}} | Action | External action at turn {{< math >}}$k${{< /math >}} (structured command) | §1.1 |
| {{< math >}}$O_k${{< /math >}} | Observation | External observation at turn {{< math >}}$k${{< /math >}} (environment feedback) | §1.1 |
| {{< math >}}$H_k^{\text{ext}}${{< /math >}} | Sequence | External history: {{< math >}}$(A_0, O_1, \ldots, A_{k-1}, O_k)${{< /math >}} | §1.1 |
| {{< math >}}$H_k^{\text{agent}}${{< /math >}} | Sequence | Agent-centric history: includes thoughts {{< math >}}$th_i${{< /math >}} | §1.1 |
| {{< math >}}$\mathcal{E}${{< /math >}} | Environment | The external world/task environment | §1.1 |
| {{< math >}}$\rho${{< /math >}} | Function | Environment transition: {{< math >}}$\rho(O_{k+1} \| H_k^{\text{ext}}, A_k)${{< /math >}} | §1.1 |
| {{< math >}}$\text{Parser}${{< /math >}} | Function | Deterministic mapping: {{< math >}}$th_k \to A_k${{< /math >}} | §1.1 |
| {{< math >}}$X_k${{< /math >}} | State | Agent state at turn {{< math >}}$k${{< /math >}} (compressed history) | §1.2 |
| {{< math >}}$f_{\text{agent}}${{< /math >}} | Function | State update: {{< math >}}$X_{k+1} = f_{\text{agent}}(X_k, th_k, A_k, O_{k+1})${{< /math >}} | §1.2 |
| {{< math >}}$\pi${{< /math >}} | Policy | Macro policy: {{< math >}}$\pi(A_k \| X_k)${{< /math >}} | §1.2 |
| {{< math >}}$R_{k+1}${{< /math >}} | Scalar | Reward received after action {{< math >}}$A_k${{< /math >}} | §1.3 |
| {{< math >}}$G_k${{< /math >}} | Scalar | Return: {{< math >}}$\sum_{t=0}^{\infty} \gamma^t R_{k+t+1}${{< /math >}} | §1.3 |
| {{< math >}}$\gamma${{< /math >}} | Scalar | Discount factor, {{< math >}}$\gamma \in [0, 1]${{< /math >}} | §1.3 |
| {{< math >}}$r${{< /math >}} | Function | Reward function: {{< math >}}$R_{k+1} = r(X_k, th_k, A_k, O_{k+1})${{< /math >}} | §1.3 |
| {{< math >}}$\pi^*${{< /math >}} | Policy | Optimal policy maximizing expected return | §1.3 |
| {{< math >}}$\mathbf{v}_k${{< /math >}} | Sequence | Token sequence at turn {{< math >}}$k${{< /math >}}: {{< math >}}$(v_{k,1}, \ldots, v_{k,T_k})${{< /math >}} | §1.4 |
| {{< math >}}$v_{k,t}${{< /math >}} | Token | The {{< math >}}$t${{< /math >}}-th token in turn {{< math >}}$k${{< /math >}}'s sequence | §1.4 |
| {{< math >}}$T_k${{< /math >}} | Integer | Length of token sequence at turn {{< math >}}$k${{< /math >}} | §1.4 |
| {{< math >}}$p_{\theta}${{< /math >}} | Distribution | LLM policy (micro): {{< math >}}$p_{\theta}(\mathbf{v}_k \| X_k)${{< /math >}} | §1.4 |
| {{< math >}}$\theta${{< /math >}} | Parameters | LLM model parameters | §1.4 |
| {{< math >}}$\text{Decode}${{< /math >}} | Function | Token sequence to string: {{< math >}}$th_k = \text{Decode}(\mathbf{v}_k)${{< /math >}} | §1.4 |
| {{< math >}}$\mathcal{V}${{< /math >}} | Set | Vocabulary: set of all tokens (active design choice) | §1.1 |
| {{< math >}}$\mathcal{V}^*${{< /math >}} | Set | All possible token sequences (Kleene star over {{< math >}}$\mathcal{V}${{< /math >}}) | §1.4 |
| {{< math >}}$\mathcal{A}${{< /math >}} | Set | Action space (set of all possible actions) | §1.4 |
| {{< math >}}$\tau${{< /math >}} | Trajectory | Complete interaction sequence | §1.5 |
| {{< math >}}$T${{< /math >}} | Integer | Final turn index in trajectory (trajectory has turns {{< math >}}$0, \ldots, T${{< /math >}}) | §1.5 |
| {{< math >}}$P(\tau \| \theta, \rho)${{< /math >}} | Probability | Probability of trajectory {{< math >}}$\tau${{< /math >}} under policy {{< math >}}$\theta${{< /math >}} and environment {{< math >}}$\rho${{< /math >}} | §1.5 |

**Notation Conventions**:
- **Subscript {{< math >}}$k${{< /math >}}**: Refers to turn/step index in the interaction sequence
- **Subscript {{< math >}}$t${{< /math >}}**: Refers to token position within a single turn's generation
- **Uppercase** ({{< math >}}$A, O, R, G, X${{< /math >}}): Random variables or their realizations
- **Lowercase** ({{< math >}}$th, r, f${{< /math >}}): Functions or deterministic quantities
- **Bold** ({{< math >}}$\mathbf{v}${{< /math >}}): Sequences or vectors
- **Calligraphic** ({{< math >}}$\mathcal{E}, \mathcal{A}, \mathcal{V}${{< /math >}}): Sets or abstract spaces

---

## Summary: The Essence of Language Agents

**What we've established:**

Language agents have **three fundamental capabilities** unavailable to traditional RL:

1. **Active vocabulary management** ({{< math >}}$\mathcal{V}${{< /math >}}): Agents design their own "language" for expressing actions, not constrained to fixed action spaces
2. **Active context management** ({{< math >}}$f_{\text{agent}}${{< /math >}}): Agents control what information to retain through learned state compression (memory management)
3. **Two-layer decision structure** ({{< math >}}$\pi \circ p_{\theta}${{< /math >}}): Macro policy implemented through micro token generation, enabling Chain-of-Thought reasoning

**Why language is the universal interface:** Language is uniquely suited for all three capabilities because it is **compositional and compressible**. It fluidly expresses both high-level reasoning (for {{< math >}}$th_k${{< /math >}}) and low-level instructions (for {{< math >}}$A_k${{< /math >}}), while serving as its own medium for memory compression ({{< math >}}$f_{\text{agent}}${{< /math >}}). No other modality unifies expressiveness, interpretability, and compression in one coherent framework.

**Why existing frameworks miss this:**

- ReAct demonstrates interleaving reasoning and acting works empirically but provides no mathematical framework
- Traditional RL fixes action spaces, state representations, and decision layers; language agents make all three active design choices
- State compression isn't optional—it's physically necessary (computability constraint)

**The implications:**

- **Parser design is critical**: It bridges rich thought to structured action, but creates credit assignment nightmares
- **Context management = policy**: Bad {{< math >}}$f_{\text{agent}}${{< /math >}} creates bottlenecks no LLM power can fix
- **Two-layer structure enables and constrains**: Unlocks CoT reasoning but adds complexity to credit assignment

**The bottom line:**

Success requires getting three design choices right:
1. Can your agent say what it needs to say? (Vocabulary design: {{< math >}}$\mathcal{V}${{< /math >}})
2. Can your agent remember what it needs to remember? (Context management: {{< math >}}$f_{\text{agent}}${{< /math >}})
3. Can your agent think before it acts? (Parser robustness: {{< math >}}$th_k \to A_k${{< /math >}})

Get these right, and complex reasoning follows. Get them wrong, and no amount of model scale will save you.

---

## References

**Reinforcement Learning Foundations**:
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.

**Language Agents**:
- Yao, S., et al. (2023). "ReAct: Synergizing Reasoning and Acting in Language Models." *ICLR*.
