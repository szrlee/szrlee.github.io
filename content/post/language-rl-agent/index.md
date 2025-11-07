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

## Introduction

**Why do some agents succeed while others fail?** While frameworks like ReAct (Yao et al., 2023) show that interleaving reasoning and acting works, they don't explain *why* it works or when it fails. We lack a principled understanding of what makes language agents fundamentally different from traditional RL agents.

This post establishes a mathematical framework revealing that **language serves as a universal interface** for RL agents, providing fundamental capabilities absent in traditional RL:

1. **Active vocabulary management** ($\mathcal{V}$): Agents design their own "language" for expressing actions, not constrained to fixed action spaces
2. **Active context management** ($f_{\text{agent}}$): Agents control what information to retain through learned state compression (memory management)
3. **Two-layer decision structure**: Macro policy $\pi(A_k | X_k)$ implemented through micro token generation $p_{\theta}(\mathbf{v} | X_k)$, enabling Chain-of-Thought reasoning

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

**Internal Thought** ($th_k$): At turn $k$, the agent generates free-form reasoning, planning, or intermediate text based on its internal state. This is the agent's private, internal information—the first step in its decision-making process.

**External Action** ($A_k$): The structured instruction extracted from $th_k$ through a deterministic Parser function that will affect the external world:
$$A_k = \text{Parser}(th_k)$$

**External Observation** ($O_k$): After action $A_{k-1}$ acts on the environment, the information returned to the agent.

#### Two Types of History

**1. External History** ($H_k^{\text{ext}}$):

The objectively occurring, externally observable event sequence—the "ground truth" of world evolution:

$$H_k^{\text{ext}} = (A_0, O_1, A_1, O_2, \ldots, A_{k-1}, O_k)$$

**2. Agent-Centric History** ($H_k^{\text{agent}}$):

The complete information accessible to the agent during decision-making, including its internal thought process:

$$H_k^{\text{agent}} = (\text{system\_prompt}, O_0, th_0, A_0, O_1, th_1, A_1, O_2, \ldots, th_{k-1}, A_{k-1}, O_k)$$

This history is the complete information foundation for the agent's learning and construction of its internal mental model.

#### Environment

**Environment** ($\mathcal{E}$): The external environment's behavior is characterized by a probabilistic transition function $\rho$, which gives the probability of the next observation based on external history and the agent's action:

$$\rho(O_{k+1} | H_k^{\text{ext}}, A_k)$$

In our framework, $\rho$ is externally given and typically unknown—a "black box."

#### Notation Summary

The table below summarizes the key notation introduced in this section and used throughout the post:

| Symbol | Type | Definition | First Use |
|--------|------|------------|-----------|
| $k$ | Index | Turn/step index in the interaction | §1.1 |
| $th_k$ | Text | Internal thought at turn $k$ (free-form reasoning) | §1.1 |
| $A_k$ | Action | External action at turn $k$ (structured command) | §1.1 |
| $O_k$ | Observation | External observation at turn $k$ (environment feedback) | §1.1 |
| $H_k^{\text{ext}}$ | Sequence | External history: $(A_0, O_1, \ldots, A_{k-1}, O_k)$ | §1.1 |
| $H_k^{\text{agent}}$ | Sequence | Agent-centric history: includes thoughts $th_i$ | §1.1 |
| $\mathcal{E}$ | Environment | The external world/task environment | §1.1 |
| $\rho$ | Function | Environment transition: $\rho(O_{k+1} \| H_k^{\text{ext}}, A_k)$ | §1.1 |
| $\text{Parser}$ | Function | Deterministic mapping: $th_k \to A_k$ | §1.1 |
| $X_k$ | State | Agent state at turn $k$ (compressed history) | §1.2 |
| $f_{\text{agent}}$ | Function | State update: $X_{k+1} = f_{\text{agent}}(X_k, th_k, A_k, O_{k+1})$ | §1.2 |
| $\pi$ | Policy | Macro policy: $\pi(A_k \| X_k)$ | §1.2 |
| $R_{k+1}$ | Scalar | Reward received after action $A_k$ | §1.3 |
| $G_k$ | Scalar | Return: $\sum_{t=0}^{\infty} \gamma^t R_{k+t+1}$ | §1.3 |
| $\gamma$ | Scalar | Discount factor, $\gamma \in [0, 1]$ | §1.3 |
| $r$ | Function | Reward function: $R_{k+1} = r(X_k, th_k, A_k, O_{k+1})$ | §1.3 |
| $\pi^*$ | Policy | Optimal policy maximizing expected return | §1.3 |
| $\mathbf{v}_k$ | Sequence | Token sequence at turn $k$: $(v_{k,1}, \ldots, v_{k,T_k})$ | §1.4 |
| $v_{k,t}$ | Token | The $t$-th token in turn $k$'s sequence | §1.4 |
| $T_k$ | Integer | Length of token sequence at turn $k$ | §1.4 |
| $p_{\theta}$ | Distribution | LLM policy (micro): $p_{\theta}(\mathbf{v}_k \| X_k)$ | §1.4 |
| $\theta$ | Parameters | LLM model parameters | §1.4 |
| $\text{Decode}$ | Function | Token sequence to string: $th_k = \text{Decode}(\mathbf{v}_k)$ | §1.4 |
| $\mathcal{V}$ | Set | Vocabulary: set of all tokens (active design choice) | §1.1 |
| $\mathcal{V}^*$ | Set | All possible token sequences (Kleene star over $\mathcal{V}$) | §1.4 |
| $\mathcal{A}$ | Set | Action space (set of all possible actions) | §1.4 |
| $\tau$ | Trajectory | Complete interaction sequence | §1.5 |
| $T$ | Integer | Final turn index in trajectory (trajectory has turns $0, \ldots, T$) | §1.5 |
| $P(\tau \| \theta, \rho)$ | Probability | Probability of trajectory $\tau$ under policy $\theta$ and environment $\rho$ | §1.5 |

**Notation Conventions**:
- **Subscript $k$**: Refers to turn/step index in the interaction sequence
- **Subscript $t$**: Refers to token position within a single turn's generation
- **Uppercase** ($A, O, R, G, X$): Random variables or their realizations
- **Lowercase** ($th, r, f$): Functions or deterministic quantities
- **Bold** ($\mathbf{v}$): Sequences or vectors
- **Calligraphic** ($\mathcal{E}, \mathcal{A}, \mathcal{V}$): Sets or abstract spaces

#### Key Design Choices: Why Language Changes Everything

**Here's what traditional RL agents cannot do:** In classic RL, the action space $\mathcal{A}$ is fixed. In Atari, you have {up, down, left, right, fire}. In chess, you have legal moves. The agent optimizes policy $\pi(a|s)$ over this frozen set.

**Language agents break this constraint.** They have two degrees of freedom unavailable to traditional RL:

**1. Active Vocabulary Management ($\mathcal{V}$)**

Unlike fixed action spaces in traditional RL, language agents can **actively design and manage their vocabulary**—the set of tokens they use to express thoughts and actions. This includes:

- **Domain-specific tokens**: Extending the vocabulary with task-relevant tokens (e.g., function names, domain concepts)
- **Structured output vocabularies**: Designing token sets that naturally express structured actions (e.g., JSON, XML tags)
- **Hierarchical vocabularies**: Multi-level token sets enabling both high-level planning and low-level execution

The vocabulary $\mathcal{V}$ directly determines what can be expressed in $th_k$, which through the Parser determines the effective action space $\mathcal{A}$. This is a **learnable design choice**, not a fixed constraint.

**2. Active Context Management ($X_k$ via $f_{\text{agent}}$)**

Language agents must **actively manage what information to retain** in their state $X_k$ at each turn. The state update function $f_{\text{agent}}$ is not just a passive compression—it's an **active policy** for context management:

- **What to remember**: Selecting which past observations, thoughts, and actions to retain
- **What to forget**: Discarding irrelevant information to stay within context limits
- **How to compress**: Choosing representations (verbatim, summarized, structured)
- **When to retrieve**: Deciding when to access external memory vs. internal context

The design of $f_{\text{agent}}$ is as important as the policy $\pi$ itself—poor context management creates an information bottleneck that no amount of model capacity can overcome.

**Why this matters**: Both vocabulary and context are expressed in natural language—the same medium humans use. This makes language agents uniquely:
- **Interpretable**: You can read what the agent thinks and why
- **Debuggable**: You can pinpoint where reasoning fails
- **Jointly optimizable**: Vocabulary and context can be improved together

This is why language is a universal interface: it unifies expressiveness (vocabulary) and memory (context) in one coherent framework.

---

### 1.2 Why State Compression is Inevitable

#### The Dilemma

**In theory**, an agent could make perfect decisions using complete history: $\pi(A_k | H_k^{\text{agent}})$, where $H_k^{\text{agent}}$ contains every thought, action, and observation from turn 0 to $k$.

**In practice**, this is impossible. As $k$ grows, $H_k^{\text{agent}}$ grows without bound. This isn't merely about computational cost (though Transformer's $O(|H_k^{\text{agent}}|^2)$ scaling hurts). It's about **computability**: no finite device can process infinite input.

**The implication is stark**: State compression is not a design choice. It's a physical necessity.

#### Inevitable Conclusion: Agent State ($X_k$)

To make decisions possible, the agent **must** compress the infinitely growing agent-centric history $H_k^{\text{agent}}$ into a fixed-size internal representation. We call this representation the **agent state** $X_k$:

$$X_k \approx \text{compress}(H_k^{\text{agent}})$$

This state $X_k$ is the agent's "mental model" or "working memory" of the world that it relies on for decision-making. From this point, the agent's policy is based on this computable state:

$$\pi(A_k | X_k)$$

#### The Critical Function: State Update ($f_{\text{agent}}$)

State must evolve as new information arrives:

$$X_{k+1} = f_{\text{agent}}(X_k, th_k, A_k, O_{k+1})$$

**This function is the agent's memory policy**—deciding what to remember and what to forget. Its design is as crucial as the action policy $\pi$ itself. In fact, optimizing $f_{\text{agent}}$ is a **meta-learning problem**: learning a compression policy that preserves task-relevant information to maximize the primary policy's expected return.

**Why it matters**: Compression is lossy. $X_k$'s quality sets the performance ceiling. Even the world's most powerful LLM cannot compensate for bad memory management.

#### Practical Paradigms for $f_{\text{agent}}$

**Sliding Window**: The simplest approach—$X_k$ only contains the most recent $N$ turns of $(th, A, O)$ tuples.

**Language Model-based Summarization**: Use language model calls to periodically "compress" old $X_k$ and new $(th_k, A_k, O_{k+1})$.

**Structured Memory**: Extract information from $H_k^{\text{agent}}$ and store it in an external vector database or knowledge graph. Here, $X_k$ is a complex object containing dialogue summaries, entity lists, etc., and $f_{\text{agent}}$ defines how to read and write this structured memory.

**Learnable Memory Modules**: Advanced approaches using neural architectures that jointly optimize memory selection with the policy itself.

---

### 1.3 Optimization Objective and Reward Formation

The agent's behavior is not random—it's driven by a clear objective: maximizing long-term cumulative reward.

#### Ultimate Goal: Maximize Return

The agent's ultimate goal is to maximize a long-term value called **return** ($G_k$), which is the cumulative sum of all future rewards considering time discount factor $\gamma \in [0, 1]$:

$$G_k = \sum_{t=0}^{\infty} \gamma^t R_{k+t+1} = R_{k+1} + \gamma R_{k+2} + \gamma^2 R_{k+3} + \ldots$$

The optimal policy $\pi^*$ aims to maximize the expected value of this return:

$$\pi^* = \arg\max_{\pi} \mathbb{E}[G_k | \pi]$$

#### Reward Formulation

In any practically operational system, reward $R_{k+1}$ calculation must rely on information the agent can access. We define reward formation as a reward function $r$, with the most general form:

$$R_{k+1} = r(X_k, th_k, A_k, O_{k+1})$$

**This definition is crucial** because it reveals the dual core role of state $X_k$ and thought $th_k$: they are not only decision inputs but also evaluation (reward function $r$) inputs. A low-quality state $X_k$ (poor compression of $H_k^{\text{agent}}$) means the reward $R_{k+1} = r(X_k, th_k, A_k, O_{k+1})$ is computed from **partially observed information**. This creates a **non-Markovian reward signal**—the observed reward becomes a biased estimate of the true reward $r(H_k^{\text{agent}}, th_k, A_k, O_{k+1})$ that would be computed from complete history. This is analogous to the classic POMDP problem, but applied to the reward function itself: poor state compression degrades not just the policy, but the learning signal that guides it.

---

### 1.4 The Two-Layer Decision Structure: Thought Before Action

**The key difference**: Traditional RL agents directly output actions. Language agents first generate thoughts (natural language reasoning), then parse actions from those thoughts.

This creates a two-layer structure with profound implications.

#### Layer 1: Macro Task Layer ($M_{\text{turn}}$)

This layer completely inherits from the general framework—it's the level where the agent conducts meaningful interaction with the environment. Its state is macro state $X_k$, action is macro action $A_k$. Its ultimate goal is learning an optimal macro policy $\pi^*(A_k | X_k)$ to maximize long-term return $G_k$.

#### Layer 2: Micro Generation Layer ($M_{\text{micro}}$)

This layer's core function is to **implement** the macro policy $\pi$. It describes how the agent's "thought" is generated token by token by the LLM.

**Basic Units**: We must distinguish two concepts:

- **Token sequence** ($\mathbf{v}_k$): The fundamental data structure directly output by LLM policy $p_{\theta}$, a sequence composed of tokens: $\mathbf{v}_k = (v_{k,1}, v_{k,2}, \ldots, v_{k,T_k})$

- **Thought string** ($th_k$): The human-readable text string converted from token sequence $\mathbf{v}_k$ through decode function: $th_k = \text{Decode}(\mathbf{v}_k)$

**Note**: When referring to $th_k$, without special indication, it can refer to either the thought string or the $\mathbf{v}_k$ generated by the LLM at that time. The mapping from token to thought string is many-to-one, which may present technical issues.

**Generation Process**: This process is controlled by LLM parameters $\theta$, defining the probability of generating a specific token sequence $\mathbf{v}_k$ given state $X_k$. For autoregressive models:

$$p_{\theta}(\mathbf{v}_k | X_k) = \prod_{t=1}^{T_k} p_{\theta}(v_{k,t} | X_k, v_{k,1:t-1})$$

where $v_{k,1:t-1} = (v_{k,1}, \ldots, v_{k,t-1})$ denotes the token history up to position $t-1$ in turn $k$.

#### Connecting Macro and Micro

The macro policy connects to micro generation through the following core equation. This equation is built on token sequence probabilities, precisely handling the characteristic that "one action can be implemented by multiple thoughts (multiple token sequences)":

$$\pi(A_k | X_k) \equiv \sum_{\mathbf{v} \in \mathcal{V}^*} \mathbf{1} [ \text{Parser}(\text{Decode}(\mathbf{v})) = A_k ] \cdot p_{\theta}(\mathbf{v} | X_k)$$

where $\mathcal{V}^*$ represents all possible token sequence sets. This formula shows that a macro action's probability is the sum of probabilities of all "token sequences that can be decoded and parsed into that action."

**Core Learning Task**: The essence of agent training is using experience data obtained from environment interaction (i.e., sequences containing $(X_k, th_k(\mathbf{v}_k), A_k, O_{k+1}, R_{k+1})$ information) to adjust micro generation layer parameters $\theta$, thereby optimizing macro layer policy $\pi$, ultimately achieving the goal of maximizing long-term return. The challenge: gradients must flow through the non-differentiable $\text{Parser}$ function, requiring sampling-based RL methods like REINFORCE.

**Generality**: This two-layer definition has universality and can cover multiple generation model architectures:

- **Autoregressive models**: As defined above, generating $\mathbf{v}_k$ by predicting tokens one by one
- **Diffusion models**: Generating entire token sequence $\mathbf{v}_k$'s representation through iterative denoising from noise
- **Other models**: Such as Tokenizer-free models, etc., where the core idea applies equally

#### Why This Structure Matters

**The opportunity**: Separating thought ($th_k$) from execution ($A_k$) unlocks Chain-of-Thought reasoning—complex planning without hardcoding logic into action space $\mathcal{A}$.

**The bottleneck**: The Parser is both bridge and weakness. Poor parsing wastes perfect reasoning. Robust Parser design is critical.

**The credit assignment nightmare**: When $A_k$ fails, where's the blame?
- Was $th_k$ wrong conceptually?
- Was $th_k$ right but Parser-incompatible?
- Was $A_k$ actually fine but environment-inappropriate?

This three-way ambiguity makes learning harder than single-layer RL. Technically, this manifests as a **high-variance gradient problem**: a single action $A_k$ corresponds to many valid token sequences, but policy gradient methods only sample one, leading to high variance in gradient estimates.

**The silver lining**: Reverse parsing enables data augmentation. Given good action $A_k$, generate multiple thought chains $th_k$ that lead to it. This creates rich $(X_k, th_k, A_k)$ training data, teaching the model "how to think to act correctly."

---

### 1.5 Multi-turn Language Agent Trajectory Probability Modeling

A trajectory, typically denoted $\tau$, is a complete sequence of events produced by agent-environment interaction. In a complete trajectory, there are two core sources of randomness:

1. **Agent's decisions**: Under given state $X_k$, which "thought" $\mathbf{v}_k$ the agent generates is a probabilistic event determined by its internal LLM policy $p_{\theta}$

2. **Environment's responses**: After the agent executes action $A_k$, which "observation" $O_{k+1}$ the environment produces is a probabilistic event determined by the environment's transition function $\rho$

Other steps, such as parsing action $A_k$ from thought $\mathbf{v}_k$ (through `Parser` function), calculating reward $R_{k+1}$ (through `r` function), and updating agent state $X_{k+1}$ (through $f_{\text{agent}}$ function, assumed deterministic) are deterministic.

#### Trajectory Definition (Agent-Centric History)

First, we define a trajectory $\tau$ of length $T$ turns as a sequence of core events starting from the initial state:

$$\tau = (X_0, \mathbf{v}_0, A_0, O_1, X_1, \mathbf{v}_1, A_1, O_2, \ldots, X_T, \mathbf{v}_T, A_T, O_{T+1})$$

where:
- $X_k$ is the agent's state at turn $k$ (e.g., all history prefixes)
- $\mathbf{v}_k$ is the token sequence generated by the LLM at turn $k$: $\mathbf{v}_k = (v_{k,1}, \ldots, v_{k, |\mathbf{v}_k|})$
- $A_k$ is the action parsed from $\mathbf{v}_k$: $A_k = \text{Parser}(\text{Decode}(\mathbf{v}_k))$
- $O_{k+1}$ is the environment's observation response to action $A_k$

#### Trajectory Probability

$$P(\tau | \theta, \rho) = p(X_0) \prod_{k=0}^{T} \left[ \underbrace{p_{\theta}(\mathbf{v}_k | X_k)}_{\text{Agent's Policy}} \cdot \underbrace{\rho(O_{k+1} | H_k^{\text{ext}}, A_k)}_{\text{Environment's Dynamics}} \right]$$

where:

$$p_{\theta}(\mathbf{v}_k | X_k) = \prod_{t=1}^{|\mathbf{v}_k|}p_{\theta}(v_{k,t} | X_k, v_{k,1:t-1})$$

#### Trajectory Probability Ratio

The trajectory probability ratio of two models $\theta, \theta'$ is:

$$\frac{P(\tau | \theta, \rho)}{P(\tau| \theta', \rho)} = \frac{p(X_0) \prod_{k=0}^{T} \left[ {p_{\theta}(\mathbf{v}_k | X_k)} \cdot {\rho(O_{k+1} | H_k^{\text{ext}}, A_k)} \right]} {p(X_0) \prod_{k=0}^{T} \left[ {p_{\theta'}(\mathbf{v}_k | X_k)} \cdot {\rho(O_{k+1} | H_k^{\text{ext}}, A_k)} \right]}$$

After canceling common terms:

$$\frac{P(\tau | \theta, \rho)}{P(\tau| \theta', \rho)} = \prod_{k = 0}^T \frac{p_{\theta}(\mathbf{v}_k | X_k)}{p_{\theta'}(\mathbf{v}_k | X_k)} = \prod_{k=0}^T \prod_{t = 1}^{|\mathbf{v}_k|} \frac{p_{\theta}(v_{k,t} | X_k, v_{k,1:t-1})}{p_{\theta'}(v_{k,t} | X_k, v_{k,1:t-1})}$$

This factorization enables RL algorithms (PPO, GRPO, etc.) to optimize $\theta$ by computing importance weights at either the turn level or token level.

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

- **Implements**: Encapsulates world laws $\rho(O_{k+1} | H_k^{\text{ext}}, A_k)$
- **Behavior**: Receives a macro action $A_k$, executes world state transition
- **Returns**:
  - `O_{k+1}` (Observation): Environment's next observation
  - `bool` (Terminated): Whether episode terminates due to task success/failure
  - `bool` (Truncated): Whether episode is cut short due to external limits (e.g., timeout)
  - `Dict` (Info): Additional information for debugging
- **Key**: This method does **not** return reward

**reset() → Tuple[O_0, Dict]**

- **Behavior**: Reset environment, start a new interaction episode. Returns initial observation

**action_space / observation_space**

- **Behavior**: Define legal action $A_k$ and observation $O_k$ structure and types, following gymnasium.Space specification

---

#### Agent Interface

The agent is the system's core, integrating five major functions: **perception, thinking, action, evaluation, and learning**.

**generate_thought_and_action(state: X_k) → Tuple[th_k, A_k]**

- **Implements**: Encapsulates the complete decision chain from state to action
- **Internal flow**:
  1. **Thought Generation**: Call micro generation layer $p_{\theta}(th_k | X_k)$ to generate coherent thought text $th_k$
  2. **Action Parsing**: Call deterministic $\text{Parser}(th_k)$ function to extract structured macro action $A_k$ from thought
- **Returns**: $(th_k, A_k)$ tuple—complete thought chain and final action for this decision

**evaluate_step(X_k, th_k, A_k, O_{k+1}) → R_{k+1}**

- **Implements**: Encapsulates reward function $r(X_k, th_k, A_k, O_{k+1})$. This is the core method actively called by the agent
- **Behavior**: The agent conducts **self-evaluation** based on its pre-decision state $X_k$, complete thought process $th_k$, executed action $A_k$, and environment-given consequence $O_{k+1}$, calculating reward value $R_{k+1}$ for this step
- **Examples**:
  - A software engineering agent's `evaluate_step` might execute unit tests and calculate reward based on test pass rate
  - A dialogue agent's `evaluate_step` might call a sentiment analysis model to judge user satisfaction as reward

**learn(trajectory_batch: List[Tuple])**

- **Implements**: Connects to a specific reinforcement learning backend (e.g., [VeRL](https://github.com/volcengine/verl) - Volcano Engine Reinforcement Learning for LLMs)
- **Behavior**: Receives a batch of complete experience trajectories, where each trajectory point contains $(X_k, th_k, A_k, R_{k+1}, X_{k+1}, ...)$. It formats this data and passes it to the RL backend's optimizer (e.g., PPO, GRPO, ReMax, etc.) to perform updates to model parameters $\theta$

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

#### Environment ($\mathcal{E}$)

In SWE-Benchmark settings, the environment $\mathcal{E}$ is a highly isolated and standardized code repository.

- **Implementation**: Each task instance runs in a sandbox similar to bubble wrap, providing filesystem and network isolation. This ensures the agent's actions won't accidentally affect external systems and guarantees experiment reproducibility

- **World State Transition** ($\rho(O_{k+1} | H_k^{\text{ext}}, A_k)$): The environment's physical laws are defined by the underlying operating system (usually Linux) and pre-installed software (like git, python, pytest). When the agent executes an action $A_k$ (e.g., a bash command), the environment undergoes state transition according to these laws (e.g., files are modified, processes are created) and captures stdout and stderr produced by that action as the next observation $O_{k+1}$

#### Agent-Environment Interface Correspondence

**2.1 Macro Action ($A_k$) and Action Space ($\mathcal{A}$)**

The agent's macro actions $A_k$ are a series of predefined, structured tool calls. Action space $\mathcal{A}$ is the set of all these legal tool calls. Typical tools include:

- **bash**: Execute a shell command
  - Formal representation: $A_k = (\text{tool: "bash"}, \{\text{command: string}\})$
  - Example: $A_k = (\text{bash}, \{\text{command: "ls -F /testbed"}\})$

- **edit**: Modify files—itself is a composite tool
  - View file: $A_k = (\text{edit}, \{\text{command: "view", path: string, view\_range: [int, int]}\})$
  - String replacement: $A_k = (\text{edit}, \{\text{command: "str\_replace", path: string, old\_str: string, new\_str: string}\})$
  - Insert code: $A_k = (\text{edit}, \{\text{command: "insert", path: string, new\_str: string, insert\_line: int}\})$

- **submit**: Terminate task and submit final solution
  - Formal representation: $A_k = (\text{submit}, \{\})$

**2.2 Thought ($th_k$), Action Parsing (Parser), and Observation ($O_k$)**

This is the core link connecting micro generation with macro interaction.

- **Internal Thought** ($th_k$): The complete text generated by the LLM given current state $X_k$. It usually contains reasoning process, analysis of current situation, and next step plan. Example:

```
Let's look at the utils module, which seems to handle parameter parsing:
<tool_call>
<function>edit</function>
<parameter name="command">view</parameter>
<parameter name="path">/testbed/spectree/utils.py</parameter>
</tool_call>
```

- **Action Parsing** ($A_k = \text{Parser}(th_k)$): Parser is a deterministic function responsible for extracting structured macro action $A_k$ from free-form thought text $th_k$. In practice, this is typically implemented through regular expressions or XML/JSON parsing to identify and extract content in `<tool_call>` or similar tags. If parsing fails, it may produce a special "no-op" or "error" action

- **External Observation** ($O_k$): Information returned to the agent after the environment executes action $A_{k-1}$. In SWE-Benchmark, this is usually the combination of bash command stdout and stderr. To avoid information overload, returned observations are typically truncated or abbreviated. A well-designed tool should have clear success or failure return information so the agent can understand the consequences of its actions

#### State Construction and Update ($X_k$ and $f_{\text{agent}}$)

- **Agent-Centric History** ($H_k^{\text{agent}}$): This is the complete information source for agent decision-making. In SWE-Benchmark practice, it's usually organized as a dialogue-style record:

$$H_k^{\text{agent}} = (\text{system\_prompt}, O_0, th_0, A_0, O_1, th_1, A_1, O_2, \ldots, th_{k-1}, A_{k-1}, O_k)$$

where $O_0$ contains the initial task description (Problem Statement) and environment introduction.

- **State Update** ($X_{k+1} = f_{\text{agent}}(X_k, th_k, A_k, O_{k+1})$): Due to the "history explosion" problem, complete $H_k^{\text{agent}}$ cannot directly serve as LLM input. The agent's state $X_k$ is actually the Prompt input to the LLM. $f_{\text{agent}}$ is the strategy for constructing this Prompt from history—the memory and forgetting mechanism. Common implementations include:

  - **Sliding Window**: A simple strategy, e.g., SWE-agent-lm only retains the most recent 5 observations $O_i$ and complete thought-action history $(th, A)$ when constructing next state $X_k$

  - **Intelligent Compression**: More advanced methods, like Claude model—when history approaches context window limit, it retains about 30% of key historical steps and summarizes or compresses the remaining 70%

  - **Complete History**: In early SWE-agent or training stages, sometimes the entire $H_k^{\text{agent}}$ is concatenated as $X_k$. While this is information-lossless, it's extremely costly and limited by model context length

#### Optimization Objective and Reward Formation ($r$)

- **Reward Function** ($R_{k+1} = r(X_k, th_k, A_k, O_{k+1})$): In SWE-Benchmark, reward implementation is **sparse and delayed**.

  - For most intermediate steps (like bash, edit), reward $R_{k+1}$ is constantly 0. The agent receives no explicit right/wrong signal during exploration
  - Only when the agent executes final action $A_k = (\text{submit}, \{\})$ is a non-zero reward calculation triggered

- **evaluate_step Implementation**: This reward calculation process is implemented by the `agent.evaluate_step(...)` method in our interface abstraction. In SWE-Benchmark, this method executes an evaluation script (eval.sh) with the following specific flow:

  1. **Environment Reset**: Script first uses `git reset --hard` and `git clean -fd` to restore code repository to clean initial version
  2. **Apply Patch**: Apply all code modifications generated by the agent during interaction (in .patch file form) to the code repository
  3. **Run Tests**: Activate virtual environment, then use testing frameworks like pytest to run predefined test cases
  4. **Parse Results**: Script captures pytest output logs (eval log)
  5. **Calculate Reward**: By parsing logs, determine if tests passed. Final reward $R_{\text{final}}$ is given based on this result, e.g.:
     - All tests pass: $R_{\text{final}} = +1$
     - Tests fail: $R_{\text{final}} = -1$ (or other value less than 1)

This process perfectly interprets our definition: reward $R_{k+1}$ is generated by the agent (its evaluation module) after action $A_k$ acts on the environment and produces observation $O_{k+1}$ (here referring to test results). This sparse reward characteristic also brings enormous credit assignment challenges to reinforcement learning algorithms (like PPO, ReMax).

**A prescriptive insight from our framework:** This sparse reward design is suboptimal. Our framework suggests a more effective agent could leverage its `evaluate_step` capability to **generate denser, intermediate rewards**. For example, after an `edit` action, the agent could self-evaluate by running a linter, static type-checker, or unit tests on modified functions—generating internal reward signals $R_{k+1} > 0$ for syntactically correct code or passing local tests, even before the final submission. This demonstrates how our framework provides not just a descriptive model, but a blueprint for designing more sample-efficient agents.

---

## Summary: The Essence of Language Agents

**What we've established:**

Language agents have **three fundamental capabilities** unavailable to traditional RL:

1. **Active vocabulary management** ($\mathcal{V}$): Agents design their own "language" for expressing actions, not constrained to fixed action spaces
2. **Active context management** ($f_{\text{agent}}$): Agents control what information to retain through learned state compression (memory management)
3. **Two-layer decision structure** ($\pi \circ p_{\theta}$): Macro policy implemented through micro token generation, enabling Chain-of-Thought reasoning

**Why language is the universal interface:** Language is uniquely suited for all three capabilities because it is **compositional and compressible**. It fluidly expresses both high-level reasoning (for $th_k$) and low-level instructions (for $A_k$), while serving as its own medium for memory compression ($f_{\text{agent}}$). No other modality unifies expressiveness, interpretability, and compression in one coherent framework.

**Why existing frameworks miss this:**

- ReAct demonstrates interleaving reasoning and acting works empirically but provides no mathematical framework
- Traditional RL fixes action spaces, state representations, and decision layers; language agents make all three active design choices
- State compression isn't optional—it's physically necessary (computability constraint)

**The implications:**

- **Parser design is critical**: It bridges rich thought to structured action, but creates credit assignment nightmares
- **Context management = policy**: Bad $f_{\text{agent}}$ creates bottlenecks no LLM power can fix
- **Two-layer structure enables and constrains**: Unlocks CoT reasoning but adds complexity to credit assignment

**The bottom line:**

Success requires getting three design choices right:
1. Can your agent say what it needs to say? (Vocabulary design: $\mathcal{V}$)
2. Can your agent remember what it needs to remember? (Context management: $f_{\text{agent}}$)
3. Can your agent think before it acts? (Parser robustness: $th_k \to A_k$)

Get these right, and complex reasoning follows. Get them wrong, and no amount of model scale will save you.

---

## References

**Reinforcement Learning Foundations**:
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.

**Language Agents**:
- Yao, S., et al. (2023). "ReAct: Synergizing Reasoning and Acting in Language Models." *ICLR*.

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
