---
title: 'The Stability Gap: Why Top-K Routing Breaks RL Optimization'
subtitle: 'How Discrete Expert Selection Creates Pathological Optimization Landscapes'
summary: 'A rigorous mathematical analysis showing that Top-K expert routing in Mixture of Experts creates two fundamental pathologies: gradient blackout (zero gradients almost everywhere) and first-order approximation failure (discontinuous policy mapping), explaining why MoE-RL training can be unstable.'
authors:
  - admin
tags:
  - Reinforcement Learning
  - Mixture of Experts
  - Large Language Models
  - LLM-RL
  - Optimization
  - Training Dynamics
categories:
  - Research
  - Theory
date: '2025-12-07T00:00:00Z'
lastmod: '2025-12-07T00:00:00Z'
featured: true
draft: false
math: true
toc: true

# Featured image
image:
  caption: 'Top-K Routing Stability Gap'
  focal_point: ''
  preview_only: false

# Projects (optional)
projects: []
---

## The Problem

Training Mixture of Experts (MoE) language models with Reinforcement Learning can be unstable. While dense LLMs have continuous and differentiable policy mappings, MoE-based models like Mixtral, DeepSeek-MoE, and Qwen-MoE introduce the **Top-K operator**—a discrete switching mechanism that creates discontinuities in the optimization landscape.

This discreteness introduces two fundamental mathematical pathologies that break standard RL assumptions used in PPO, GRPO, and other LLM-RL algorithms.

---

## TL;DR: The Two Pathologies

**Challenge 1: Gradient Blackout.** The gradient of the token distribution {{< math >}}$\pi_\theta(y_t | x, y_{\lt t})${{< /math >}} with respect to unselected experts' logits is exactly zero almost everywhere. Unlike non-smooth convex functions where subgradients guide optimization, the Top-K landscape offers no directional information on how to switch to a better expert.

**Challenge 2: First-Order Approximation Failure.** Modern LLM-RL algorithms (PPO, GRPO) rely on a surrogate objective that approximates the true objective to first order. This approximation requires the policy mapping to be smooth. Top-K routing violates this—an infinitesimal parameter change can cause a discrete expert switch, making the surrogate jump discontinuously and invalidating the gradient-based optimization entirely.

| Pathology | Dense LLMs | MoE LLMs with Top-K |
|-----------|---------------|-----------|
| Gradient flow | Smooth, non-zero almost everywhere | Zero almost everywhere for unselected experts' logits |
| Token distribution mapping | Continuous and differentiable | Discontinuous at routing boundaries |
| First-order approximation | Valid: {{< math >}}$\nabla L_\mu \approx \nabla J${{< /math >}} | Invalid at routing boundaries |

---

## Part 1: The Gradient Blackout

### Setup: Autoregressive LLM with MoE

Consider an autoregressive language model generating a response {{< math >}}$y = (y_1, y_2, \ldots, y_T)${{< /math >}} given a prompt {{< math >}}$x${{< /math >}}. At each timestep {{< math >}}$t${{< /math >}}, the model predicts the next token {{< math >}}$y_t${{< /math >}} given the context:

- **State:** {{< math >}}$s_t = (x, y_{\lt t})${{< /math >}} — the prompt concatenated with previously generated tokens
- **Action:** {{< math >}}$a_t = y_t${{< /math >}} — the next token to generate
- **Policy:** {{< math >}}$\pi_\theta(a_t | s_t) = \pi_\theta(y_t | x, y_{\lt t})${{< /math >}} — the token probability distribution

In an MoE transformer, each MoE layer has a router that computes logits {{< math >}}$h \in \mathbb{R}^N${{< /math >}} for {{< math >}}$N${{< /math >}} experts based on the hidden state. For a fixed {{< math >}}$K \lt N${{< /math >}}, the Top-K operator selects the indices of the {{< math >}}$K${{< /math >}} largest logits:

{{< math >}}
$$\mathcal{K}(h) = \{j : h_j \text{ is among the } K \text{ largest elements of } h\}$$
{{< /math >}}

The MoE layer output is:

{{< math >}}
$$\text{MoE}(z) = \sum_{j \in \mathcal{K}(h)} \frac{e^{h_j}}{\sum_{k \in \mathcal{K}(h)} e^{h_k}} E_j(z)$$
{{< /math >}}

where {{< math >}}$z${{< /math >}} is the hidden state, {{< math >}}$E_j${{< /math >}} is expert {{< math >}}$j${{< /math >}}'s FFN, and {{< math >}}$h = h(z; \theta_r)${{< /math >}} depends on router parameters {{< math >}}$\theta_r${{< /math >}}. The final token distribution {{< math >}}$\pi_\theta(y_t | x, y_{\lt t})${{< /math >}} depends on outputs from all MoE layers.

### The Zero Gradient Problem

When training with RL, we optimize the policy {{< math >}}$\pi_\theta(y_t | x, y_{\lt t})${{< /math >}} to maximize reward. Consider the gradient with respect to an **unselected** expert's logit {{< math >}}$h_u${{< /math >}}, where {{< math >}}$u \notin \mathcal{K}(h)${{< /math >}}.

**Step 1: Locally Constant Set.** Let {{< math >}}$h_{(K)}${{< /math >}} denote the {{< math >}}$K${{< /math >}}-th largest element of {{< math >}}$h${{< /math >}}, and let {{< math >}}$e_u${{< /math >}} be the {{< math >}}$u${{< /math >}}-th standard basis vector. Assuming no ties (which holds almost everywhere), since {{< math >}}$u \notin \mathcal{K}(h)${{< /math >}}, we have {{< math >}}$h_u \lt h_{(K)}${{< /math >}}. For any scalar perturbation {{< math >}}$\epsilon${{< /math >}} with {{< math >}}$h_u + \epsilon \lt h_{(K)}${{< /math >}}:

{{< math >}}
$$\mathcal{K}(h + \epsilon \cdot e_u) = \mathcal{K}(h)$$
{{< /math >}}

The set of selected experts remains unchanged as long as {{< math >}}$h_u${{< /math >}} stays below the selection threshold.

**Step 2: Zero Dependency.** Since {{< math >}}$u \notin \mathcal{K}(h)${{< /math >}}:
- Expert {{< math >}}$E_u${{< /math >}}'s output does not contribute to the hidden state
- The logit {{< math >}}$h_u${{< /math >}} does not appear in the softmax normalization

**Result:** The gradient of the token probability with respect to unselected expert logits is zero:

{{< math >}}
$$\frac{\partial \pi_\theta(y_t | x, y_{\lt t})}{\partial h_u} = 0 \quad \text{almost everywhere}$$
{{< /math >}}

### Why Subgradients Don't Help

Normally, we handle non-smooth points (like ReLU at 0) using subgradients. However, there's a crucial distinction:

**Non-smooth but continuous (e.g., ReLU):**
- The function {{< math >}}$f(x) = \max(0, x)${{< /math >}} is continuous everywhere
- At {{< math >}}$x=0${{< /math >}}, the subgradient {{< math >}}$\partial f(0) = [0, 1]${{< /math >}} provides valid descent directions
- Optimization can proceed by choosing any element of the subdifferential

**Discontinuous (Top-K):**
- The selection function is **discontinuous** at decision boundaries
- **On the plateau:** The gradient is exactly {{< math >}}$\mathbf{0}${{< /math >}}—no signal at all
- **At the cliff:** Where {{< math >}}$h_i = h_j${{< /math >}} for the {{< math >}}$K${{< /math >}}-th and {{< math >}}$(K+1)${{< /math >}}-th ranked experts, the output jumps discontinuously as they swap positions

At a discontinuity, the classical subgradient is not defined. The *Clarke Generalized Gradient* can be defined for locally Lipschitz functions, but the MoE layer output {{< math >}}$\text{MoE}(z)${{< /math >}} is not locally Lipschitz at switching boundaries—it has jump discontinuities.

**Key insight:** The pathology is not that gradients are "undefined" at boundaries, but rather:
1. Away from boundaries: {{< math >}}$\frac{\partial \pi_\theta(y_t | x, y_{\lt t})}{\partial h_u} = 0${{< /math >}} exactly (no signal)
2. At boundaries: the function jumps discontinuously, so no first-order approximation is valid

**Bottom line:** During LLM-RL training, the router receives no gradient signal about whether switching to a different expert would generate better responses. The model cannot learn to route tokens to more suitable experts based on reward feedback.

---

## Part 2: The First-Order Approximation Failure

### The Trust Region Principle and Its Practical Approximations

The theoretical foundation of modern LLM-RL comes from **Trust Region Policy Optimization (TRPO)** ([Schulman et al., 2015](https://arxiv.org/abs/1502.05477)). However, practical algorithms like PPO and GRPO do not implement actual trust region optimization—they use **clipping mechanisms** to *mimic* the trust region principle. Understanding this distinction is crucial.

In the LLM setting, we use the autoregressive MDP formulation:
- **State:** {{< math >}}$s_t = (x, y_{\lt t})${{< /math >}} — prompt plus tokens generated so far
- **Action:** {{< math >}}$a_t = y_t${{< /math >}} — the next token
- **Policy:** {{< math >}}$\pi_\theta(y_t | x, y_{\lt t})${{< /math >}} — the LLM's token distribution

### The Surrogate Objective and Why It Works

The key insight of TRPO is optimizing a **surrogate objective** {{< math >}}$L_\mu(\pi)${{< /math >}} instead of the true objective {{< math >}}$J(\pi)${{< /math >}} directly:

{{< math >}}
$$L_{\mu}(\pi) = J(\mu) + \mathbb{E}_{s \sim d_\mu} \mathbb{E}_{a \sim \pi(\cdot|s)} [A_\mu(s, a)]$$
{{< /math >}}

where {{< math >}}$d_\mu${{< /math >}} is the state visitation distribution under the sampling policy {{< math >}}$\mu${{< /math >}}, and {{< math >}}$A_\mu${{< /math >}} is the advantage function.

This surrogate is useful because it satisfies two critical conditions **at** {{< math >}}$\pi = \mu${{< /math >}}:

1. **Equal values:** {{< math >}}$L_\mu(\mu) = J(\mu)${{< /math >}}
2. **Equal gradients:** {{< math >}}$\nabla_\theta L_\mu|_{\pi_\theta=\mu} = \nabla_\theta J|_{\pi_\theta=\mu}${{< /math >}}

The surrogate is a **first-order Taylor approximation** of the true objective—it matches both value and gradient at the point of tangency. Away from {{< math >}}$\pi = \mu${{< /math >}}, the approximation degrades.

### The TRPO Lower Bound

TRPO quantifies exactly how much the approximation degrades. The original theorem ([Schulman et al., 2015](https://arxiv.org/abs/1502.05477)) gives:

{{< math >}}
$$J(\pi) \geq L_{\mu}(\pi) - \frac{4\epsilon\gamma}{(1-\gamma)^2} \cdot (D_{TV}^{\max})^2$$
{{< /math >}}

where {{< math >}}$\epsilon = \max_{s,a}|A(s,a)|${{< /math >}} and {{< math >}}$D_{TV}^{\max} = \max_s D_{TV}(\pi(\cdot|s) \| \mu(\cdot|s))${{< /math >}}. For finite-horizon undiscounted settings ({{< math >}}$\gamma = 1${{< /math >}}, horizon {{< math >}}$T${{< /math >}}), the bound becomes:

{{< math >}}
$$J(\pi) \geq L_{\mu}(\pi) - C \cdot T^2 \cdot (D_{TV}^{\max})^2$$
{{< /math >}}

The penalty scales **quadratically with both horizon and TV distance** because state distribution mismatch accumulates over time.

### The Gap Between Theory and Practice

Here's the critical point: **PPO/GRPO do not implement this bound**. They use a constant clipping factor (e.g., {{< math >}}$\epsilon = 0.2${{< /math >}}) regardless of sequence length, while the theory requires the trust region to **shrink as** {{< math >}}$O(1/T^2)${{< /math >}}.

In practice, PPO/GRPO are best understood as **stochastic gradient ascent (SGA)** methods that compute a clipped gradient estimator. [Li et al., 2025](https://richardli.xyz/rl-collapse-1) analyze how mismatch between sampling policy {{< math >}}$\mu${{< /math >}} and target policy {{< math >}}$\pi${{< /math >}} affects optimization. Crucially, the token-level importance sampling (IS) gradient used in PPO/GRPO is exactly the gradient of the surrogate objective {{< math >}}$\nabla_\theta L_\mu${{< /math >}}, not the true gradient {{< math >}}$\nabla_\theta J${{< /math >}}:

{{< math >}}
$$\underbrace{\mathbb{E}_{s_t \sim d_\mu} \mathbb{E}_{y_t \sim \mu} \left[ \frac{\pi_\theta(y_t|s_t)}{\mu(y_t|s_t)} A_\mu(s_t, y_t) \nabla_\theta \log \pi_\theta(y_t|s_t) \right]}_{\text{Token-level IS gradient (what PPO computes)}} = \nabla_\theta L_\mu$$
{{< /math >}}

{{< math >}}
$$\nabla_\theta L_\mu \neq \nabla_\theta J = \underbrace{\mathbb{E}_{s_t \sim d_\pi} \mathbb{E}_{y_t \sim \pi} \left[ A_\pi(s_t, y_t) \nabla_\theta \log \pi_\theta(y_t|s_t) \right]}_{\text{True policy gradient}}$$
{{< /math >}}

where {{< math >}}$s_t = (x, y_{\lt t})${{< /math >}} is the state (prompt + generated prefix) and {{< math >}}$y_t${{< /math >}} is the action (next token). The token-level IS ratio {{< math >}}$\pi_\theta(y_t|s_t)/\mu(y_t|s_t)${{< /math >}} corrects for the **token distribution mismatch**, but the expectation over states is still taken under {{< math >}}$d_\mu${{< /math >}}, not {{< math >}}$d_\pi${{< /math >}}. This **prefix distribution mismatch** is the source of bias, which scales with both horizon and policy divergence.

This bias is **tolerable** when:
- The off-policiness is solely induced by policy parameter updates
- The mismatch {{< math >}}$D_{TV}^{\max}${{< /math >}} is small and controlled (e.g., by the clipping mechanism)

This bias becomes **intolerable** when:
- The mismatch has diverse, uncontrolled sources (e.g., expert shifts in MoE)
- {{< math >}}$D_{TV}^{\max}${{< /math >}} is large, amplifying the approximation error

Their success relies on:
1. The **first-order approximation** {{< math >}}$\nabla_\theta L_\mu \approx \nabla_\theta J${{< /math >}} being valid
2. The policy remaining **close enough** to {{< math >}}$\mu${{< /math >}} that the surrogate is meaningful
3. The mapping from parameters to policy being **smooth**

### How Top-K Breaks the First-Order Approximation

Let {{< math >}}$f: \Theta \to \Pi${{< /math >}} be the map from parameters {{< math >}}$\theta \in \Theta${{< /math >}} to the token distribution {{< math >}}$\pi_\theta(y_t | x, y_{\lt t}) \in \Pi${{< /math >}}.

**In dense LLMs (GPT, LLaMA, etc.):** {{< math >}}$f${{< /math >}} is smooth. The surrogate {{< math >}}$L_\mu(\pi)${{< /math >}} is a valid first-order approximation of {{< math >}}$J(\pi)${{< /math >}}, and gradient-based optimization works as expected.

**In MoE LLMs (Mixtral, DeepSeek-MoE, etc.):** {{< math >}}$f${{< /math >}} is **piecewise smooth but globally discontinuous**—smooth within each routing region, but with jump discontinuities at region boundaries.

At a switching point {{< math >}}$\theta^*${{< /math >}} (where expert rankings swap for some token), consider a direction {{< math >}}$v${{< /math >}} crossing the decision boundary:

{{< math >}}
$$\lim_{t \to 0^+} \pi_{\theta^* + tv}(y_t | x, y_{\lt t}) \neq \lim_{t \to 0^+} \pi_{\theta^* - tv}(y_t | x, y_{\lt t})$$
{{< /math >}}

**The first-order approximation completely fails at these boundaries:**
- At the discontinuity, the gradient {{< math >}}$\nabla_\theta J${{< /math >}} does not exist in the classical sense
- The surrogate {{< math >}}$L_\mu(\pi)${{< /math >}} cannot provide a valid first-order approximation to a discontinuous {{< math >}}$J(\pi)${{< /math >}}
- The clipping mechanism in PPO/GRPO cannot help—it assumes the underlying policy mapping is smooth

### The Consequences for LLM-RL Training

When the router crosses a decision boundary during training:

**1. The Surrogate Becomes Meaningless:**
PPO/GRPO optimize {{< math >}}$L_\mu(\pi)${{< /math >}} as a proxy for {{< math >}}$J(\pi)${{< /math >}}. At a discontinuity, the surrogate jumps while the gradient estimator sees only the local (pre-jump) landscape. The optimizer is effectively blind to what happens after crossing.

**2. Gradient Estimates Are Invalid:**
The clipped gradient estimator assumes {{< math >}}$\nabla_\theta L_\mu \approx \nabla_\theta J${{< /math >}}. At a discontinuity, neither gradient exists in the classical sense, and the computed "gradient" points in an arbitrary direction.

**3. Large, Uncontrolled Approximation Error:**
When the router switches experts, the effective {{< math >}}$D_{TV}^{\max}${{< /math >}} (per-token TV distance) can be large—the output distribution changes discretely, not continuously. The TRPO bound shows the surrogate-to-objective gap scales as {{< math >}}$O(T^2 \cdot (D_{TV}^{\max})^2)${{< /math >}}. When {{< math >}}$D_{TV}^{\max}${{< /math >}} jumps due to expert switching, this creates a regime where the gradient estimator is systematically wrong, pushing optimization toward incorrect solutions. This may contribute to the training instability observed when training MoE LLMs with RL.

---

## Part 3: Implications for MoE LLM-RL

### Why LLM-RL with MoE is Hard

The combination of these two pathologies creates a perfect storm for RL training:

1. **Exploration is blind:** The router receives no gradient signal for unselected experts. When generating response {{< math >}}$y${{< /math >}} to prompt {{< math >}}$x${{< /math >}}, the model cannot learn whether routing tokens to different experts would produce higher-reward responses.

2. **Exploitation is unstable:** When the optimizer does find a beneficial switch point, crossing it can cause instability due to the first-order approximation failure. This may manifest as reward spikes followed by degradation during RL training.

3. **The optimization landscape is adversarial:** Flat plateaus (zero gradient) punctuated by cliffs (discontinuities) with no smooth paths between expert configurations. The model gets stuck in suboptimal routing patterns.

### Potential Solutions for MoE LLM-RL

Understanding these pathologies suggests directions for solutions:

**For the gradient blackout:**
- Soft routing (e.g., softmax over all experts) restores gradient flow but sacrifices sparsity and inference speed
- Auxiliary losses that provide signal to unselected experts (e.g., load balancing with gradient flow)
- Exploration bonuses for trying different expert combinations during rollouts

**For the first-order approximation failure:**
- Entropy regularization on the router to smooth the routing distribution
- Annealing from soft to hard routing during RL training
- Modified KL constraints that account for discrete expert switches
- Freezing the router during RL (sacrificing routing adaptation)

---

## Summary

The instability of RL training for MoE LLMs is not a bug to be fixed with hyperparameter tuning—it's a fundamental consequence of the Top-K operator's mathematical properties:

| Property | Effect on LLM-RL Optimization |
|----------|----------------------|
| Discrete expert selection | Zero gradient for unselected experts—no signal for improving routing |
| Jump discontinuities at boundaries | Large {{< math >}}$D_{TV}^{\max}${{< /math >}} when experts switch, causing {{< math >}}$O(T^2 \cdot (D_{TV}^{\max})^2)${{< /math >}} approximation error |
| First-order approximation failure | Surrogate {{< math >}}$L_\mu${{< /math >}} invalid at discontinuities—gradient estimates systematically wrong |
| No gradient signal for switching | Cannot learn which expert would generate better tokens |

Until routing mechanisms are developed that preserve gradient information while maintaining sparsity, training MoE LLMs with RL will remain fundamentally more challenging than training dense LLMs.

---

## References

**Mixture of Experts:**
- Shazeer, N., et al. (2017). "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer." *ICLR*.
- Fedus, W., Zoph, B., & Shazeer, N. (2022). "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity." *JMLR*.

**Trust Region Methods:**
- Schulman, J., et al. (2015). "Trust Region Policy Optimization." *ICML*.
- Schulman, J., et al. (2017). "Proximal Policy Optimization Algorithms." *arXiv*.

**LLM-RL Analysis:**
- Liu, J., Li, Y., Fu, Y., Wang, J., Liu, Q., & Shen, Y. (2025). "[When Speed Kills Stability: Demystifying RL Collapse from the Training-Inference Mismatch](https://richardli.xyz/rl-collapse)." *Blog Series*.

**Non-smooth Optimization:**
- Clarke, F. H. (1990). *Optimization and Nonsmooth Analysis.* SIAM.

---

*Last updated: December 7, 2025*
