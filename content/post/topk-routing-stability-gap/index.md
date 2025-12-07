---
title: 'The Stability Gap: Why Top-K Routing Breaks RL Optimization'
subtitle: 'How Discrete Expert Selection Creates Pathological Optimization Landscapes'
summary: 'A rigorous mathematical analysis showing that Top-K expert routing in Mixture of Experts creates two fundamental pathologies: gradient blackout (zero gradients almost everywhere) and trust region violation (discontinuous policy changes), explaining the notorious instability of MoE-RL training.'
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

Training Mixture of Experts (MoE) language models with Reinforcement Learning is notoriously unstable. While dense LLMs enjoy smooth optimization landscapes, MoE-based models like Mixtral, DeepSeek-MoE, and Qwen-MoE introduce the **Top-K operator**—a discrete switching mechanism that creates discontinuities in the optimization landscape.

This discreteness introduces two fundamental mathematical pathologies that break standard RL assumptions used in PPO, GRPO, and other LLM-RL algorithms.

---

## TL;DR: The Two Pathologies

**Challenge 1: Gradient Blackout.** The gradient of the token distribution {{< math >}}$\pi_\theta(y_t | x, y_{<t})${{< /math >}} with respect to unselected experts' logits is exactly zero almost everywhere. Unlike non-smooth convex functions where subgradients guide optimization, the Top-K landscape offers no directional information on how to switch to a better expert.

**Challenge 2: Trust Region Violation.** Modern LLM-RL algorithms (PPO, GRPO) optimize a surrogate objective under a KL divergence constraint. This requires differentiability for gradient computation and continuity for constraint satisfaction. Top-K routing violates both—an infinitesimal parameter change can cause a discrete expert switch, making both the surrogate and KL divergence jump discontinuously.

| Pathology | Dense LLMs | MoE LLMs with Top-K |
|-----------|---------------|-----------|
| Gradient flow | Smooth, non-zero almost everywhere | Zero almost everywhere for unselected experts' logits |
| Token distribution mapping | Continuous and differentiable | Discontinuous at routing boundaries |
| Trust region validity | Surrogate optimization works | Surrogate jumps at switch points |

---

## Part 1: The Gradient Blackout

### Setup: Autoregressive LLM with MoE

Consider an autoregressive language model generating a response {{< math >}}$y = (y_1, y_2, \ldots, y_T)${{< /math >}} given a prompt {{< math >}}$x${{< /math >}}. At each timestep {{< math >}}$t${{< /math >}}, the model predicts the next token {{< math >}}$y_t${{< /math >}} given the context:

- **State:** {{< math >}}$s_t = (x, y_{<t})${{< /math >}} — the prompt concatenated with previously generated tokens
- **Action:** {{< math >}}$a_t = y_t${{< /math >}} — the next token to generate
- **Policy:** {{< math >}}$\pi_\theta(a_t | s_t) = \pi_\theta(y_t | x, y_{<t})${{< /math >}} — the token probability distribution

In an MoE transformer, each MoE layer has a router that computes logits {{< math >}}$h \in \mathbb{R}^N${{< /math >}} for {{< math >}}$N${{< /math >}} experts based on the hidden state. For a fixed {{< math >}}$K < N${{< /math >}}, the Top-K operator selects the indices of the {{< math >}}$K${{< /math >}} largest logits:

{{< math >}}
$$\mathcal{K}(h) = \{j : h_j \text{ is among the } K \text{ largest elements of } h\}$$
{{< /math >}}

The MoE layer output is:

{{< math >}}
$$\text{MoE}(z) = \sum_{j \in \mathcal{K}(h)} \frac{e^{h_j}}{\sum_{k \in \mathcal{K}(h)} e^{h_k}} E_j(z)$$
{{< /math >}}

where {{< math >}}$z${{< /math >}} is the hidden state, {{< math >}}$E_j${{< /math >}} is expert {{< math >}}$j${{< /math >}}'s FFN, and {{< math >}}$h = h(z; \theta_r)${{< /math >}} depends on router parameters {{< math >}}$\theta_r${{< /math >}}. The final token distribution {{< math >}}$\pi_\theta(y_t | x, y_{<t})${{< /math >}} depends on outputs from all MoE layers.

### The Zero Gradient Problem

When training with RL, we optimize the policy {{< math >}}$\pi_\theta(y_t | x, y_{<t})${{< /math >}} to maximize reward. Consider the gradient with respect to an **unselected** expert's logit {{< math >}}$h_u${{< /math >}}, where {{< math >}}$u \notin \mathcal{K}(h)${{< /math >}}.

**Step 1: Locally Constant Set.** Let {{< math >}}$h_{(K)}${{< /math >}} denote the {{< math >}}$K${{< /math >}}-th largest element of {{< math >}}$h${{< /math >}}, and let {{< math >}}$e_u${{< /math >}} be the {{< math >}}$u${{< /math >}}-th standard basis vector. Assuming no ties (which holds almost everywhere), since {{< math >}}$u \notin \mathcal{K}(h)${{< /math >}}, we have {{< math >}}$h_u < h_{(K)}${{< /math >}}. For any scalar perturbation {{< math >}}$\epsilon${{< /math >}} with {{< math >}}$h_u + \epsilon < h_{(K)}${{< /math >}}:

{{< math >}}
$$\mathcal{K}(h + \epsilon \cdot e_u) = \mathcal{K}(h)$$
{{< /math >}}

The set of selected experts remains unchanged as long as {{< math >}}$h_u${{< /math >}} stays below the selection threshold.

**Step 2: Zero Dependency.** Since {{< math >}}$u \notin \mathcal{K}(h)${{< /math >}}:
- Expert {{< math >}}$E_u${{< /math >}}'s output does not contribute to the hidden state
- The logit {{< math >}}$h_u${{< /math >}} does not appear in the softmax normalization

**Result:** The gradient of the token probability with respect to unselected expert logits is zero:

{{< math >}}
$$\frac{\partial \pi_\theta(y_t | x, y_{<t})}{\partial h_u} = 0 \quad \text{almost everywhere}$$
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
1. Away from boundaries: {{< math >}}$\frac{\partial \pi_\theta(y_t | x, y_{<t})}{\partial h_u} = 0${{< /math >}} exactly (no signal)
2. At boundaries: the function jumps discontinuously, so no first-order approximation is valid

**Bottom line:** During LLM-RL training, the router receives no gradient signal about whether switching to a different expert would generate better responses. The model cannot learn to route tokens to more suitable experts based on reward feedback.

---

## Part 2: The Trust Region Violation

### The Trust Region Framework for LLM-RL

Modern LLM-RL algorithms like PPO and GRPO are built on the **Trust Region** principle from [Schulman et al. (2015)](https://arxiv.org/abs/1502.05477). The key idea is to derive a **lower bound** on the true objective that can be optimized safely.

In the LLM setting, we use the autoregressive MDP formulation:
- **State:** {{< math >}}$s_t = (x, y_{<t})${{< /math >}} — prompt plus tokens generated so far
- **Action:** {{< math >}}$a_t = y_t${{< /math >}} — the next token
- **Policy:** {{< math >}}$\pi_\theta(y_t | x, y_{<t})${{< /math >}} — the LLM's token distribution

**The True Objective:** The expected reward under policy {{< math >}}$\tilde{\pi}${{< /math >}}:

{{< math >}}
$$\eta(\tilde{\pi}) = \eta(\pi) + \mathbb{E}_{(x,y) \sim \tilde{\pi}}\left[\sum_{t=1}^{T} \gamma^{t-1} A^{\pi}(s_t, y_t)\right]$$
{{< /math >}}

where {{< math >}}$A^\pi(s_t, y_t)${{< /math >}} is the advantage of generating token {{< math >}}$y_t${{< /math >}} in context {{< math >}}$s_t = (x, y_{<t})${{< /math >}}.

**The Surrogate Objective:** Replace the state distribution under {{< math >}}$\tilde{\pi}${{< /math >}} with that under {{< math >}}$\pi${{< /math >}}:

{{< math >}}
$$L_{\pi}(\tilde{\pi}) = \eta(\pi) + \sum_{s_t} \rho_{\pi}(s_t) \sum_{y_t} \tilde{\pi}(y_t|s_t) A^{\pi}(s_t, y_t)$$
{{< /math >}}

This surrogate matches the true objective to **first order**: {{< math >}}$\nabla_\theta L_{\pi}|_{\theta_{old}} = \nabla_\theta \eta|_{\theta_{old}}${{< /math >}}.

**The Monotonic Improvement Theorem:** The surrogate provides a lower bound:

{{< math >}}
$$\eta(\tilde{\pi}) \geq L_{\pi}(\tilde{\pi}) - \frac{4\epsilon\gamma}{(1-\gamma)^2} \cdot D_{KL}^{max}(\pi, \tilde{\pi})$$
{{< /math >}}

where {{< math >}}$\epsilon = \max_{s_t, y_t}|A^{\pi}(s_t, y_t)|${{< /math >}} is the maximum absolute advantage. This leads to the PPO/GRPO optimization:

{{< math >}}
$$\max_{\theta} L_{\pi_{\theta_{old}}}(\pi_\theta) \quad \text{subject to} \quad D_{KL}(\pi_{\theta_{old}} \| \pi_\theta) \le \delta$$
{{< /math >}}

**Key property:** The bound holds for any two policies, but **practical optimization** relies on:
1. {{< math >}}$L_{\pi}(\tilde{\pi})${{< /math >}} is **differentiable** in policy parameters (to compute gradients)
2. {{< math >}}$D_{KL}${{< /math >}} is **continuous** so the constraint can be satisfied via line search
3. The first-order approximation {{< math >}}$\nabla_\theta L_{\pi} = \nabla_\theta \eta${{< /math >}} guides optimization toward improvement

### How Top-K Violates This in MoE LLMs

Let {{< math >}}$f: \Theta \to \Pi${{< /math >}} be the map from parameters {{< math >}}$\theta \in \Theta${{< /math >}} to the token distribution {{< math >}}$\pi_\theta(y_t | x, y_{<t}) \in \Pi${{< /math >}}.

**In dense LLMs (GPT, LLaMA, etc.):** {{< math >}}$f${{< /math >}} is continuous—both {{< math >}}$L_{\pi_{old}}(\pi_\theta)${{< /math >}} and {{< math >}}$D_{KL}(\pi_{old} \| \pi_\theta)${{< /math >}} vary smoothly with {{< math >}}$\theta${{< /math >}}, and the first-order approximation {{< math >}}$\nabla_\theta L_{\pi} = \nabla_\theta \eta${{< /math >}} is valid.

**In MoE LLMs (Mixtral, DeepSeek-MoE, etc.):** {{< math >}}$f${{< /math >}} is **piecewise smooth but globally discontinuous**—smooth within each routing region, but with jump discontinuities at region boundaries.

At a switching point {{< math >}}$\theta^*${{< /math >}} (where expert rankings swap for some token), consider a direction {{< math >}}$v${{< /math >}} crossing the decision boundary. Approaching from opposite sides yields different token distributions:

{{< math >}}
$$\lim_{t \to 0^+} \pi_{\theta^* + tv}(y_t | x, y_{<t}) \neq \lim_{t \to 0^+} \pi_{\theta^* - tv}(y_t | x, y_{<t})$$
{{< /math >}}

This discontinuity **breaks the optimization**:
- The surrogate {{< math >}}$L_{\pi}(\tilde{\pi})${{< /math >}} jumps discontinuously—gradients don't exist at the boundary
- The KL divergence {{< math >}}$D_{KL}(\pi_{old} \| \pi_\theta)${{< /math >}} also jumps—line search cannot satisfy the constraint smoothly
- The lower bound {{< math >}}$\eta \geq L_{\pi} - C \cdot D_{KL}^{max}${{< /math >}} still holds, but provides no optimization guidance across the discontinuity

### The Consequences for LLM-RL Training

When the router crosses a decision boundary during training:

**1. Gradient-Based Optimization Fails:**
PPO/GRPO compute {{< math >}}$\nabla_\theta L_{\pi}${{< /math >}} to find an improving direction. At a discontinuity, this gradient doesn't exist—there's no local information about how to improve across the boundary.

**2. KL Constraint Cannot Be Satisfied Smoothly:**
For {{< math >}}$\theta${{< /math >}} approaching a boundary, even an infinitesimal step crossing it causes:

{{< math >}}
$$\lim_{\epsilon \to 0^+} D_{KL}(\pi_{\theta}(\cdot | x, y_{<t}) \| \pi_{\theta + \epsilon v}(\cdot | x, y_{<t})) > 0$$
{{< /math >}}

The KL divergence has a positive lower bound at the discontinuity. The line search, which tries to find the largest step satisfying {{< math >}}$D_{KL} \le \delta${{< /math >}}, cannot smoothly approach the boundary.

**3. Chattering Behavior:**
The optimizer oscillates around switching points. It takes a step, crosses the boundary, both {{< math >}}$L_{\pi}${{< /math >}} and {{< math >}}$D_{KL}${{< /math >}} jump, the step is rejected or reversed, it crosses back, and so on. This is the training instability frequently observed when training MoE LLMs with RL.

---

## Part 3: Implications for MoE LLM-RL

### Why LLM-RL with MoE is Hard

The combination of these two pathologies creates a perfect storm for RL training:

1. **Exploration is blind:** The router receives no gradient signal for unselected experts. When generating response {{< math >}}$y${{< /math >}} to prompt {{< math >}}$x${{< /math >}}, the model cannot learn whether routing tokens to different experts would produce higher-reward responses.

2. **Exploitation is unstable:** When the optimizer does find a beneficial switch point, crossing it causes violent instability due to trust region violation. This manifests as reward spikes followed by crashes during RL training.

3. **The optimization landscape is adversarial:** Flat plateaus (zero gradient) punctuated by cliffs (discontinuities) with no smooth paths between expert configurations. The model gets stuck in suboptimal routing patterns.

### Potential Solutions for MoE LLM-RL

Understanding these pathologies suggests directions for solutions:

**For the gradient blackout:**
- Soft routing (e.g., softmax over all experts) restores gradient flow but sacrifices sparsity and inference speed
- Auxiliary losses that provide signal to unselected experts (e.g., load balancing with gradient flow)
- Exploration bonuses for trying different expert combinations during rollouts

**For trust region violation:**
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
| Jump discontinuities at boundaries | Surrogate objective {{< math >}}$L_\pi${{< /math >}} jumps, PPO/GRPO optimization fails |
| Non-differentiable token distribution | Gradient-based KL constraint satisfaction breaks |
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

**Non-smooth Optimization:**
- Clarke, F. H. (1990). *Optimization and Nonsmooth Analysis.* SIAM.

---

*Last updated: December 7, 2025*
