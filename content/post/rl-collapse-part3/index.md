---
title: "Part 3: Trust Region Optimization via Sequence Masking"
date: 2025-11-04
math: true
authors:
  - admin
  - jiacai-liu
tags:
  - Reinforcement Learning
  - Off-Policy
  - Trust Region
  - Chain-of-Thought
categories:
  - Research
series:
  - RL Collapse
series_order: 3
---

**Authors:** [Yingru Li](http://richardli.xyz), [Jiacai Liu](https://www.jiacailiu.cn/)

{{% callout note %}}
**Original Blog:** [When Speed Kills Stability: Demystifying RL Collapse from the Training-Inference Mismatch](https://richardli.xyz/rl-collapse)
{{% /callout %}}

{{% callout note %}}
**Series Context**

- [**Part 1**](https://richardli.xyz/rl-collapse-1): We established the SGA (Stochastic Gradient Ascent) framework and identified two failure modes of off-policy mismatch: Bias (measured by $D_{TV}$) and Variance (measured by $\chi^2$-divergence).
- [**Part 2**](https://richardli.xyz/rl-collapse-2): We analyzed gradient estimators and showed that Token-level IS (PPO/GRPO) has $O(T^2 \Delta_{\max})$ bias, while Sequence-Level Truncated IS (Seq-TIS) achieves a controllable bias-variance trade-off via clipping: $\rho(y) \to \min(\rho(y), C)$.
{{% /callout %}}

{{% callout note %}}
**TL;DR**

In a standard statistical setting, Part 2 solved the problem with Seq-TIS.

However, when training Agents or Reasoning Models (Chain-of-Thought), two practical phenomena violate the assumptions underlying Seq-TIS:

1. **Out-of-Distribution (OOD) High-Weight Samples:** Extremely high importance weights ($\rho \gg C$) often correspond to samples outside the behavior policy's support—numerical errors or distribution shift artifacts. Clipping these samples still includes them in the gradient update. **Solution: Enforce a Hard Trust Region via Rejection/Masking (Seq-MIS).**
2. **Length-Dependent Rejection Bias:** The importance ratio $\rho(y) = \prod_t \rho_t$ grows exponentially with sequence length $T$, causing systematic rejection of long sequences regardless of per-step quality. **Solution: Geometric Rejection Sampling (Geo-RS), which enforces a Per-Token Trust Region using a length-normalized KL divergence criterion.**
{{% /callout %}}

---

## Citation

```bibtex
@online{liu-li-2025-rl-collapse,
  title = {When Speed Kills Stability: Demystifying {RL} Collapse from the Training-Inference Mismatch},
  author = {Liu, Jiacai and Li, Yingru and Fu, Yuqian and Wang, Jiawei and Liu, Qian and Shen, Yu},
  year = {2025},
  month = sep,
  url = {https://richardli.xyz/rl-collapse}
}
```

---

## Recap: The Trust Region Framework

In Part 1, we established the theoretical foundation for trust region optimization. Here we recap the key results that motivate the methods in this part.

### The Surrogate Objective and Its Limitations

When optimizing a policy $\pi$ using samples from a behavior policy $\mu$, we cannot directly optimize the true objective $J(\pi)$. Instead, we optimize a **surrogate objective**:

{{< math >}}
$$
L_\mu(\pi) = J(\mu) + \mathbb{E}_{s \sim d_\mu} \mathbb{E}_{a \sim \pi(\cdot|s)} [A_\mu(s, a)]
$$
{{< /math >}}

where $d_\mu$ is the state visitation distribution under $\mu$, and $A_\mu$ is the advantage function.

The surrogate is a first-order approximation that satisfies:
- $L_\mu(\mu) = J(\mu)$ (equal values at $\pi = \mu$)
- $\nabla L_\mu(\pi)|_{\pi=\mu} = \nabla J(\pi)|_{\pi=\mu}$ (equal gradients at $\pi = \mu$)

However, the approximation degrades as $\pi$ moves away from $\mu$.

### The TRPO Lower Bound

The **Performance Difference Lemma** quantifies the gap between the surrogate and true objectives:

{{< math >}}
$$
J(\pi) - J(\mu) = \mathbb{E}_{s \sim d_{\pi}} \mathbb{E}_{a \sim \pi(\cdot|s)} [A_\mu(s, a)]
$$
{{< /math >}}

The key difference from the surrogate is that the true improvement uses the state distribution $d_\pi$ (under the new policy), while the surrogate uses $d_\mu$ (under the old policy).

Using the **Simulation Lemma**, which bounds how state distributions diverge over time:

{{< math >}}
$$
D_{TV}(d_{\pi,t} \| d_{\mu,t}) \leq t \cdot D_{TV}^{\max}(\pi, \mu)
$$
{{< /math >}}

we derive the **TRPO lower bound**:

{{< math >}}
$$
\boxed{J(\pi) \geq L_\mu(\pi) - C \cdot T^2 \cdot D_{TV}^{\max}(\pi, \mu)}
$$
{{< /math >}}

where:
- $C$ is a constant depending on the maximum advantage $\max_{s,a}|A_\mu(s,a)|$
- $T$ is the horizon (sequence length)
- $D_{TV}^{\max} = \max_s D_{TV}(\pi(\cdot|s) \| \mu(\cdot|s))$ is the maximum per-token TV distance

### The Trust Region Requirement

For surrogate optimization to guarantee improvement in the true objective, the policy must stay within a **trust region**:

{{< math >}}
$$
D_{TV}^{\max}(\pi, \mu) \leq \delta
$$
{{< /math >}}

The critical insight is that **the valid trust region size must shrink with sequence length**:

{{< math >}}
$$
\delta \propto \frac{1}{T^2}
$$
{{< /math >}}

This $T^2$ dependence arises because state distribution errors accumulate linearly over $T$ steps, and the total error (summed over all steps) scales quadratically.

### Soft vs. Hard Trust Regions

The TRPO framework suggests two approaches to enforce trust regions:

| Type | Mechanism | Implementation |
|------|-----------|----------------|
| **Soft (Clipping)** | Down-weight samples outside the region | $\min(\rho, C)$ — sample included with bounded weight |
| **Hard (Rejection)** | Exclude samples outside the region | $\mathbb{I}(\rho \leq C)$ — sample excluded entirely |

**Soft trust regions** use clipped importance sampling: $\min(\rho_t, 1+\epsilon)$. This is computationally efficient but retains potentially problematic samples.

**This part develops hard trust region methods**—Seq-MIS and Geo-RS—that completely exclude samples outside the trusted region. We show when and why hard rejection outperforms soft clipping.

---

## 1. OOD High-Weight Samples: Why Rejection Outperforms Clipping

### 1.1 The Problem: Clipping Retains Problematic Samples

In Part 2, we derived the Seq-TIS estimator:

{{< math >}}
$$
\hat{g}_{\text{seq-tis}}(y) = \min(\rho(y), C) \cdot f(y)
$$
{{< /math >}}

The implicit assumption was that all samples $y \sim \mu$ are valid learning signals—samples with high weights $\rho(y) = \pi(y)/\mu(y)$ simply require variance control via clipping.

However, this assumption fails in practice. Consider a sample with $\rho(y) = 10,000$. This means:

{{< math >}}
$$
\frac{\pi(y)}{\mu(y)} = 10,000 \implies \mu(y) = \frac{\pi(y)}{10,000}
$$
{{< /math >}}

Such extreme ratios typically arise when $\mu(y)$ is near the numerical precision floor (e.g., $\mu(y) \approx 10^{-9}$). These are **Out-of-Distribution (OOD)** samples—samples generated by policies outside the trust region. They occur due to:

1. **Numerical precision artifacts** in probability computation
2. **Distribution shift** between $\pi$ and $\mu$ beyond the valid IS regime

**The Clipping Problem:** When we apply Seq-TIS, we compute $\min(10000, C) \cdot f(y) = C \cdot f(y)$. The sample is still included in the gradient update with weight $C$. If $f(y)$ is malformed (due to the OOD nature of $y$), we introduce a systematic error into every gradient step.

### 1.2 The Solution: Hard Trust Region via Rejection (Seq-MIS)

Instead of soft clipping, we enforce a **Hard Trust Region**: samples outside the trusted region are rejected entirely.

{{% callout note %}}
**Definition: Sequence-Level Masked IS (Seq-MIS)**

{{< math >}}
$$
\hat{g}_{\text{seq-mis}}(y) = \mathbb{I}(\rho(y) \le C) \cdot \rho(y) \cdot f(y)
$$
{{< /math >}}

where $\mathbb{I}(\cdot)$ is the indicator function.
{{% /callout %}}

**Mathematical Interpretation:** The trust region is defined as:

{{< math >}}
$$
\mathcal{T}_C = \{y : \rho(y) \le C\} = \left\{y : \frac{\pi(y)}{\mu(y)} \le C\right\}
$$
{{< /math >}}

Only samples within this region contribute to the gradient. Samples with $\rho(y) \gt C$ are treated as unreliable and excluded.

**Connection to TRPO:** This implements the trust region concept from TRPO theory ([Part 1](https://richardli.xyz/rl-collapse-1)), where the trust region constraint $D_{TV}(\pi \| \mu) \le \delta$ must be enforced to guarantee improvement. Seq-MIS enforces this constraint via **rejection** (hard trust region) rather than **penalization** (soft trust region), ensuring that gradient updates only use samples from the valid trust region.

### 1.3 Bias-Variance Analysis of Seq-MIS

**Bias:** The Seq-MIS estimator is biased. By importance sampling identity:

{{< math >}}
$$
\mathbb{E}_\mu[\hat{g}_{\text{seq-mis}}] = \mathbb{E}_\mu[\mathbb{I}(\rho \le C) \cdot \rho \cdot f] = \sum_y \mu(y) \cdot \mathbb{I}(\rho(y) \le C) \cdot \frac{\pi(y)}{\mu(y)} \cdot f(y)
$$
{{< /math >}}

{{< math >}}
$$
= \sum_y \pi(y) \cdot \mathbb{I}(\rho(y) \le C) \cdot f(y) = \mathbb{E}_\pi[f \cdot \mathbb{I}(\rho \le C)]
$$
{{< /math >}}

The true gradient is $g = \mathbb{E}_\pi[f]$. Therefore:

{{< math >}}
$$
\text{Bias} = \mathbb{E}_\mu[\hat{g}_{\text{seq-mis}}] - g = \mathbb{E}_\pi[f \cdot \mathbb{I}(\rho \le C)] - \mathbb{E}_\pi[f] = -\mathbb{E}_\pi[f \cdot \mathbb{I}(\rho > C)]
$$
{{< /math >}}

The bias is the negative of the contribution from rejected samples (weighted by $\pi$).

**Variance:** Since $\mathbb{I}(\rho \le C) \cdot \rho \le C$ when the indicator is 1, and 0 otherwise:

{{< math >}}
$$
\mathbf{Var}(\hat{g}_{\text{seq-mis}}) \le \mathbb{E}_\mu[\|\hat{g}_{\text{seq-mis}}\|^2] \le C^2 \cdot \mathbb{E}_\mu[\|f\|^2] = O(T^2 C^2)
$$
{{< /math >}}

### 1.4 When to Use Each Estimator

| **Estimator** | **Mechanism** | **Use Case** |
| --- | --- | --- |
| **Seq-TIS** | $\min(\rho, C) \cdot f$ | Moderate mismatch; maximize sample efficiency |
| **Seq-MIS** | $\mathbb{I}(\rho \le C) \cdot \rho \cdot f$ | Large mismatch; OOD samples likely; prioritize robustness |

The choice depends on the reliability of high-weight samples. When the behavior policy $\mu$ is well-calibrated and mismatch is controlled, Seq-TIS extracts more information. When OOD samples are prevalent, Seq-MIS provides a Hard Trust Region that prevents gradient corruption.

---

## 2. Length-Dependent Rejection Bias: The Failure of Sequence-Level IS for Long Horizons

### 2.1 The Problem: Exponential Growth of the Importance Ratio

For autoregressive generation, the sequence-level importance ratio is a product of per-token ratios:

{{< math >}}
$$
\rho(y) = \prod_{t=0}^{T-1} \rho_t, \quad \text{where} \quad \rho_t = \frac{\pi(y_t|x, y_{\lt t})}{\mu(y_t|x, y_{\lt t})}
$$
{{< /math >}}

Even when the per-token mismatch is small, this product grows (or shrinks) exponentially with sequence length $T$.

**Formal Analysis:** Let $\bar{\rho} = \mathbb{E}[\rho_t]$ denote the expected per-token ratio. If we assume (for simplicity) that the $\rho_t$ are independent with $\bar{\rho} \gt 1$, then:

{{< math >}}
$$
\mathbb{E}[\rho(y)] = \prod_{t=0}^{T-1} \mathbb{E}[\rho_t] = \bar{\rho}^T
$$
{{< /math >}}

Similarly, for the log-ratio:

{{< math >}}
$$
\log \rho(y) = \sum_{t=0}^{T-1} \log \rho_t
$$
{{< /math >}}

If each $\log \rho_t$ has mean $\delta \gt 0$ (i.e., $\pi$ assigns slightly higher probability than $\mu$ on average), then:

{{< math >}}
$$
\mathbb{E}[\log \rho(y)] = T \cdot \delta \implies \rho(y) \approx e^{T\delta}
$$
{{< /math >}}

**Numerical Example:** Consider $\bar{\rho} = 1.001$ (0.1% per-token drift):

| Sequence Length $T$ | $\rho(y) \approx 1.001^T$ | Within Trust Region ($C=5$)? |
| --- | --- | --- |
| 10 | 1.01 | Accepted |
| 1,000 | 2.72 | Accepted |
| 2,000 | 7.39 | Rejected |
| 5,000 | 148.4 | Rejected |

### 2.2 The Consequence: Systematic Length Bias

This exponential scaling creates a **length-dependent acceptance probability**. For any fixed threshold $C$, there exists a critical length $T^*$ beyond which almost all samples are rejected:

{{< math >}}
$$
T^* = \frac{\log C}{\log \bar{\rho}}
$$
{{< /math >}}

For $C = 5$ and $\bar{\rho} = 1.001$: $T^* = \frac{\log 5}{\log 1.001} \approx 1609$ tokens.

**The Problem for Reasoning Models:** Chain-of-Thought (CoT) models and agents often generate sequences of 2,000-10,000+ tokens. With standard Seq-TIS or Seq-MIS:

1. Short responses (< 1000 tokens) are almost always accepted
2. Long reasoning chains (> 2000 tokens) are almost always rejected or heavily clipped
3. The model receives **systematically biased feedback** favoring short outputs

This is not a variance problem—it is a **structural bias** against long-horizon reasoning, independent of the quality of individual reasoning steps.

---

## 3. Geometric Rejection Sampling: A Per-Token Trust Region

### 3.1 From Extensive to Intensive Metrics

The fundamental problem with sequence-level IS is that $\rho(y) = \prod_t \rho_t$ is an **extensive quantity**—it scales with sequence length. We need an **intensive quantity** that measures the *average* per-token divergence, independent of length.

{{% callout note %}}
**Definition: Geometric Mean of the Importance Ratio**

{{< math >}}
$$
\rho_{\text{geo}}(y) = \left( \prod_{t=0}^{T-1} \rho_t \right)^{1/T} = \rho(y)^{1/T}
$$
{{< /math >}}
{{% /callout %}}

This is the geometric mean of the per-token ratios. It is length-invariant: if every $\rho_t = r$, then $\rho_{\text{geo}} = r$ regardless of $T$.

### 3.2 Mathematical Foundation: Connection to Per-Token KL Divergence

The geometric mean has a natural interpretation in terms of KL divergence. Taking the logarithm:

{{< math >}}
$$
\log \rho_{\text{geo}}(y) = \frac{1}{T} \sum_{t=0}^{T-1} \log \frac{\pi(y_t|x, y_{\lt t})}{\mu(y_t|x, y_{\lt t})}
$$
{{< /math >}}

This is the **sample average of the per-token log-ratios** along trajectory $y$.

**Connection to KL Divergence:** Let $s_t = (x, y_{\lt t})$ denote the state (context) at step $t$. Recall that:

- **Forward KL:** {{< math >}}$D_{KL}(\pi \| \mu) = \mathbb{E}_{y_t \sim \pi}\left[\log \frac{\pi(y_t|s_t)}{\mu(y_t|s_t)}\right]${{< /math >}}
- **Reverse KL:** {{< math >}}$D_{KL}(\mu \| \pi) = \mathbb{E}_{y_t \sim \mu}\left[\log \frac{\mu(y_t|s_t)}{\pi(y_t|s_t)}\right] = -\mathbb{E}_{y_t \sim \mu}\left[\log \frac{\pi(y_t|s_t)}{\mu(y_t|s_t)}\right]${{< /math >}}

Since samples are drawn from $\mu$ (the behavior policy), each term $\log \rho_t = \log \frac{\pi(y_t|s_t)}{\mu(y_t|s_t)}$ is a single-sample estimate of the **negative reverse KL**:

{{< math >}}
$$
\mathbb{E}_{y_t \sim \mu}\left[\log \frac{\pi(y_t|s_t)}{\mu(y_t|s_t)}\right] = -D_{KL}(\mu(\cdot|s_t) \| \pi(\cdot|s_t))
$$
{{< /math >}}

Therefore, $\log \rho_{\text{geo}}(y)$ can be interpreted as a trajectory-averaged sample of the negative reverse KL:

{{< math >}}
$$
\mathbb{E}_{y \sim \mu}\left[\log \rho_{\text{geo}}(y)\right] = -\frac{1}{T} \sum_{t=0}^{T-1} \mathbb{E}_{s_t \sim d_\mu}\left[D_{KL}(\mu(\cdot|s_t) \| \pi(\cdot|s_t))\right]
$$
{{< /math >}}

**Interpretation:** $\log \rho_{\text{geo}}(y)$ measures the average per-step log-likelihood ratio along the specific trajectory $y$. Unlike the sequence-level ratio, this quantity:

1. **Does not grow with $T$** (it's an average, not a sum)
2. **Can be positive or negative** ($\log \rho_{\text{geo}} \gt 0$ when $\pi$ assigns higher probability, $\lt 0$ when $\mu$ assigns higher probability)
3. **Detects both directions of drift** ($\rho_{\text{geo}} \ll 1$: policy forgetting; $\rho_{\text{geo}} \gg 1$: policy collapse)

{{% callout note %}}
**Connection to TRPO Theory (Part 1):** In [Part 1](https://richardli.xyz/rl-collapse-1), we showed that TRPO requires the trust region size to shrink with horizon: $\delta \propto 1/T^2$. This ensures the surrogate objective remains a valid approximation.

**Geo-RS achieves length-invariance via per-token log-ratio control.** By constraining $|\log \rho_{\text{geo}}| \le \epsilon$, we enforce:

{{< math >}}
$$
\left| \frac{1}{T} \sum_{t=0}^{T-1} \log \frac{\pi(y_t|s_t)}{\mu(y_t|s_t)} \right| \le \epsilon
$$
{{< /math >}}

This bounds the **average per-token log-ratio** along the trajectory. The key insight is that this constraint is **independent of sequence length $T$**—unlike sequence-level filtering where the threshold must scale as $O(1/T^2)$ to satisfy TRPO requirements, Geo-RS uses a fixed threshold $\epsilon$ that automatically adapts because it measures the *average* rather than the *total* divergence.

Thus, Geo-RS is a **practical implementation of the TRPO hard trust region** in the LLM context, with the crucial property that the acceptance criterion is **length-invariant**.
{{% /callout %}}

### 3.3 The Two-Sided Hard Trust Region (Geo-RS)

With the geometric ratio, we can define a **Per-Token Trust Region** that is independent of sequence length:

{{% callout note %}}
**Definition: Geometric Rejection Sampling (Geo-RS)**

$$
\hat{g}\_{\text{geo-rs}}(y) = \mathbb{I}\left( C\_{\text{low}} \le \rho\_{\text{geo}}(y) \le C\_{\text{high}} \right) \cdot f(y)
$$

Equivalently, in log-space:

$$
\hat{g}\_{\text{geo-rs}}(y) = \mathbb{I}\left( \log C\_{\text{low}} \le \frac{1}{T}\sum\_{t=0}^{T-1} \log \rho\_t \le \log C\_{\text{high}} \right) \cdot f(y)
$$
{{% /callout %}}

**Why Two-Sided?** The trust region enforces constraints in both directions:

| Condition | Meaning | Detection |
| --- | --- | --- |
| $\rho_{\text{geo}} \lt C_{\text{low}}$ | $\pi$ assigns much lower probability than $\mu$ on average | Policy has drifted away from high-likelihood regions of $\mu$ |
| $\rho_{\text{geo}} \gt C_{\text{high}}$ | $\pi$ assigns much higher probability than $\mu$ on average | Policy may be collapsing/overfitting to specific patterns |

**Typical Values:** $C_{\text{low}} = 0.5$, $C_{\text{high}} = 2.0$ (or equivalently, $|\log \rho_{\text{geo}}| \le \log 2 \approx 0.69$).

### 3.4 Bias-Variance Analysis of Geo-RS

**Bias:** Geo-RS is biased because it rejects samples:

{{< math >}}
$$
\mathbb{E}_\mu[\hat{g}_{\text{geo-rs}}] = \mathbb{E}_\mu[f \cdot \mathbb{I}(C_{\text{low}} \le \rho_{\text{geo}} \le C_{\text{high}})]
$$
{{< /math >}}

Unlike importance sampling, Geo-RS does **not** reweight by $\rho$—it only filters. This introduces bias but eliminates the variance explosion from high weights.

**Variance:** Since there is no importance weight multiplication, the variance is bounded by:

{{< math >}}
$$
\mathbf{Var}(\hat{g}_{\text{geo-rs}}) \le \mathbb{E}_\mu[\|f\|^2] = O(T^2)
$$
{{< /math >}}

**Length Invariance:** The acceptance criterion $C_{\text{low}} \le \rho_{\text{geo}} \le C_{\text{high}}$ is length-independent. A 100-token sequence and a 10,000-token sequence are judged by the same per-token divergence threshold.

### 3.5 Combining Geo-RS with Seq-TIS

In practice, we may want to combine both mechanisms:

1. **Geo-RS filter:** Reject samples with extreme per-token divergence (length-invariant safety)
2. **Seq-TIS weighting:** Apply importance sampling with clipping for accepted samples (bias correction)

{{% callout note %}}
**Definition: Geo-RS-Seq-TIS**

{{< math >}}
$$
\hat{g}_{\text{geo-rs-seq-tis}}(y) = \mathbb{I}\left( C_{\text{low}} \le \rho_{\text{geo}}(y) \le C_{\text{high}} \right) \cdot \min(\rho(y), C) \cdot f(y)
$$
{{< /math >}}
{{% /callout %}}

This estimator:
1. First checks if the sample is within the per-token trust region (Geo-RS)
2. Then applies clipped importance weighting for accepted samples (Seq-TIS)

**When to use which component:**

| Component | Purpose |
| --- | --- |
| Geo-RS ($\rho_{\text{geo}}$ filter) | Ensures length-invariant acceptance; detects per-token drift |
| Seq-TIS ($\min(\rho, C)$ weight) | Corrects for importance sampling bias within accepted samples |

---

## 4. Summary: Hierarchy of Estimators and Selection Guidelines

We have developed a hierarchy of estimators, each addressing specific failure modes:

| **Estimator** | **Formula** | **Trust Region Type** | **Primary Use Case** |
| --- | --- | --- | --- |
| **Token-IS (PPO)** | $\sum_t \min(\rho_t, C) A_t \nabla \log \pi_t$ | Per-token, soft | Stable but has $O(T^2\Delta_{\max})$ bias |
| **Seq-TIS** | $\min(\rho, C) \cdot f$ | Sequence-level, soft | Optimal bias-variance when all samples are valid |
| **Seq-MIS** | $\mathbb{I}(\rho \le C) \cdot \rho \cdot f$ | Sequence-level, hard | OOD sample filtering; large mismatch scenarios |
| **Geo-RS** | $\mathbb{I}(C_{\text{low}} \le \rho_{\text{geo}} \le C_{\text{high}}) \cdot f$ | Per-token, hard | Long-horizon tasks; length-invariant filtering |
| **Geo-RS-Seq-TIS** | Geo-RS filter × Seq-TIS weight | Hybrid | Long-horizon + importance correction |

For long-horizon reasoning tasks, **Geometric Rejection Sampling (Geo-RS)** provides a principled, length-invariant Hard Trust Region that prevents the systematic length bias inherent in standard importance sampling estimators.
