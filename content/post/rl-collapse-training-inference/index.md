---
title: 'When Speed Kills Stability: Demystifying RL Collapse from the Training-Inference Mismatch'
subtitle: 'Why the gap between fast inference and faithful training silently collapses LLM-RL, and how masked importance sampling (MIS) fixes it'
summary: 'The relentless push for faster inference creates a dangerous training-inference mismatch that silently kills RL with LLMs. We trace how it leads to collapse, most acutely in reasoning and agentic RL, and propose masked importance sampling (MIS): one importance ratio per response, with sequences outside the trust region masked out of the loss. Its length-normalized form, Geo-MIS, masks on the geometric mean of the per-token ratios, a one-sample estimate of the mean per-token KL that Trust Region Masking later computes exactly. In our multi-turn tool-use runs, token-level corrections still collapsed, while MIS trained past 700 steps where the uncorrected run collapsed near step 280.'
authors:
  - Jiacai Liu
  - admin
  - Yuqian Fu
  - Jiawei Wang
  - Qian Liu
  - Yu Shen
tags:
  - Reinforcement Learning
  - Language Models
  - Deep Learning
  - Training Dynamics
  - Importance Sampling
  - Masked Importance Sampling
categories:
  - Research
  - Theory
date: '2025-09-17T00:00:00Z'
lastmod: '2026-11-15T00:00:00Z'
featured: true
math: true
draft: false

citation: |
  @online{liu-li-2025-rl-collapse,
    title = {When Speed Kills Stability: Demystifying {RL} Collapse from the Training-Inference Mismatch},
    author = {Liu, Jiacai and Li, Yingru and Fu, Yuqian and Wang, Jiawei and Liu, Qian and Shen, Yu},
    year = {2025},
    month = sep,
    url = {https://richardli.xyz/rl-collapse}
  }

sidebar:
  toc: true
  toc_pin: [mis, geo-mis]
  links:
    - title: 'Series: RL Collapse'
      items:
        - text: 'Part 1: Why Off-Policy Breaks RL'
          note: 'Bias (total variation) and variance (χ²) as two failure modes, in an SGA framework.'
          url: '/post/rl-collapse-part1/'
        - text: 'Part 2: Token vs. Sequence-Level Correction'
          note: 'Token-level IS is biased; sequence-level IS is unbiased but has exponential variance.'
          url: '/post/rl-collapse-part2/'
        - text: 'Part 3: Trust Region Optimization via Sequence Masking'
          note: 'Seq-MIS, and the two-sided, length-invariant Geo-Mask.'
          url: '/post/rl-collapse-part3/'
    - title: Related
      items:
        - text: 'Trust Region Masking (ICML 2026)'
          note: 'The follow-up paper: sequence masks on the exact mean and max token-level KL.'
          url: 'https://arxiv.org/abs/2512.23075'
        - text: 'Rollout Correction in VeRL'
          note: 'MIS and sequence-level correction in a public training stack.'
          url: 'https://verl.readthedocs.io/en/latest/algo/rollout_corr.html'
        - text: 'Original Notion version'
          note: 'As first published, September 17, 2025.'
          url: 'https://yingru.notion.site/When-Speed-Kills-Stability-Demystifying-RL-Collapse-from-the-Training-Inference-Mismatch-271211a558b7808d8b12d403fd15edda'
  notation:
    - sym: '\pvllm'
      text: 'vLLM sampler, the behavior policy; written μ in Parts 1–3.'
    - sym: '\pfsdp'
      text: 'FSDP trainer, the target policy; written π in Parts 1–3.'
    - sym: '\rho(y|x)'
      text: 'Sequence-level ratio, trainer over sampler; per-token ratio ρ_t.'
    - sym: 'C'
      text: 'Threshold: TIS clips the ratio at C, MIS masks sequences above it.'
    - sym: '\rho_{\mathrm{geo}}'
      text: 'Geometric mean of the per-token ratios, ρ^(1/T), over the T model-generated tokens; Geo-MIS keeps a sequence when it lies in [C_low, C_high].'
    - sym: 'D_t'
      text: 'KL divergence from the sampler to the trainer at the context of token t.'
  updates:
    - date: 'Dec 2, 2025'
      text: '[DeepSeek-V3.2](https://arxiv.org/abs/2512.02556) uses a length-normalized sequence mask, closely related to our geometric sequence masking.'
    - date: 'Dec 1, 2025'
      text: 'Discussed in [Stabilizing Reinforcement Learning with LLMs: Formulation and Practices](https://arxiv.org/abs/2512.01374) from the Qwen team.'
    - date: 'Nov 12, 2025'
      text: 'Rollout Correction for general off-policy problems merged into VeRL: [usage](https://verl.readthedocs.io/en/latest/algo/rollout_corr.html), [more details](https://verl.readthedocs.io/en/latest/algo/rollout_corr_math.html) (Yingru Li).'
    - date: 'Nov 10, 2025'
      text: 'Discussed in a [vLLM blog post](https://blog.vllm.ai/2025/11/10/bitwise-consistent-train-inference.html), which implemented bitwise-consistent on-policy RL.'
    - date: 'Oct 31, 2025'
      text: 'The VeRL fully async module integrated our work ([PR](https://github.com/volcengine/verl/pull/3955)).'
    - date: 'Oct 20, 2025'
      text: 'Slime integrated our work ([PR](https://github.com/THUDM/slime/pull/429)), by SGLang RL: Chenyang Zhao and Jiajun Li.'
    - date: 'Oct 16, 2025'
      text: 'Cited by [SWE-grep](https://cognition.ai/blog/swe-grep#motivation) at Cognition, which also uses sequence-level masked importance sampling.'
    - date: 'Oct 15, 2025'
      text: 'A Megatron training crash was resolved with geometric-level masking.'
    - date: 'Oct 13, 2025'
      text: '[VeRL 0.6.0](https://github.com/volcengine/verl/releases/tag/v0.6.0) integrated our work ([PR](https://github.com/volcengine/verl/pull/3694)), including sequence masking with the geometric mean of the importance weight (Geo-MIS/Geo-RS) (Yingru Li).'
---

**Authors:** [Jiacai Liu](https://scholar.google.com/citations?user=zt9Jfh4AAAAJ)\*, [Yingru Li](/)\*†, Yuqian Fu, Jiawei Wang, Qian Liu, Yu Shen†

*\*Co-first authors. †Corresponding authors. Work done at ByteDance. First published September 17, 2025.*

**Follow-up paper:** [Trust Region Masking for Long-Horizon LLM Reinforcement Learning](https://arxiv.org/abs/2512.23075) (ICML 2026)

<div class="alert alert-note" id="tldr">
<div>

**What this post contributes.** Fast inference engines and faithful training frameworks compute slightly different probabilities from the same weights. This training-inference mismatch silently turns on-policy RL into off-policy RL, and in reasoning and agentic RL it can collapse a run. The post gives a diagnosis and a fix.

**Diagnosis: five findings** ([§3](#3-anatomy-of-the-training-collapse))

1. **A metric that warns.** `vllm-kl` measures the mismatch, and its spikes come together with the spikes in entropy, perplexity and gradient norm that precede a collapse ([§3.1](#31-measuring-the-mismatch-the-vllm-kl-metric)).
2. **The engines disagree on low-probability tokens.** The trainer can give a sampled token a probability orders of magnitude below the sampler's, and the batches where this happens are the ones whose gradients explode ([§3.2](#low-probability-tokens)).
3. **Out-of-distribution tool responses make it worse.** After a tool response the model samples low-probability tokens more often, so multi-turn agentic runs fail first ([§3.3](#ood-tool-responses)).
4. **Hardware is a first-order variable.** With identical code the mismatch ranks H20 < L20 < A100, and a run that collapses on one GPU trains on another ([§3.4](#hardware)).
5. **The mismatch grows during training.** Once a run enters a high-mismatch state it stays there; we hypothesize a feedback loop between the optimizer and numerical error ([§3.5](#36-the-mismatch-is-not-static-a-vicious-cycle-driven-by-optimization)).

**Fix: sequence-level masking** ([§4](#4-attempts-to-alleviate-the-training-inference-mismatch)). Token-level corrections reweight each token but not the distribution of prefixes, and in our multi-turn tool-use runs they still collapsed. **Masked importance sampling (MIS)** computes one ratio for the whole response, {{< math >}}$\rho(y|x) =${{< /math >}} {{< math >}}$\pfsdp(y|x)/\pvllm(y|x)${{< /math >}}, and masks the policy loss of every sequence whose ratio exceeds a threshold {{< math >}}$C${{< /math >}}. **Geo-MIS** applies the mask, on both sides, to the geometric mean of the per-token ratios, so that one pair of thresholds serves responses of every length {{< math >}}$T${{< /math >}}:

{{< math >}}
$$
\begin{gathered}
\text{MIS:}\quad \rho(y|x) \gets \rho(y|x)\,\mathbb{I}\{\rho(y|x) \le C\}, \\
\text{Geo-MIS:}\quad \rho_{\mathrm{geo}} = \rho(y|x)^{1/T}, \qquad \rho_{\mathrm{geo}} \gets \rho_{\mathrm{geo}}\,\mathbb{I}\{C_{\text{low}} \le \rho_{\mathrm{geo}} \le C_{\text{high}}\}.
\end{gathered}
$$
{{< /math >}}

On L20 GPUs, MIS trained for more than 700 steps where the uncorrected run collapsed near step 280, raised the peak training reward from 0.56 to 0.64, and matched the uncorrected peak test accuracy (0.70 vs. 0.68; Figure 14, [§4.3](#mis)). Geo-MIS was released in VeRL 0.6.0. Its statistic {{< math >}}$-\log\rho_{\mathrm{geo}}${{< /math >}} is a one-sample estimate of the mean per-token KL of the response, the quantity `vllm-kl` monitors; the follow-up paper, Trust Region Masking (ICML 2026), computes that KL exactly and adds a max-KL criterion ([§4.4](#geo-mis)).

</div>
</div>

{{< stepfig "rlc-loop" >}}

**How to read this post.** [§1](#1-the-mystery-of-the-sudden-collapse) and [§2](#2-a-fundamental-conflict-the-growing-gap-between-inference-and-training) state the problem. [§3](#3-anatomy-of-the-training-collapse) is the evidence, one finding per subsection. [§4](#4-attempts-to-alleviate-the-training-inference-mismatch) is the method: why the correction must act on the whole sequence, then MIS, then Geo-MIS and its link to Trust Region Masking. [§5](#other-settings) lists the engine and hardware settings we tried, with a summary table, and [§6](#5-conclusion-and-key-takeaways-for-practitioners) has the takeaways for practitioners. Short on time: read the box above, the figure, [§4.3](#mis) and [§6](#5-conclusion-and-key-takeaways-for-practitioners).

---

## 1. The Mystery of the Sudden Collapse

In the rapidly advancing field of reinforcement learning for large language models (LLM-RL), a frustrating pattern of sudden training collapse is emerging. Whether in complex **reasoning RL** or multi-turn **agentic RL**, many have observed training runs that, after a period of stable learning, catastrophically fail.

We recently encountered this firsthand while conducting agentic RL experiments for multi-turn tool-integrated reasoning (TIR) on Qwen3 models. This occurred across both on-policy and off-policy variants of the GRPO algorithm on our L20 GPU cluster. Figure 1 shows the reward and gradient norm dynamics of our four crashed experiments on Qwen3-14B-Base. After a period of steady progress, rewards collapse, and in two of the runs the gradient norm explodes as they do. Our initial investigation focused on common culprits:

- We examined the code and confirmed that our agent loop follows a token-in-token-out process.
- We tuned the hyperparameters `beta1` and `beta2` in the Adam optimizer.
- We also applied batch normalization to the advantages to balance the updates.

However, none of these standard fixes worked. Since even the simpler on-policy experiments failed, we suspected the issue was not with the RL algorithm but with a more fundamental part of the training stack. This led us to investigate a critical and increasingly prevalent challenge in modern LLM-RL: the unavoidable gap between highly optimized inference engines and faithful training frameworks.

<figure class="post-fig">
<img src="/figs/img/rl-collapse/fig01-collapsed-runs.png" alt="Four GRPO runs on Qwen3-14B-Base: rewards rise and then fall sharply between steps 250 and 465, with large gradient-norm spikes in the on-policy and clip-higher 0.28 runs as they collapse.">
<figcaption><strong>Figure 1.</strong> Rewards (<strong>left</strong>) and gradient norms (<strong>right</strong>) from our four failed GRPO TIR experiments on Qwen3-14B-Base. All experiments sample <code>1024</code> trajectories (<code>64</code> prompts × <code>16</code> responses) at each training step and use a learning rate of <code>1e-6</code>. The <code>ppo_mini_batch_size</code> is set to <code>1024</code> and <code>256</code> for on-policy and off-policy experiments, respectively. The off-policy runs differ in clip-higher, DAPO's upper clip ratio.</figcaption>
</figure>

---

## 2. A Fundamental Conflict: The Growing Gap Between Inference and Training

Rollout speed is a core bottleneck in LLM-RL. To reach the throughput it needs, modern inference engines (e.g., vLLM, SGLang, TensorRT-LLM) use aggressive optimizations such as speculative decoding, low-precision computation (INT8/FP8), and specialized, batch-variant CUDA kernels. They maintain sampling fidelity, but their primary objective is throughput, often measured in tokens per second.

Training frameworks (e.g., FSDP, DeepSpeed, Megatron-LM) strike a different balance. They prioritize numerical stability and precision for gradient computation, and often keep master weights and optimizer states in higher-precision formats such as FP32.

**This divergence in priorities makes a training-inference mismatch hard to avoid**, and the push for faster rollouts is making the gap wider, not smaller. One can enforce identical calculations (e.g., with ["batch invariant kernels"](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)), but this costs substantial throughput, which works against the reason for using a high-speed inference engine. The speed-versus-consistency trade-off is at the heart of the problem: it is a persistent challenge rather than a simple engineering fix.

In our stack, this mismatch manifested between our vLLM inference sampler and our FSDP trainer. The actual parameter update was

{{< math >}}
$$
\mathbb{E}_{x\sim \mathcal{D}}\,\mathbb{E}_{y\sim \pvllm(\cdot|x)}\left[ R(x,y)\, \nabla_{\theta}\log \pfsdp(y|x) \right],
$$
{{< /math >}}

whereas the theoretical parameter update should be

{{< math >}}
$$
\mathbb{E}_{x\sim \mathcal{D}}\,\mathbb{E}_{y\sim \pfsdp(\cdot|x)}\left[ R(x,y)\, \nabla_{\theta}\log \pfsdp(y|x) \right].
$$
{{< /math >}}

Here {{< math >}}$x${{< /math >}} is the prompt sampled from the distribution {{< math >}}$\mathcal{D}${{< /math >}}, {{< math >}}$y${{< /math >}} is the response, {{< math >}}$R${{< /math >}} is the reward function, {{< math >}}$\theta${{< /math >}} is the parameter of the LLM, and {{< math >}}$\pvllm${{< /math >}} and {{< math >}}$\pfsdp${{< /math >}} are the policies implemented in the vLLM engine and the FSDP engine, respectively. To investigate this, we first needed a way to measure it.

---

## 3. Diagnosis: Five Findings {#3-anatomy-of-the-training-collapse}

### Setup

Unless otherwise specified, the experiments in Sections 3 to 5 use:

- **Framework:** [VeRL](https://github.com/volcengine/verl), with the vLLM v1 sampler (AsyncvLLMServer) and an FSDP trainer.
- **Model and algorithm:** Qwen3-14B-Base trained with GRPO on multi-turn tool-integrated reasoning (TIR).
- **Batch and optimizer:** 64 prompts × 16 responses = 1024 trajectories per step, learning rate 1e-6, `ppo_mini_batch_size` 1024 for on-policy and 256 for off-policy runs.
- **Hardware:** L20 GPUs.

Experiments that change the hardware, the model, or the task (single-turn reasoning RL) say so.

### 3.1 Finding 1: `vllm-kl` Spikes Come with the Signs of Collapse {#31-measuring-the-mismatch-the-vllm-kl-metric}

**One number, `vllm-kl`, measures the mismatch. Its spikes line up with spikes in entropy, perplexity and gradient norm, and it rises before the reward collapses.**

A very straightforward metric for measuring the training-inference mismatch is **vllm-kl**:

{{< math >}}
$$
\mathbb{E}_{s\sim d_{\pvllm}}\left[\text{KL}\left(\pvllm(\cdot|s)\,\|\,\pfsdp(\cdot|s)\right)\right] = \mathbb{E}_{s\sim d_{\pvllm},\,a\sim \pvllm(\cdot|s)} \left[\log\left(\frac{\pvllm(a|s)}{\pfsdp(a|s)}\right)\right],
$$
{{< /math >}}

where {{< math >}}$s${{< /math >}} is the context prefix (state), {{< math >}}$a${{< /math >}} is the token (action), and {{< math >}}$d_\pi${{< /math >}} is the distribution of prefixes visited by policy {{< math >}}$\pi${{< /math >}} (defined precisely in [§4.1](#distribution-correction)). Our experiments involve tool calls, so the response {{< math >}}$y${{< /math >}} may include tool responses; our `vllm-kl` metric only considers tokens generated by the model itself, and averages over them. The formula above is the plain log-ratio form; the code below uses the [K3 estimator](http://joschu.net/blog/kl-approx.html), {{< math >}}$r - 1 - \log r${{< /math >}} with {{< math >}}$r = \pfsdp(a|s)/\pvllm(a|s)${{< /math >}}, which has the same expectation, is never negative, and has lower variance. It computes `vllm-kl` in [VeRL](https://github.com/volcengine/verl), assuming the inference engine's token probabilities are already available:

<details class="spoiler">
<summary>K3 estimator for <code>vllm-kl</code></summary>

```python
rollout_log_probs = batch.batch["rollout_log_probs"]  # pi_vllm
actor_old_log_probs = batch.batch["old_log_probs"]    # pi_fsdp
response_mask = batch.batch["response_mask"]
log_ratio = actor_old_log_probs - rollout_log_probs
vllm_k3_kl_matrix = torch.exp(log_ratio) - log_ratio - 1
vllm_k3_kl = masked_mean(vllm_k3_kl_matrix, response_mask)
```

</details>

<span id="32-the-warning-signs-correlated-instability"></span>

Our first clue was that high `vllm-kl` values were not isolated events. They correlated strongly with other signs of instability.

#### Entropy and rewards fluctuate with `vllm-kl` {#321-fluctuations-in-fsdp-entropy-and-rewards}

In many of our experiments, abnormal spikes in `vllm-kl` coincide with anomalous fluctuations in the entropy of the FSDP policy {{< math >}}$\pfsdp${{< /math >}} and in rewards. Figure 2 is an intuitive example: **the locations of the entropy spikes almost perfectly match the locations of the `vllm-kl` spikes.** The correlation is less pronounced in the rewards, but a massive `vllm-kl` spike around step 250 coincided with a low-quality batch and a noticeable dip in reward. **When the mismatch is large, both the vLLM policy {{< math >}}$\pvllm${{< /math >}} and the FSDP policy {{< math >}}$\pfsdp${{< /math >}} enter an unstable region.**

<figure class="post-fig">
<img src="/figs/img/rl-collapse/fig02-entropy-reward-kl.png" alt="An on-policy run: entropy (left) and rewards (right) plotted with vllm-kl; entropy spikes line up with vllm-kl spikes, and a large vllm-kl spike near step 250 coincides with a reward dip.">
<figcaption><strong>Figure 2.</strong> An on-policy experiment showing entropy (<strong>left</strong>) and rewards (<strong>right</strong>) alongside <code>vllm-kl</code>, illustrating their correlation during training.</figcaption>
</figure>

#### FSDP perplexity and gradient norm rise before the collapse {#322-rising-fsdp-ppl-and-gradient-norm-precede-policy-collapse}

More critically, spikes in `vllm-kl` coincided with spikes in both the FSDP [perplexity](https://en.wikipedia.org/wiki/Perplexity) (`fsdp-ppl`) and the gradient norm. In our experiments, the `fsdp-ppl` of a response {{< math >}}$y${{< /math >}} is

{{< math >}}
$$
\text{fsdp-ppl} = \exp\left(\frac{-1}{\left| \mathcal{T}_{\mathcal{M}}(y) \right|}\sum_{t\in \mathcal{T}_{\mathcal{M}}(y)}\log \pfsdp\left( y_t|y_{\lt t} \right)\right),
$$
{{< /math >}}

where {{< math >}}$\mathcal{T}_{\mathcal{M}}(y)${{< /math >}} is the index set of the tokens generated by the model itself in the response {{< math >}}$y${{< /math >}}. The batch's `fsdp-ppl` is the average over all responses in the batch. Figure 3 shows an on-policy and an off-policy version of GRPO. In both, the spikes in `vllm-kl` line up almost exactly with spikes in `fsdp-ppl` and the gradient norm. Moreover, `vllm-kl` rises significantly before the training reward collapses.

<figure class="post-fig">
<img src="/figs/img/rl-collapse/fig03a-onpolicy-spikes.png" alt="On-policy run, steps 200 to 280: gradient norm, fsdp-ppl and rewards each plotted with vllm-kl; spikes in vllm-kl coincide with spikes in gradient norm and fsdp-ppl near step 210 and from step 265 on, as rewards drop.">
<img src="/figs/img/rl-collapse/fig03b-offpolicy-spikes.png" alt="Off-policy run with clip-higher 0.28: gradient norm, fsdp-ppl and rewards each plotted with vllm-kl; the same coincident spikes appear before the reward collapses.">
<figcaption><strong>Figure 3.</strong> An on-policy experiment from step 200 to 280 (<strong>top</strong>) and an off-policy experiment with clip-higher (DAPO's upper clip ratio) 0.28 (<strong>bottom</strong>), showing gradient norm (<strong>left</strong>), fsdp-ppl (<strong>middle</strong>) and rewards (<strong>right</strong>) alongside <code>vllm-kl</code>.</figcaption>
</figure>

Late in training the policy's entropy is low (about 0.2 in Figure 2), so its perplexity on its own samples should stay close to 1. In batches with a large mismatch (high `vllm-kl`), `fsdp-ppl` instead spiked (to about 1.27 in Figure 3, top). **The FSDP engine was assigning very low probabilities to some tokens that the inference engine had sampled, and these were the batches whose gradient norm spiked.** This observation helped us pinpoint where mismatches are more likely to occur. As we will see, when these extremely low-FSDP-probability tokens were sampled, their probabilities in the vLLM engine were not nearly as low.

### 3.2 Finding 2: The Mismatch Concentrates in Low-Probability Tokens {#low-probability-tokens}

**The mismatch is not uniform: it sits in the tokens the sampler considers unlikely.**

By analyzing batches with varying levels of `vllm-kl`, we found a stark pattern. As a token's inference probability approaches zero, the training probability can become orders of magnitude smaller, making PPL and gradients explode. To make the conclusions as general as possible, we selected batches from various experiments before training collapse, taken at different training steps. Most of them have relatively high `vllm-kl`, which lets us study mismatch patterns under pronounced conditions; Group 1 serves as a low-mismatch reference. We collected rollout batches in three `vllm-kl` ranges, with five batches (around 5M tokens) in each group:

- **Group 1 (low):** each batch has `vllm-kl` ≤ 1e-3, sampled on H20 GPUs.
- **Group 2 (medium):** each batch has `vllm-kl` in [1e-3, 2e-2], sampled on L20 GPUs.
- **Group 3 (high):** each batch has `vllm-kl` in [2e-2, 1e-1], sampled on L20 GPUs.

Group 1 comes from H20 GPUs and Groups 2 and 3 from L20 GPUs, so the level of mismatch and the hardware vary together here. [Section 3.4](#hardware) isolates the effect of the hardware with otherwise identical runs.

Figure 4 shows the relationship between the vLLM output probability {{< math >}}$\pvllm(a|s)${{< /math >}} and the mismatch, measured by {{< math >}}$\log \pfsdp(a|s) - \log \pvllm(a|s)${{< /math >}}, at each level of `vllm-kl`:

<figure class="post-fig">
<img src="/figs/img/rl-collapse/fig04a-mismatch-group1.png" alt="Group 1 scatter of log-probability difference against vLLM probability: points hug zero across the range, with a modest spread near zero probability.">
<figcaption><strong>Figure 4 (a).</strong> {{< math >}}$\pvllm${{< /math >}} vs. {{< math >}}$\log \pfsdp - \log \pvllm${{< /math >}} for tokens in the batches of group 1.</figcaption>
</figure>

<figure class="post-fig">
<img src="/figs/img/rl-collapse/fig04b-mismatch-group2.png" alt="Group 2 scatter of log-probability difference against vLLM probability: a wider, mostly negative spread that is largest as the vLLM probability approaches zero.">
<figcaption><strong>Figure 4 (b).</strong> {{< math >}}$\pvllm${{< /math >}} vs. {{< math >}}$\log \pfsdp - \log \pvllm${{< /math >}} for tokens in the batches of group 2.</figcaption>
</figure>

<figure class="post-fig">
<img src="/figs/img/rl-collapse/fig04c-mismatch-group3.png" alt="Group 3 scatter of log-probability difference against vLLM probability: a dense band near zero with a long tail down to minus 15, densest where the vLLM probability is near zero.">
<figcaption><strong>Figure 4 (c).</strong> {{< math >}}$\pvllm${{< /math >}} vs. {{< math >}}$\log \pfsdp - \log \pvllm${{< /math >}} for tokens in the batches of group 3.</figcaption>
</figure>

From the three figures we observe that:

1. The mismatch tends to be more pronounced when the vLLM probability {{< math >}}$\pvllm${{< /math >}} approaches zero, and extreme values of {{< math >}}$\log \pfsdp - \log \pvllm${{< /math >}} are more likely there.
2. The batches collected on L20 GPUs (groups 2 and 3) show a training-inference mismatch mainly in the form of the FSDP probability {{< math >}}$\pfsdp${{< /math >}} being much smaller than the vLLM probability {{< math >}}$\pvllm${{< /math >}}.

Two caveats in reading these plots. Every token here was sampled by vLLM, so the average of {{< math >}}$\log \pvllm - \log \pfsdp${{< /math >}} over the sampled tokens is the KL divergence, which is never negative: some downward shift is expected, and what matters is the size of the tail, which reaches −15 in group 3. And the upper edge of each cloud is the bound {{< math >}}$\log \pfsdp - \log \pvllm \le -\log \pvllm${{< /math >}}, which widens as {{< math >}}$\pvllm \to 0${{< /math >}} by construction.

### 3.3 Finding 3: Out-of-Distribution Tool Responses Amplify the Mismatch {#ood-tool-responses}

**After a tool response the model samples low-probability tokens more often. The mismatch therefore sits mainly in non-first-round outputs, and more tool calls bring an earlier collapse.**

#### The mismatch is more severe in non-first-round outputs {#341-the-mismatch-is-more-severe-in-non-first-round-outputs}

Finding 2 explains why the problem was so acute in our multi-turn TIR experiments, **particularly in non-first-round model outputs**. The process works as follows:

1. The **agent** receives a tool response, which is often structured text (e.g., content wrapped in `<python_output>` and `</python_output>` tags) that is OOD compared to its pre-training and SFT data.
2. Faced with this unfamiliar OOD context, the **agent's policy** becomes more uncertain and more likely to sample low-probability tokens in its subsequent turns (also observed in [SimpleTIR](https://github.com/ltzheng/SimpleTIR)).
3. As we just established, these low-probability tokens are where severe mismatch occurs, creating the conditions for an `fsdp-ppl` and gradient explosion.

Next, we plot the mismatch in the batches of the three groups of Finding 2 (around 5k trajectories per group), separating first-round model outputs from non-first-round outputs. We use two views:

1. **Log-ppl scatter plot.** The x-axis is the log of the ppl under the vLLM policy {{< math >}}$\pvllm${{< /math >}} (`vllm-log-ppl`), and the y-axis is the log of the ppl under the FSDP policy {{< math >}}$\pfsdp${{< /math >}} (`fsdp-log-ppl`).
2. **Probability scatter plot.** The x-axis is the token probability under {{< math >}}$\pvllm${{< /math >}} and the y-axis is the token probability under {{< math >}}$\pfsdp${{< /math >}}.

**Group 3: batches with `vllm-kl` in [2e-2, 1e-1]**

<figure class="post-fig">
<img src="/figs/img/rl-collapse/scatter-group3.png" alt="Group 3, four scatter plots: first-round outputs stay near the diagonal, while non-first-round outputs spread far above it in log-ppl (mean absolute difference 0.111) and fill the whole square in token probability.">
<figcaption>Group 3 (<code>vllm-kl</code> in [2e-2, 1e-1]). <strong>Top:</strong> log-ppl under vLLM vs. FSDP. <strong>Bottom:</strong> token probability under vLLM vs. FSDP. <strong>Left:</strong> first-round outputs. <strong>Right:</strong> non-first-round outputs.</figcaption>
</figure>

<details class="spoiler">
<summary>Groups 1 and 2: the same plots at lower mismatch</summary>

**Group 1: batches with `vllm-kl` ≤ 1e-3**

<figure class="post-fig">
<img src="/figs/img/rl-collapse/scatter-group1.png" alt="Group 1, four scatter plots: log-ppl and token probability, vLLM against FSDP, for first-round and non-first-round outputs. All four lie tightly on the diagonal, with correlations of about 0.998 or higher.">
<figcaption>Group 1 (<code>vllm-kl</code> ≤ 1e-3). Same layout as Group 3.</figcaption>
</figure>

**Group 2: batches with `vllm-kl` in [1e-3, 2e-2]**

<figure class="post-fig">
<img src="/figs/img/rl-collapse/scatter-group2.png" alt="Group 2, four scatter plots: first-round outputs stay on the diagonal, while non-first-round outputs show points above it in log-ppl (mean absolute difference 0.021) and a wider spread in token probability.">
<figcaption>Group 2 (<code>vllm-kl</code> in [1e-3, 2e-2]), same layout as Group 3.</figcaption>
</figure>

</details>

From these plots we observe that:

1. The `vllm-log-ppl` of non-first-round outputs is generally larger than that of first-round outputs: the model samples more low-probability tokens when it faces unfamiliar OOD context.
2. The mismatch occurs mainly in non-first-round outputs, which show a larger mean absolute difference between the FSDP and vLLM policies, in both log-ppl and token probability, and a lower Pearson correlation.
3. As `vllm-kl` increases, the mismatch worsens mainly in non-first-round outputs.
4. `fsdp-log-ppl` is consistently above `vllm-log-ppl`. Some gap is expected, as noted above, but it is largest for non-first-round outputs, where FSDP assigns much lower probability than vLLM to the sampled tokens.

#### More tool calls, more training instability {#342-more-tool-calls-more-training-instability}

The following experiments also show that OOD tool responses exacerbate the training-inference mismatch and training instability. We ran off-policy GRPO on H20 GPUs with Qwen3-14B-Base as the base model, clip-higher 0.28 and 4 mini-batches, and set the maximum number of tool calls per trajectory (`max_tool_turn`) to 20 or 100. Figures 5 and 6 show the results: the more tool calls, the earlier the training collapses. Every collapse comes with a gradient explosion and an explosion in `vllm-kl`.

<figure class="post-fig">
<img src="/figs/img/rl-collapse/fig05-max-tool-turn.jpg" alt="Off-policy GRPO with max_tool_turn of 20 and 100: gradient norm, rewards and vllm-kl; the run allowing 100 tool calls collapses earlier, with spikes in gradient norm and vllm-kl.">
<figcaption><strong>Figure 5.</strong> Gradient norm (<strong>left</strong>), rewards (<strong>middle</strong>) and <code>vllm-kl</code> (<strong>right</strong>) of off-policy GRPO experiments with different values of <code>max_tool_turn</code>. All experiments ran on H20 GPUs with Qwen3-14B-Base, clip-higher 0.28 and 4 mini-batches.</figcaption>
</figure>

<figure class="post-fig">
<img src="/figs/img/rl-collapse/fig06-tool-calls.jpg" alt="Maximum and average number of tool calls per trajectory over training, for max_tool_turn of 20 and 100.">
<figcaption><strong>Figure 6.</strong> Maximum (<strong>left</strong>) and average (<strong>right</strong>) number of tool calls in the same experiments as Figure 5.</figcaption>
</figure>

### 3.4 Finding 4: Hardware Is a First-Order Variable {#hardware}

**The exact same code and model produce drastically different levels of mismatch on different GPUs.** To measure this, we ran on-policy algorithms with identical code and hyperparameters, switching only the GPUs used for inference and training. Figure 7 shows the training dynamics on L20, H20 and A100.

<figure class="post-fig">
<img src="/figs/img/rl-collapse/fig07-hardware.png" alt="vllm-kl (left, log scale) and rewards (right) for the same on-policy run on H20, L20 and A100 GPUs: vllm-kl is lowest on H20 and highest on A100, where rewards are unstable.">
<figcaption><strong>Figure 7.</strong> <code>vllm-kl</code> (<strong>left</strong>) and rewards (<strong>right</strong>) of on-policy GRPO on different GPUs, with the same code and environment.</figcaption>
</figure>

In our experiments, the magnitude of `vllm-kl` follows **H20 < L20 < A100**. It is on the order of 5e-4 to 1e-3 on H20, around 1e-3 to 1e-2 on L20, and on A100 it swings between about 1e-3 and 5e-1, with frequent spikes above 1e-1 (that run lasted only about 50 steps). With such a severe mismatch on A100, normal training is not feasible and the reward curve is highly unstable.

The clearest evidence came when we took a failed L20 experiment and resumed it from a checkpoint on H20 GPUs (Figure 8). On H20 the run stayed stable, with a low gradient norm and rewards around 0.6, while the original run on L20 collapsed: the hardware has a first-order effect on the problem.

<figure class="post-fig">
<img src="/figs/img/rl-collapse/fig08-l20-to-h20.png" alt="On L20 (red) the gradient norm spikes and rewards collapse after step 200, while the same checkpoint resumed on H20 (green) keeps a low gradient norm and rewards around 0.6.">
<figcaption><strong>Figure 8.</strong> On-policy GRPO on L20 and H20, showing gradient norm (<strong>left</strong>) and rewards (<strong>right</strong>). We resumed on H20 GPUs (green) from the step-200 checkpoint of the failed L20 experiment (red).</figcaption>
</figure>

Two changes on the hardware side reduce the mismatch: moving from L20 to H20, and disabling cascade attention in vLLM on A100. Both are in [§5.3](#hardware-fixes).

### 3.5 Finding 5: The Mismatch Grows During Training {#36-the-mismatch-is-not-static-a-vicious-cycle-driven-by-optimization}

**The mismatch is not a fixed property of the stack. Once a run enters a high-mismatch state it keeps producing high-mismatch batches, and we hypothesize a feedback loop between the optimizer and numerical error.**

The evidence is our "batch-filter" experiment, which shows that the mismatch is coupled with the training dynamics and the model's state.

In the **batch-filter** experiment, at each training step, if the collected batch has a `vllm-kl` above a threshold, we skip the update on that batch, since such updates are prone to cause collapse. We go straight to the next step and keep collecting data until a batch with `vllm-kl` below the threshold arrives, and only then update the model.

If the degree of mismatch were entirely independent of the model's output distribution and the training dynamics, `vllm-kl` should have the same distribution at every training step. However, Figure 9 shows that **once the model entered a certain state, it consistently generated high-mismatch batches**, effectively halting training. This, together with the spiraling `vllm-kl` and `fsdp-ppl` in other runs (Figure 10), points to a dangerous feedback loop.

<figure class="post-fig">
<img src="/figs/img/rl-collapse/fig09-batch-filter.png" alt="Batch-filter experiment: vllm-kl and rewards over training steps, with filtered steps shaded pink; late in training nearly every batch exceeds the threshold and is filtered.">
<figcaption><strong>Figure 9.</strong> The on-policy batch-filter experiment, showing <code>vllm-kl</code> (<strong>left</strong>) and rewards (<strong>right</strong>). Steps whose batches were filtered out are highlighted in pink. The filter threshold is 0.1.</figcaption>
</figure>

<figure class="post-fig">
<img src="/figs/img/rl-collapse/fig10-kl-ppl-spiral.png" alt="vllm-kl and fsdp-ppl over training steps in an on-policy run: the two curves rise together in a correlated spiral.">
<figcaption><strong>Figure 10.</strong> <code>vllm-kl</code> and fsdp-ppl over training steps in an on-policy experiment. The two curves rise together in a correlated, spiral-like pattern.</figcaption>
</figure>

**We hypothesize a two-stage failure cascade:**

1. **Stage 1: increased numerical sensitivity.** The RL optimizer moves the weights into regions where `bfloat16` rounding matters more. `bfloat16` keeps only about 8 bits of mantissa, so as activations and logits grow in magnitude their absolute rounding error grows with them, and when logits become nearly tied, small absolute errors change which low-probability tokens are sampled and how likely they look.
2. **Stage 2: kernel-driven error amplification.** These initial, tiny `bfloat16` quantization errors are fed into the different kernel implementations of vLLM and FSDP. The different calculation orders act as non-linear amplifiers, so small initial deviations snowball into large differences in the final logits.

This creates a **vicious feedback loop**: mismatch leads to biased and noisy gradients, which may push the parameters further into a numerically sensitive region, which in turn worsens the mismatch at the next iteration, until the system collapses.

---

## 4. The Fix: Sequence-Level Correction and Masking {#4-attempts-to-alleviate-the-training-inference-mismatch}

<span id="42-what-helped"></span>

The mismatch follows from the speed-versus-consistency trade-off of [§2](#2-a-fundamental-conflict-the-growing-gap-between-inference-and-training), and the engine and hardware settings that reduce it are specific to one stack ([§5](#other-settings)). The fix that carries over is algorithmic: treat the sampler as a behavior policy and correct for it in the loss. This section takes four steps: why the correction has to act on the whole sequence ([§4.1](#distribution-correction)), what token-level and sequence-level corrections do in our runs ([§4.2](#experimental-validation)), our method, MIS ([§4.3](#mis)), and its length-normalized form, Geo-MIS, with its link to Trust Region Masking ([§4.4](#geo-mis)).

### 4.1 Why the Correction Must Act on the Whole Sequence {#distribution-correction}

**A per-token ratio corrects how likely each token is given its prefix, but not how likely the prefix is. Only the ratio of the whole sequence corrects both.**

The training-inference mismatch turns our otherwise on-policy RL problem into an off-policy one: the policy that generates rollouts (the *behavior policy*, {{< math >}}$\pvllm${{< /math >}}) differs from the policy being trained (the *target policy*, {{< math >}}$\pfsdp${{< /math >}}). One theoretically sound way to correct for this distribution shift is **importance sampling (IS)**. However, the specific form of IS is critical for keeping the gradient unbiased and the training stable.

Inspired by [Yao et al. (2025)](https://fengyao.notion.site/off-policy-rl), who first highlighted this implicit off-policy issue caused by the training-inference mismatch, we analyze two forms of IS: the theoretically sound **sequence-level IS** and the common but biased **token-level IS** approximation.

{{< stepfig "rlc-tokseq" >}}

#### The principled estimator: sequence-level IS

The correct, unbiased policy gradient estimator applies a single importance ratio to the entire generated sequence (trajectory) {{< math >}}$y${{< /math >}}. This re-weights the expectation from the behavior policy to the target policy and yields the true gradient of the objective {{< math >}}$J(\theta)${{< /math >}}.

Let's derive the sequence-level IS estimator {{< math >}}$g_{\mathrm{seq}}(\theta)${{< /math >}} step by step.

1. The objective is to maximize the expected reward under the target **FSDP policy**:

   {{< math >}}
   $$
   J(\theta) = \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pfsdp(\cdot|x)}[R(x,y)]
   $$
   {{< /math >}}

2. The true policy gradient is therefore:

   {{< math >}}
   $$
   g(\theta) = \nabla_{\theta} J(\theta) = \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pfsdp(\cdot|x)}\left[R(x,y)\, \nabla_{\theta} \log \pfsdp(y|x)\right]
   $$
   {{< /math >}}

3. Since we can only sample from the **vLLM policy**, we use importance sampling to change the distribution of the expectation:

   {{< math >}}
   $$
   g_{\mathrm{seq}}(\theta) = \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pvllm(\cdot|x)}\left[ \frac{\pfsdp(y|x)}{\pvllm(y|x)} \cdot R(x,y) \cdot \nabla_{\theta} \log \pfsdp(y|x) \right]
   $$
   {{< /math >}}

This is essentially the off-policy **REINFORCE** algorithm ([Williams, 1992](https://doi.org/10.1007/BF00992696)). In expectation, the estimator equals the standard advantage form of the policy gradient: the importance ratio exactly corrects the expectation, revealing the true on-policy gradient underneath, which can then be refined.

<details class="spoiler">
<summary>The detailed derivation</summary>

**Step 1: convert the expectation to its on-policy form.** The IS estimator takes its expectation with respect to the behavior policy {{< math >}}$\pvllm${{< /math >}}. Writing the expectation out as an integral, the behavior policy density cancels:

{{< math >}}
$$
\begin{aligned}
g_{\mathrm{seq}}(\theta) &= \int \left( \frac{\pfsdp(y|x)}{\pvllm(y|x)} \cdot R(x,y) \cdot \nabla_{\theta} \log \pfsdp(y|x) \right) \pvllm(y|x) \,dy \\
&= \int \left( R(x,y) \cdot \nabla_{\theta} \log \pfsdp(y|x) \right) \pfsdp(y|x) \,dy.
\end{aligned}
$$
{{< /math >}}

This leaves the true on-policy gradient, with the expectation now over the target policy {{< math >}}$\pfsdp${{< /math >}}, so {{< math >}}$g_{\mathrm{seq}}(\theta)${{< /math >}} is an unbiased estimator of the true policy gradient:

{{< math >}}
$$
g(\theta) = \mathbb{E}_{y \sim \pfsdp} \left[ R(x,y) \cdot \nabla_{\theta} \log \pfsdp(y|x) \right].
$$
{{< /math >}}

**Step 2: decompose into timesteps and apply causality.** Working with the on-policy expression, we write the reward as a sum of per-step rewards, {{< math >}}$R(x,y) = \sum_{k=0}^{|y|-1} r(s_k, a_k)${{< /math >}} (for an outcome reward, {{< math >}}$r${{< /math >}} is nonzero only at the last token), expand the trajectory-level terms and apply **causality** (an action at step {{< math >}}$t${{< /math >}} only affects future rewards). This replaces the total reward {{< math >}}$R(x,y)${{< /math >}} with the **return-to-go** {{< math >}}$G_t = \sum_{k=t}^{|y|-1} r(s_k, a_k)${{< /math >}}. Writing {{< math >}}$\pi_\theta${{< /math >}} for {{< math >}}$\pfsdp${{< /math >}} in the rest of this derivation:

{{< math >}}
$$
g(\theta) = \mathbb{E}_{y \sim \pi_{\theta}} \left[ \sum_{t=0}^{|y|-1} G_t \cdot \nabla_{\theta} \log \pi_{\theta}(a_t|s_t) \right].
$$
{{< /math >}}

**Step 3: introduce the advantage function.** Replacing {{< math >}}$G_t${{< /math >}} by its conditional expectation given {{< math >}}$(s_t, a_t)${{< /math >}}, the action value {{< math >}}$Q^{\pi_{\theta}}(s_t, a_t)${{< /math >}}, leaves the gradient unchanged (tower property). For variance reduction, we then subtract a state-dependent **baseline**, the value function {{< math >}}$V^{\pi_{\theta}}(s_t)${{< /math >}}, whose contribution is zero in expectation. This gives the **advantage** {{< math >}}$A^{\pi_{\theta}}(s_t, a_t) =${{< /math >}} {{< math >}}$Q^{\pi_{\theta}}(s_t, a_t) - V^{\pi_{\theta}}(s_t)${{< /math >}}:

{{< math >}}
$$
g(\theta) = \mathbb{E}_{y \sim \pi_{\theta}} \left[ \sum_{t=0}^{|y|-1} A^{\pi_{\theta}}(s_t, a_t) \cdot \nabla_{\theta} \log \pi_{\theta}(a_t|s_t) \right].
$$
{{< /math >}}

**Step 4: convert to a state-level expectation.** Finally, rewrite the expectation over trajectories as an expectation over the state-visitation distribution {{< math >}}$d_{\pi_{\theta}}${{< /math >}} induced by the target policy.

*(a) Linearity of expectation.* Swap the expectation and the sum. We can extend the sum to infinity because terms for timesteps {{< math >}}$t${{< /math >}} beyond a trajectory's length {{< math >}}$|y|${{< /math >}} are zero:

{{< math >}}
$$
g(\theta) = \sum_{t=0}^{\infty} \mathbb{E}_{y \sim \pi_{\theta}} \left[ A^{\pi_{\theta}}(s_t, a_t) \cdot \nabla_{\theta} \log \pi_{\theta}(a_t|s_t) \right].
$$
{{< /math >}}

*(b) Expand the expectation into an explicit sum* over every state {{< math >}}$s${{< /math >}} and action {{< math >}}$a${{< /math >}}, weighted by their joint probability {{< math >}}$P(s_t=s, a_t=a) =${{< /math >}} {{< math >}}$P(s_t=s) \cdot \pi_{\theta}(a|s)${{< /math >}}:

{{< math >}}
$$
g(\theta) = \sum_{t=0}^{\infty} \sum_{s \in \mathcal{S}} P(s_t=s \mid \pi_{\theta}) \sum_{a \in \mathcal{A}} \pi_{\theta}(a|s) \cdot \left[ A^{\pi_{\theta}}(s, a) \cdot \nabla_{\theta} \log \pi_{\theta}(a|s) \right].
$$
{{< /math >}}

*(c) Introduce the state occupancy measure.* Group all terms related to state {{< math >}}$s${{< /math >}}:

{{< math >}}
$$
g(\theta) = \sum_{s \in \mathcal{S}} \left( \sum_{t=0}^{\infty} P(s_t=s \mid \pi_{\theta}) \right) \left( \sum_{a \in \mathcal{A}} \pi_{\theta}(a|s) \cdot \left[ \dots \right] \right).
$$
{{< /math >}}

The first factor, {{< math >}}$\sum_{t=0}^{\infty} P(s_t=s)${{< /math >}}, is the **state occupancy measure** {{< math >}}$d_{\pi_{\theta}}(s)${{< /math >}}. The second is an expectation over actions, {{< math >}}$\mathbb{E}_{a \sim \pi_{\theta}(\cdot|s)}[\dots]${{< /math >}}:

{{< math >}}
$$
g(\theta) = \sum_{s \in \mathcal{S}} d_{\pi_{\theta}}(s) \cdot \mathbb{E}_{a \sim \pi_{\theta}(\cdot|s)} \left[ A^{\pi_{\theta}}(s, a) \cdot \nabla_{\theta} \log \pi_{\theta}(a|s) \right].
$$
{{< /math >}}

*(d) The final form.* A sum over states weighted by the occupancy measure is an expectation over {{< math >}}$d_{\pi_{\theta}}${{< /math >}}:

{{< math >}}
$$
g(\theta) = \mathbb{E}_{s \sim d_{\pi_{\theta}}} \left[ \mathbb{E}_{a \sim \pi_{\theta}(\cdot|s)} \left[ A^{\pi_{\theta}}(s,a) \cdot \nabla_{\theta} \log \pi_{\theta}(a|s) \right] \right].
$$
{{< /math >}}

</details>

This derivation leads to the advantage form of the policy gradient:

{{< math >}}
$$
g_{\mathrm{seq}}(\theta) = \mathbb{E}_{s \sim d_{\pfsdp}}\, \mathbb{E}_{a \sim \pfsdp(\cdot|s)} \left[ A^{\pfsdp}(s,a) \cdot \nabla_{\theta} \log \pfsdp(a|s) \right].
$$
{{< /math >}}

Here {{< math >}}$s=(x,y_{\lt t})${{< /math >}} is the state (prefix), {{< math >}}$a=y_t${{< /math >}} is the action (token), and **{{< math >}}$d_{\pfsdp}${{< /math >}} is the state occupancy measure** under the target FSDP policy: the probability that a rollout under that policy reaches the prefix {{< math >}}$s${{< /math >}}. It is the probability of the prompt times the probabilities the policy gives to the model-generated tokens of the prefix.

<details class="spoiler">
<summary>The formal definition of the occupancy measure</summary>

The occupancy measure of a policy {{< math >}}$\pi${{< /math >}} is the expected number of times a state {{< math >}}$s${{< /math >}} is visited when following {{< math >}}$\pi${{< /math >}}:

{{< math >}}
$$
d_{\pi}(s) := \mathbb{E}_{x' \sim \mathcal{D},\, y' \sim \pi(\cdot|x')} \left[ \sum_{t'=0}^{|y'|-1} \mathbb{I}\{ (x', y'_{\lt t'}) = s \} \right].
$$
{{< /math >}}

For a prefix {{< math >}}$s=(x,y_{\lt t})${{< /math >}} this is the probability of reaching it,

{{< math >}}
$$
d_{\pi}(s) = \mathcal{D}(x) \cdot \prod_{k=0}^{t-1} \pi(y_k|x,y_{\lt k}).
$$
{{< /math >}}

Tool-response tokens come from the environment, not the policy, so only model-generated tokens enter this product.

</details>

This estimator is **unbiased**: {{< math >}}$g_{\mathrm{seq}}(\theta) = g(\theta)${{< /math >}}. For numerical stability, **truncated importance sampling (TIS)** clips the sequence-level ratio {{< math >}}$\rho(y|x) = \pfsdp(y|x)/\pvllm(y|x)${{< /math >}} at a constant {{< math >}}$C${{< /math >}}. Clipping bounds the variance but brings back a controllable bias: the bias-variance trade-off analyzed in [Part 2](/post/rl-collapse-part2/).

#### A common biased estimator: token-level IS

A common heuristic, often inspired by algorithms like PPO and used in [Yao et al. (2025)](https://fengyao.notion.site/off-policy-rl), applies a per-token importance ratio. It typically has lower variance than the sequence-level ratio, but for autoregressive models it is a **biased estimator**: it corrects each token's distribution but not the distribution of prefixes.

The estimator applies the importance ratio inside the sum over timesteps, so {{< math >}}$g_{\mathrm{tok}}(\theta)${{< /math >}} is defined as

{{< math >}}
$$
\mathbb{E}_{x \sim \mathcal{D},\, y \sim \pvllm(\cdot|x)}\left[ R(x,y) \cdot \sum_{t=0}^{|y|-1} \frac{\pfsdp(y_t|x,y_{\lt t})}{\pvllm(y_t|x,y_{\lt t})} \cdot \nabla_{\theta} \log \pfsdp(y_t|x,y_{\lt t}) \right]
$$
{{< /math >}}

Grouping the terms by state and subtracting a baseline puts it in the same form as the true gradient above:

{{< math >}}
$$
g_{\mathrm{tok}}(\theta) = \mathbb{E}_{s \sim d_{\pvllm}}\, \mathbb{E}_{a \sim \pfsdp(\cdot|s)} \left[ A^{\pvllm}(s,a) \cdot \nabla_{\theta} \log \pfsdp(a|s) \right]
$$
{{< /math >}}

<details class="spoiler">
<summary>The two steps in between</summary>

1. Grouping the terms by state turns the expectation over trajectories into an expectation over the states visited under the **vLLM policy**. The return {{< math >}}$R(x,y)${{< /math >}} of a trajectory sampled by {{< math >}}$\pvllm${{< /math >}} is a Monte Carlo estimate of the state-action value {{< math >}}$Q^{\pvllm}(s,a)${{< /math >}}, so

   {{< math >}}
   $$
   g_{\mathrm{tok}}(\theta) = \mathbb{E}_{s \sim d_{\pvllm}}\, \mathbb{E}_{a \sim \pvllm(\cdot|s)} \left[ \frac{\pfsdp(a|s)}{\pvllm(a|s)} \cdot Q^{\pvllm}(s,a) \cdot \nabla_{\theta} \log \pfsdp(a|s) \right]
   $$
   {{< /math >}}

2. Subtracting the baseline {{< math >}}$V^{\pvllm}(s)${{< /math >}} leaves this unchanged, because

   {{< math >}}
   $$
   \mathbb{E}_{a \sim \pvllm(\cdot|s)}\Big[\frac{\pfsdp(a|s)}{\pvllm(a|s)} \nabla_{\theta} \log \pfsdp(a|s)\Big] = \mathbb{E}_{a \sim \pfsdp(\cdot|s)}\big[\nabla_{\theta} \log \pfsdp(a|s)\big] = 0,
   $$
   {{< /math >}}

   and it turns {{< math >}}$Q^{\pvllm}${{< /math >}} into the advantage {{< math >}}$A^{\pvllm}${{< /math >}}. Using the per-token ratio to change the expectation over actions from {{< math >}}$\pvllm${{< /math >}} to {{< math >}}$\pfsdp${{< /math >}} gives the form above.

</details>

This form reveals the gradient bias of token-level IS. {{< math >}}$g_\text{tok}(\theta)${{< /math >}} approximates the true gradient well when {{< math >}}$\pvllm${{< /math >}} stays within a trust region of {{< math >}}$\pfsdp${{< /math >}}, so that {{< math >}}$d_{\pvllm}\approx d_{\pfsdp}${{< /math >}} and {{< math >}}$A^{\pvllm}\approx A^{\pfsdp}${{< /math >}}. [Part 1](/post/rl-collapse-part1/) bounds the bias by {{< math >}}$O(T^2 D_{TV}^{\max})${{< /math >}}; it becomes harmful when the mismatch is large or uncontrolled.

#### The source of bias in token-level IS

Comparing {{< math >}}$g_{\mathrm{tok}}(\theta)${{< /math >}} with the true gradient {{< math >}}$g_{\mathrm{seq}}(\theta)${{< /math >}} reveals two distinct errors that make the token-level estimator biased.

**Source 1: state occupancy mismatch.** A sound off-policy correction must account for two distribution shifts: the action probabilities and the state visitation probabilities. The token-level method only corrects the first.

- **True gradient ({{< math >}}$g_{\mathrm{seq}}${{< /math >}}):** the expectation is over states visited under the **correct target FSDP distribution**, {{< math >}}$\mathbb{E}_{s \sim d_{\pfsdp}}${{< /math >}}.
- **Flawed gradient ({{< math >}}$g_{\mathrm{tok}}${{< /math >}}):** the expectation is over states visited under the **incorrect behavior vLLM distribution**, {{< math >}}$\mathbb{E}_{s \sim d_{\pvllm}}${{< /math >}}.

This implicitly assumes the state occupancy ratio {{< math >}}$d_{\pfsdp}(s) / d_{\pvllm}(s)${{< /math >}} is 1. For a prefix {{< math >}}$s=(x,y_{\lt t})${{< /math >}} that ratio is a product of per-token ratios,

{{< math >}}
$$
\frac{d_{\pfsdp}(s)}{d_{\pvllm}(s)} = \prod_{k=0}^{t-1} \frac{\pfsdp(y_k|x,y_{\lt k})}{\pvllm(y_k|x,y_{\lt k})},
$$
{{< /math >}}

so it compounds with the length of the prefix and drifts away from 1 even when every per-token ratio is close to 1. By ignoring it, {{< math >}}$g_{\mathrm{tok}}(\theta)${{< /math >}} introduces a bias that grows with the length of the response.

**Source 2: mismatched reward signal.** The token-level gradient weights the update with a reward signal from the **wrong policy**.

- **True gradient ({{< math >}}$g_{\mathrm{seq}}${{< /math >}}):** the update is scaled by the advantage of the **target FSDP policy**, {{< math >}}$A^{\pfsdp}${{< /math >}}, the expected future reward under that policy.
- **Flawed gradient ({{< math >}}$g_{\mathrm{tok}}${{< /math >}}):** the update is scaled by the advantage of the **behavior vLLM policy**, {{< math >}}$A^{\pvllm}${{< /math >}}.

The gradient for the target policy is scaled by a reward signal that belongs to the behavior policy. Because both the state distribution and the reward signal are mismatched, the token-level gradient is biased, and the bias grows with the mismatch and with the length of the response.

{{% callout note %}}
This theory suggests that, despite the potentially lower variance of token-level methods, their gradient bias persists and can lead to training instability. **Our tool-use experiments bear this out.** Parts [1](/post/rl-collapse-part1/) and [2](/post/rl-collapse-part2/) of the series give the detailed bias and variance analysis of token-level and sequence-level methods.
{{% /callout %}}

### 4.2 Token-Level and Sequence-Level Corrections in Practice {#experimental-validation}

**Importance sampling of either kind prevents the gradient explosion. In multi-turn tool use the token-level run still collapses, while the sequence-level run survives but plateaus.**

Our analysis predicts that token-level corrections become unstable when the mismatch is large, while sequence-level corrections are more robust. Pure sequence-level IS has exponential variance, so we test truncated and masked variants. With the per-token ratio {{< math >}}$\rho_t =${{< /math >}} {{< math >}}$\pfsdp(y_t|x,y_{\lt t})/\pvllm(y_t|x,y_{\lt t})${{< /math >}} and the sequence ratio {{< math >}}$\rho = \prod_t \rho_t${{< /math >}}, both over model-generated tokens:

- **Token-TIS** weights each token by {{< math >}}$\min(\rho_t, C)${{< /math >}}.
- **Seq-TIS** weights the whole sequence by {{< math >}}$\min(\rho, C)${{< /math >}}.
- **Token-MIS** weights each token by {{< math >}}$\rho_t\,\mathbb{I}\{\rho_t \le C\}${{< /math >}}.
- **Seq-MIS**, or simply MIS, weights the whole sequence by {{< math >}}$\rho\,\mathbb{I}\{\rho \le C\}${{< /math >}} ([§4.3](#mis)).

**IS prevents gradient explosion, but token-level IS still fails.** We resumed a crashed L20 experiment from step 200 with token-level TIS and with sequence-level TIS ({{< math >}}$C=2${{< /math >}}). As Figure 11 shows, both prevent the gradient explosion seen in the vanilla run, but **the token-level TIS run still collapses**: its reward falls to zero by about step 240, without a gradient spike. The sequence-level TIS run stays stable through the end of the run, about step 280; Figure 13 follows Seq-TIS for longer.

<figure class="post-fig">
<img src="/figs/img/rl-collapse/fig14-token-vs-seq-tis.png" alt="The vanilla run's gradient norm explodes and its reward crashes by step 280; token-level TIS from step 200 avoids the explosion but its reward drops to zero around step 240, while sequence-level TIS stays near 0.5.">
<figcaption><strong>Figure 11.</strong> Ablation on TIS strategies. The vanilla on-policy experiment without IS (red) crashes after 200 steps with a gradient explosion. We resumed training from step 200 with token-level TIS (orange) and sequence-level TIS (purple). <strong>Left:</strong> gradient norm. <strong>Right:</strong> training rewards.</figcaption>
</figure>

**Token-level TIS in reasoning RL.** Token-level TIS fails in our complex TIR experiments, but it can prevent collapse in simpler reasoning RL, where the mismatch is smaller. In on-policy GRPO and RLOO experiments (Figure 12), token-level TIS prevented the collapse and let training run much longer, but it brought little or no gain in peak test accuracy: none for GRPO or for Qwen3-14B with RLOO, and a small gain for Qwen3-8B with RLOO. Accuracy plateaued and tended to sag late in training, likely because of the remaining gradient bias.

<figure class="post-fig">
<img src="/figs/img/rl-collapse/fig15a-grpo-token-tis.jpg" alt="On-policy GRPO in reasoning RL: without IS (red) the gradient norm explodes and AIME24 accuracy falls to zero by step 620, while token-level TIS (blue) keeps training stable to step 1400 but stays at the same peak of about 0.5.">
<figcaption><strong>Figure 12 (a).</strong> Gradient norm (<strong>left</strong>) and AIME24 accuracy (<strong>right</strong>) of on-policy GRPO on H100 GPUs without IS (red) and with token-level TIS ({{< math >}}$C=2${{< /math >}}, blue), in reasoning RL. Token-level TIS prevents training collapse but does not improve test performance.</figcaption>
</figure>

<figure class="post-fig">
<img src="/figs/img/rl-collapse/fig15b-rloo-token-tis.jpg" alt="On-policy RLOO with Qwen3-8B-Base and Qwen3-14B-Base: both runs without IS spike in gradient norm and lose test accuracy after about 400 steps, while token-level TIS keeps both models training to step 680 to 870, with accuracy that plateaus and sags slightly late.">
<figcaption><strong>Figure 12 (b).</strong> Gradient norm (<strong>left</strong>) and average accuracy on AIME24/AIME25 (<strong>right</strong>) of on-policy RLOO on H100 GPUs, for Qwen3-8B-Base and Qwen3-14B-Base, without IS and with token-level TIS ({{< math >}}$C=2${{< /math >}}), in reasoning RL. Without IS, both models collapse after about 400 steps. Token-level TIS keeps both training much longer; peak accuracy is about the same for 14B and slightly higher for 8B, and it sags late in training.</figcaption>
</figure>

**TIS prolongs training but can still be unstable.** We ran an on-policy TIR experiment from scratch on L20 GPUs with sequence-level TIS ({{< math >}}$C=2${{< /math >}}). As Figure 13 shows, it prevented a complete collapse, but the reward kept fluctuating after reaching a plateau, and test performance did not exceed the peak the vanilla run reached before it crashed.

<figure class="post-fig">
<img src="/figs/img/rl-collapse/fig16-seq-tis-scratch.png" alt="Training rewards and test accuracy for vanilla and sequence-level TIS from scratch: TIS avoids the crash but plateaus and fluctuates, without beating the vanilla peak.">
<figcaption><strong>Figure 13.</strong> Ablation on TIS strategies (continued). The vanilla on-policy experiment without IS (red) crashes after 200 steps. We reran it from scratch with the same configuration plus sequence-level TIS with {{< math >}}$C=2${{< /math >}} (purple). <strong>Left:</strong> training rewards. <strong>Right:</strong> test accuracy.</figcaption>
</figure>

### 4.3 Our Method: Masked Importance Sampling (MIS) {#mis}

To improve on TIS, we propose **masked importance sampling (MIS)**: keep the sequence-level ratio, but mask the policy loss of every sequence whose ratio exceeds the threshold {{< math >}}$C${{< /math >}}, i.e. {{< math >}}$\rho(y|x) \gets \rho(y|x)\, \mathbb{I}\{\rho(y|x) \le C\}${{< /math >}}. TIS keeps such a sequence with its weight capped at {{< math >}}$C${{< /math >}}; MIS removes it from the update. The result is an upper-sided hard trust region at the level of the sequence.

**Which tail the mask cuts.** The mask removes sequences on which the trainer is far *more* likely than the sampler ({{< math >}}$\rho > C${{< /math >}}). The pathology of [§3.2](#low-probability-tokens) is the opposite tail, {{< math >}}$\rho \ll 1${{< /math >}}, where the trainer assigns far lower probability; there the importance weight itself shrinks the sequence's contribution toward zero. The geometric form, [Geo-MIS](#geo-mis), is two-sided and length-normalized: it keeps a sequence only when {{< math >}}$C_{\text{low}} \le \rho^{1/T} \le C_{\text{high}}${{< /math >}}, which removes the low tail outright as well.

{{< stepfig "rlc-weights" >}}

**MIS in practice.** The ratio is a product over hundreds of tokens, so compute it in log space, over model-generated tokens only (tool-response tokens are excluded, as in `vllm-kl`):

{{< math >}}
$$
\begin{aligned}
\log\rho(y|x) &= \sum_{t \in \mathcal{T}_{\mathcal{M}}(y)} \left[\log\pfsdp(y_t|x,y_{\lt t}) - \log\pvllm(y_t|x,y_{\lt t})\right], \\
\text{keep } y &\iff \log\rho(y|x) \le \log C.
\end{aligned}
$$
{{< /math >}}

The trainer's log-probabilities come from the forward pass before the update (`old_log_probs`), so the weight carries no gradient. A minimal sketch, in the style of the K3 snippet above:

```python
log_ratio = old_log_probs - rollout_log_probs              # log pi_fsdp - log pi_vllm, per token
seq_log_ratio = (log_ratio * response_mask).sum(dim=-1)     # log rho(y|x), model tokens only
keep = seq_log_ratio <= math.log(C)                         # MIS: mask sequences with rho > C
seq_weight = torch.exp(seq_log_ratio.clamp(max=math.log(C))) * keep   # in [0, C]; masked -> 0
pg_loss = pg_loss * seq_weight.unsqueeze(-1)                # scale each sequence's per-token loss
```

**Results.** As Figure 14 shows, on L20 GPUs MIS trained for more than 700 steps where the uncorrected run collapsed near step 280. It raised the peak training reward from about 0.56 to 0.64 and matched the uncorrected run's peak test accuracy (0.70 vs. 0.68), well above Seq-TIS (about 0.61). It was not free of instability: its reward dipped to about 0.37 near step 440 before recovering. The length-invariant geometric form is given in [§4.4](#geo-mis) and developed further in [Part 3](/post/rl-collapse-part3/).

<figure class="post-fig">
<img src="/figs/img/rl-collapse/fig17-mis.png" alt="Training rewards and test accuracy: vanilla collapses near step 280 and sequence-level TIS ends near step 410, while sequence-level MIS trains to step 700 with the highest reward (about 0.64) and test accuracy reaching about 0.70.">
<figcaption><strong>Figure 14.</strong> On-policy GRPO with sequence-level MIS (orange), TIS (purple), and the vanilla version without IS (red), all on L20 GPUs. Both MIS and TIS extend training beyond the vanilla collapse; TIS ends lower and drops at the end of its run, and MIS dips near step 440 before recovering. MIS improves on TIS in both training reward and test accuracy, exceeds the vanilla run's peak reward, and matches its peak test accuracy. <strong>Left:</strong> training rewards. <strong>Right:</strong> test accuracy.</figcaption>
</figure>

**Token-level MIS vs. sequence-level MIS.** Finally, we compared token-level and sequence-level MIS. Figure 15 shows that both keep the gradient norm small, but **the token-level MIS run still collapses**: its reward falls from about 0.4 to below 0.2 by step 160, while the sequence-level run keeps improving through step 400. In our long-horizon tool-use runs, only the sequence-level correction was reliable.

<figure class="post-fig">
<img src="/figs/img/rl-collapse/fig18-token-vs-seq-mis.jpg" alt="Gradient norm and rewards for token-level and sequence-level MIS: the token-level run collapses while the sequence-level run stays stable.">
<figcaption><strong>Figure 15.</strong> On-policy GRPO with token-level MIS (purple) and sequence-level MIS (orange), on L20 GPUs with the same configuration. <strong>Left:</strong> gradient norm. <strong>Right:</strong> training rewards.</figcaption>
</figure>

### 4.4 Geo-MIS and Trust Region Masking {#geo-mis}

**Geo-MIS masks on the per-token geometric mean of the ratio. This makes the trust region independent of length and turns it into a bound on an estimate of the mean per-token KL. Trust Region Masking computes that KL exactly and adds a maximum.** This section is analysis: the experiments above use the product form of [§4.3](#mis).

**Why the product form needs a length correction.** {{< math >}}$\log\rho(y|x) = \sum_t \log\rho_t${{< /math >}} is a sum over the {{< math >}}$T${{< /math >}} model-generated tokens of the response, so it scales with length. A per-token log-ratio of {{< math >}}$0.01${{< /math >}} gives {{< math >}}$\rho = e^{1} \approx 2.7${{< /math >}} at {{< math >}}$T = 100${{< /math >}} and {{< math >}}$\rho = e^{10} \approx 22{,}000${{< /math >}} at {{< math >}}$T = 1000${{< /math >}}. A fixed threshold {{< math >}}$C${{< /math >}} therefore tests long responses far more strictly than short ones, whatever their per-token quality.

**Geo-MIS.** Take the geometric mean of the per-token ratios and mask on both sides:

{{< math >}}
$$
\begin{gathered}
\rho_{\mathrm{geo}}(y|x) = \rho(y|x)^{1/T} = \exp\Big(\frac{1}{T}\sum_{t \in \mathcal{T}_{\mathcal{M}}(y)} \log\rho_t\Big), \qquad T = |\mathcal{T}_{\mathcal{M}}(y)|, \\
\rho_{\mathrm{geo}}(y|x) \gets \rho_{\mathrm{geo}}(y|x)\,\mathbb{I}\{C_{\text{low}} \le \rho_{\mathrm{geo}}(y|x) \le C_{\text{high}}\}.
\end{gathered}
$$
{{< /math >}}

If every token has ratio {{< math >}}$r${{< /math >}}, then {{< math >}}$\rho_{\mathrm{geo}} = r${{< /math >}} at any length. This is the form released in VeRL 0.6.0 (`rollout_is_level="geometric"`, `rollout_is_mode="mask"`), where {{< math >}}$C_{\text{low}}${{< /math >}} defaults to {{< math >}}$1/C_{\text{high}}${{< /math >}}. [Part 3](/post/rl-collapse-part3/) calls the mask Geo-Mask and current VeRL calls it Geo-RS; both also apply it without a weight, or with per-token weights in place of {{< math >}}$\rho_{\mathrm{geo}}${{< /math >}}. With `seq_log_ratio` from the MIS sketch above, the lines after it become:

```python
n_tokens = response_mask.sum(dim=-1).clamp(min=1)            # T: model tokens per response
geo_log_ratio = seq_log_ratio / n_tokens                     # log rho_geo: mean per-token log-ratio
keep = (geo_log_ratio >= math.log(C_low)) & (geo_log_ratio <= math.log(C_high))
seq_weight = torch.exp(geo_log_ratio) * keep                 # Geo-MIS weight; masked -> 0
pg_loss = pg_loss * seq_weight.unsqueeze(-1)
```

**Step 1: {{< math >}}$-\log\rho_{\mathrm{geo}}${{< /math >}} is a one-sample estimate of the mean KL.** Write {{< math >}}$D_t =${{< /math >}} {{< math >}}$\mathrm{KL}\big(\pvllm(\cdot|x,y_{\lt t})\,\|${{< /math >}} {{< math >}}$\pfsdp(\cdot|x,y_{\lt t})\big)${{< /math >}} for the KL divergence from the sampler to the trainer at the context of token {{< math >}}$t${{< /math >}}. The token {{< math >}}$y_t${{< /math >}} was drawn from {{< math >}}$\pvllm${{< /math >}}, so by the definition of the KL divergence

{{< math >}}
$$
\mathbb{E}_{y_t \sim \pvllm(\cdot|x,y_{\lt t})}\big[-\log\rho_t\big] = \sum_{v} \pvllm(v|x,y_{\lt t}) \log\frac{\pvllm(v|x,y_{\lt t})}{\pfsdp(v|x,y_{\lt t})} = D_t \;\ge\; 0 .
$$
{{< /math >}}

Averaging over the response splits {{< math >}}$-\log\rho_{\mathrm{geo}}${{< /math >}} into the mean KL along it and zero-mean sampling noise:

{{< math >}}
$$
-\log\rho_{\mathrm{geo}}(y|x) = \frac{1}{T}\sum_{t}\big(-\log\rho_t\big) = \underbrace{\frac{1}{T}\sum_{t} D_t}_{\text{mean KL along } y} + \underbrace{\frac{1}{T}\sum_{t}\xi_t}_{\text{sampling noise}},
$$
{{< /math >}}

where {{< math >}}$\xi_t = -\log\rho_t - D_t${{< /math >}} has mean zero given the context {{< math >}}$(x, y_{\lt t})${{< /math >}}.

This is `vllm-kl` in its plain log-ratio form ([§3.1](#31-measuring-the-mismatch-the-vllm-kl-metric)), computed on one response instead of a batch. The product form is the same statement about the total: the expectation of {{< math >}}$-\log\rho(y|x)${{< /math >}} under the sampler is the KL divergence between the two engines' distributions over whole responses, which by the chain rule is the sum of the expected token-level KLs and so grows with {{< math >}}$T${{< /math >}}. MIS thresholds an estimate of that total, and Geo-MIS an estimate of its per-token mean.

**Step 2: the band is a trust region on that estimate.** Taking logarithms,

{{< math >}}
$$
C_{\text{low}} \le \rho_{\mathrm{geo}} \le C_{\text{high}} \iff -\log C_{\text{high}} \;\le\; -\log\rho_{\mathrm{geo}} \;\le\; \log\frac{1}{C_{\text{low}}} .
$$
{{< /math >}}

The right-hand inequality caps the estimated mean KL at {{< math >}}$\log(1/C_{\text{low}})${{< /math >}} nats per token. It is the side that removes the low-probability tail of [§3.2](#low-probability-tokens). The left-hand inequality exists because a one-sample estimate of a non-negative quantity can be negative; it removes responses on which the trainer is far more confident than the sampler, the tail that the product-form mask {{< math >}}$\rho \le C${{< /math >}} cuts. With {{< math >}}$C_{\text{low}} = 1/C_{\text{high}}${{< /math >}} the band reads {{< math >}}$|\log\rho_{\mathrm{geo}}| \le \log C_{\text{high}}${{< /math >}}. For example, {{< math >}}$C_{\text{high}} = 1.001${{< /math >}} allows about {{< math >}}$10^{-3}${{< /math >}} nats per token, the scale of `vllm-kl` in our H20 batches (Group 1 in [§3.2](#low-probability-tokens)).

**Step 3: Trust Region Masking computes the same quantity exactly, and adds a maximum.** [Trust Region Masking](https://arxiv.org/abs/2512.23075) (TRM, ICML 2026) stores the sampler's logits at rollout time, so {{< math >}}$D_t${{< /math >}} is computed over the full vocabulary in the training forward pass instead of being estimated from the one sampled token. It masks a sequence with two criteria:

{{< math >}}
$$
\frac{1}{T}\sum_{t} D_t \le \delta_{\mathrm{avg}} \quad \text{(mean KL)}, \qquad\qquad \max_{t} D_t \le \delta \quad \text{(max KL)}.
$$
{{< /math >}}

- **The mean-KL criterion is the exact version of Geo-MIS.** The noise term of Step 1 is gone and the statistic is non-negative, so a single threshold replaces the two-sided band, with {{< math >}}$\delta_{\mathrm{avg}}${{< /math >}} in the role of {{< math >}}$\log(1/C_{\text{low}})${{< /math >}}. Like Geo-MIS, it does not reject long responses more often than short ones.
- **The max-KL criterion is what the guarantee needs.** The paper's error bounds depend on the largest token-level KL over all contexts, and a mean cannot control it: a response whose mean KL is below {{< math >}}$\delta_{\mathrm{avg}}${{< /math >}} can still contain one token with a KL as large as {{< math >}}$T\delta_{\mathrm{avg}}${{< /math >}}. A maximum, in turn, rejects long responses more often. In practice the paper combines the two: the maximum catches single-token outliers and the mean limits accumulated drift.
- **Without stored logits, TRM falls back to sample-based estimators.** For the mean it uses {{< math >}}$k_3(\rho_t) = \rho_t - 1 - \log\rho_t${{< /math >}}, which is non-negative and unbiased for {{< math >}}$D_t${{< /math >}} (the estimator behind `vllm-kl` in §3.1), and for the maximum a symmetric function of {{< math >}}$\log\rho_t${{< /math >}}. Geo-MIS is the mean criterion with the {{< math >}}$k_1${{< /math >}} estimator, {{< math >}}$-\log\rho_t${{< /math >}}, in place of {{< math >}}$k_3${{< /math >}}.
- **The weights differ.** Inside an accepted sequence TRM weights each token by its own ratio {{< math >}}$\rho_t${{< /math >}}, where Geo-MIS weights the whole sequence by {{< math >}}$\rho_{\mathrm{geo}}${{< /math >}}.

{{< stepfig "rlc-kl" >}}

| Method | Statistic for one response | Kept when | Weight if kept |
|---|---|---|---|
| MIS (this post) | {{< math >}}$\log\rho = \sum_t \log\rho_t${{< /math >}}; {{< math >}}$-\log\rho${{< /math >}} estimates the total KL | {{< math >}}$\rho \le C${{< /math >}} | {{< math >}}$\rho${{< /math >}} |
| Geo-MIS (VeRL 0.6.0) | {{< math >}}$\log\rho_{\mathrm{geo}} = \frac{1}{T}\sum_t \log\rho_t${{< /math >}}; {{< math >}}$-\log\rho_{\mathrm{geo}}${{< /math >}} estimates the mean KL | {{< math >}}$C_{\text{low}} \le \rho_{\mathrm{geo}} \le C_{\text{high}}${{< /math >}} | {{< math >}}$\rho_{\mathrm{geo}}${{< /math >}} |
| TRM, mean KL | {{< math >}}$\frac{1}{T}\sum_t D_t${{< /math >}}, exact | at most {{< math >}}$\delta_{\mathrm{avg}}${{< /math >}} | {{< math >}}$\rho_t${{< /math >}} per token |
| TRM, max KL | {{< math >}}$\max_t D_t${{< /math >}}, exact | at most {{< math >}}$\delta${{< /math >}} | {{< math >}}$\rho_t${{< /math >}} per token |

---

## 5. Other Things We Tried: Engine and Hardware Settings {#other-settings}

Before correcting for the mismatch in the loss, we tried to remove it at its source. The table summarizes the six changes; the details follow.

| What we changed | Setting | What happened | Verdict |
|---|---|---|---|
| FP32 `lm_head` in vLLM | L20, resumed at step 200 | `vllm-kl` still exploded and the run collapsed | Did not help |
| Chunked prefill off | L20, resumed at step 200 | `vllm-kl` spiked to about 1.4; rewards fell to near zero | Did not help |
| `enforce_eager`, `free_cache_engine` | H100, Qwen3-4B-Base, reasoning RL | No effect, but `vllm-kl` stayed at or below 1e-3 in every run | Inconclusive at low mismatch |
| Sampling `top-p` 0.98 to 0.999 | L20, 100 steps | Fewer `vllm-kl` spikes, slower reward growth | A diagnostic, not a fix |
| H20 instead of L20 | From scratch | Much smaller mismatch; trained stably for much longer | Helped |
| Cascade attention off | A100 | `vllm-kl` from about 3e-2–1e-1 down to about 1e-3 | Helped on A100 |

Top-p sampling is a diagnostic rather than a fix: it confirms that low-probability tokens drive the mismatch, but it slows learning. Changing GPUs or attention kernels helps only where they are the source of the mismatch. None of these replaces the correction of [§4](#4-attempts-to-alleviate-the-training-inference-mismatch).

### 5.1 Sampler Settings That Did Not Help {#41-what-did-not-help}

#### Use an FP32 LM head {#411-use-an-fp32-lm-head}

Inspired by the [MiniMax-M1 technical report](https://arxiv.org/abs/2506.13585) and the blog post [Your Efficient RL Framework Secretly Brings You Off-Policy RL Training](https://fengyao.notion.site/off-policy-rl), we patched vLLM to cast `lm_head` to fp32. In our experiments, the mismatch persisted after the patch and training still collapsed. Figure 16 shows a failed on-policy run on L20 with a bf16 `lm_head` in vLLM, and a run resumed from its step-200 checkpoint with an fp32 `lm_head`. Both ultimately collapsed, and the fp32 run still showed a `vllm-kl` explosion.

<figure class="post-fig">
<img src="/figs/img/rl-collapse/fig11-fp32-lm-head.png" alt="After resuming from step 200 with an fp32 lm_head (green), vllm-kl still climbs to about 25 and rewards still collapse, just as in the original bf16 run (red).">
<figcaption><strong>Figure 16.</strong> Ablation on the precision of <code>lm_head</code> in the vLLM engine. <strong>Left:</strong> <code>vllm-kl</code>. <strong>Right:</strong> training rewards. We resumed training with an fp32 <code>lm_head</code> (green) from step 200 of the failed run with a bf16 <code>lm_head</code> (red).</figcaption>
</figure>

#### Disable chunked prefill {#412-disable-chunked-prefill}

We also resumed training from step 200 of the failed bf16 `lm_head` run above with [chunked prefill](https://docs.vllm.ai/en/stable/configuration/optimization.html#chunked-prefill) disabled, to see whether it would resolve the crash. It did not prevent a severe crash: `vllm-kl` still spiked to about 1.4 near step 220, and rewards fell to near zero around step 237, as the original run's did, before partly recovering by the end of the run at about step 257 (Figure 17).

<figure class="post-fig">
<img src="/figs/img/rl-collapse/fig12-chunked-prefill.png" alt="With chunked prefill disabled (green), vllm-kl still spikes to about 1.4 near step 220 and rewards crash to near zero around step 237 before partly recovering, much like the original run (red).">
<figcaption><strong>Figure 17.</strong> Ablation on chunked prefill. We reran training with chunked prefill disabled (green) from step 200 of the failed run with chunked prefill enabled (red). <strong>Left:</strong> <code>vllm-kl</code>. <strong>Right:</strong> training rewards.</figcaption>
</figure>

#### Toggle `enforce_eager` and `free_cache_engine` {#413-toggle-enforce_eager-and-free_cache_engine}

[VeRL's official DAPO recipe](https://verl.readthedocs.io/en/latest/algo/dapo.html) mentions that enabling CUDA graphs (`enforce_eager=False`) may degrade model performance. To see whether it affects the training-inference mismatch, we ran an ablation on the vLLM hyperparameter `enforce_eager` together with `free_cache_engine`, in reasoning RL: on-policy GRPO on H100 GPUs with Qwen3-4B-Base, over all four combinations of the two flags set to `True` or `False`, evaluated on AIME24. Figure 18 shows that neither flag has a significant effect on the training-inference mismatch or on test performance. In these runs `vllm-kl` stayed at or below about 1e-3 in every setting, though, so this ablation could not detect an effect at the mismatch levels where collapse occurs.

<figure class="post-fig">
<img src="/figs/img/rl-collapse/fig13-enforce-eager.jpg" alt="vllm-kl stays below about 1.1e-3 and AIME24 accuracy plateaus near 0.3 for all four enforce_eager / free_cache_engine combinations.">
<figcaption><strong>Figure 18.</strong> Ablation on vLLM acceleration and cache hyperparameters in reasoning RL: on-policy GRPO on H100 GPUs with the four combinations of <code>enforce_eager</code> (<strong>E</strong>) and <code>free_cache_engine</code> (<strong>F</strong>). <strong>Left:</strong> <code>vllm-kl</code>. <strong>Right:</strong> accuracy on AIME24.</figcaption>
</figure>

### 5.2 Top-p Sampling: A Diagnostic {#423-top-p-sampling-a-diagnostic}

Low-probability tokens under the vLLM policy are where the mismatch is most severe. To test this directly, we ran on-policy GRPO on L20 GPUs with the vLLM sampling `top-p` set to 0.98, 0.99 and 0.999, without importance sampling. A smaller `top-p` removes the rarest tokens at sampling time, and Figure 19 confirms that it reduces the `vllm-kl` spikes. But truncation is itself a mismatch: the sampler now draws from a top-p-truncated distribution while the trainer optimizes the full {{< math >}}$\pfsdp${{< /math >}}, and `vllm-kl`, computed from the logged log-probabilities, need not reflect this shift. Without an IS correction this adds bias, and in Figure 19 the reward rises more slowly as `top-p` decreases. None of these 100-step runs collapsed, so we read top-p as evidence about where the mismatch comes from rather than as a fix.

<figure class="post-fig">
<img src="/figs/img/rl-collapse/fig19-top-p.png" alt="vllm-kl and rewards for top-p of 0.98, 0.99 and 0.999: smaller top-p gives fewer vllm-kl spikes but slower reward growth.">
<figcaption><strong>Figure 19.</strong> Ablation on the vLLM sampling <code>top-p</code>: on-policy GRPO with <code>top-p</code> of 0.98, 0.99 and 0.999. A smaller <code>top-p</code> generally gives a smaller <code>vllm-kl</code> and slower reward growth. <strong>Left:</strong> <code>vllm-kl</code>. <strong>Right:</strong> training rewards.</figcaption>
</figure>

### 5.3 Hardware and Kernels {#hardware-fixes}

Finding 4 ([§3.4](#hardware)) says that the hardware sets the size of the mismatch. Two changes on that side helped.

#### Use another GPU series {#424-use-another-gpu-series}

After finding that the training–inference mismatch on H20 GPUs was much smaller than on L20 GPUs, we moved all our TIR experiments to H20, which greatly reduced training crashes. Figure 20 shows two on-policy GRPO runs trained from scratch with identical configurations on L20 and H20. The H20 run trained stably for much longer and performed better.

<figure class="post-fig">
<img src="/figs/img/rl-collapse/fig20-l20-vs-h20.png" alt="Training rewards and test accuracy for identical on-policy GRPO runs on L20 and H20: the H20 run stays stable longer and reaches higher values.">
<figcaption><strong>Figure 20.</strong> Training rewards (<strong>left</strong>) and test accuracy (<strong>right</strong>) of two on-policy GRPO experiments on L20 and H20 GPUs.</figcaption>
</figure>

#### Disable cascade attention in vLLM {#cascade-attention}

Following [this flash-attention pull request](https://github.com/vllm-project/flash-attention/pull/87), we set `disable_cascade_attn=True` when initializing the vLLM engine, and found that it greatly reduces the training-inference mismatch on A100 GPUs. We ran two on-policy GRPO experiments with Qwen3-14B-Base on A100 GPUs, with `disable_cascade_attn` set to `True` and `False`. As Figure 21 shows, disabling cascade attention brought `vllm-kl` down from about 3e-2–1e-1 to around 1e-3. With cascade attention enabled, the training reward stayed near 0.01 from the first step, so the rollouts were effectively broken from the start rather than degrading over training; with it disabled, training proceeded normally.

<figure class="post-fig">
<img src="/figs/img/rl-collapse/fig21-cascade-attention.png" alt="vllm-kl and rewards on A100 with cascade attention enabled and disabled: disabling it lowers vllm-kl to about 1e-3 and gives higher rewards.">
<figcaption><strong>Figure 21.</strong> Ablation on cascade attention: two on-policy experiments on A100 GPUs, one with cascade attention enabled (purple) and one with the same configuration but cascade attention disabled (red). <strong>Left:</strong> <code>vllm-kl</code>. <strong>Right:</strong> training rewards.</figcaption>
</figure>

---

## 6. Conclusion and Key Takeaways for Practitioners {#5-conclusion-and-key-takeaways-for-practitioners}

The training-inference mismatch is not a niche bug. It follows from the pursuit of throughput, and it matters most in **reasoning and agentic RL**. The findings are listed in [the box at the top](#tldr); this is what they mean in practice.

- **Expect some mismatch.** Fast inference kernels and training kernels differ for good reasons. [Bitwise-consistent training and inference](https://blog.vllm.ai/2025/11/10/bitwise-consistent-train-inference.html) is possible, but it costs throughput, so most stacks will keep some mismatch.
- **Monitor it.** `vllm-kl` is an early warning. Track it together with perplexity and gradient norm, and watch multi-turn tool-use runs first: out-of-distribution tool responses raise the share of low-probability tokens, where the two engines disagree most.
- **Validate on your target hardware.** The same code can collapse on one GPU series and train much longer on another.
- **Correct at the sequence level, and prefer masking.** In our multi-turn tool-use runs token-level corrections still collapsed; in simpler reasoning RL they prevented collapse but gave little or no gain in peak accuracy. Among the corrections we tried on L20, **MIS** trained longest and reached the highest reward. What we recommend today is a length-normalized sequence mask: [Geo-MIS](#geo-mis) (DeepSeek-V3.2 uses a closely related mask), or [Trust Region Masking](https://arxiv.org/abs/2512.23075) when the sampler's logits are stored ([§4.4](#geo-mis)).

We hypothesize that the mismatch between inference engines (e.g., vLLM) and training frameworks (e.g., Megatron-LM) will also be a significant issue for **Mixture-of-Experts (MoE) RL**, a direction for future work.
