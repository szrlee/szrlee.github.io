---
title: 'Mathematical Formulations of Rollout Correction Methods'
subtitle: 'A unified framework for handling general off-policy problems in RL training'
summary: 'Definitive mathematical formulations for rollout correction methods in VeRL, progressing from REINFORCE to PPO to Decoupled PPO. Handles policy mismatch, temporal lag, replay buffers, and off-policy algorithms with importance sampling and rejection sampling techniques.'
authors:
  - admin
tags:
  - Reinforcement Learning
  - Language Models
  - Deep Learning
  - Off-Policy Learning
  - Importance Sampling
  - VeRL
categories:
  - Research
  - Theory
  - Documentation
date: '2025-11-04T00:00:00Z'
lastmod: '2025-11-04T00:00:00Z'
featured: true
draft: false
external_link: 'https://verl.readthedocs.io/en/latest/algo/rollout_corr_math.html'
math: true

# Featured image
image:
  caption: 'VeRL Rollout Correction Framework'
  focal_point: ''
  preview_only: false

# Projects (optional)
projects: []
---

**Author:** [Yingru Li](https://richardli.xyz)

## Abstract

This document provides the definitive mathematical formulations for rollout correction methods in `verl`, following the natural progression from **REINFORCE** to **PPO** to **Decoupled PPO**.

Rollout correction provides a unified framework to handle **general off-policy problems** in RL training - any scenario where the data collection distribution differs from the training distribution.

**Applicable scenarios include:**
- **Policy mismatch**: Different precision (FP8 vs FP16 vs BF16 vs FP32), different backends (vLLM vs SGLang vs FSDP vs Megatron)
- **Temporal lag**: Model staleness, asynchronous rollout workers
- **Replay buffers**: Training on historical trajectories from earlier policy versions
- **Off-policy algorithms**: Behavioral cloning, DAPO, expert demonstrations
- **Data filtering**: Reweighting, preference learning, curriculum learning

## Key Topics

1. **Theoretical Foundation: From REINFORCE to Decoupled PPO**
   - REINFORCE policy gradient baseline
   - PPO with trust region control
   - Decoupled PPO for batch size invariance

2. **Implementation in VeRL: The Three-Policy Framework**
   - Policy roles: Rollout (behavior), Old (proximal), Current
   - Operating modes: Decoupled vs Bypass
   - Two distribution shifts and their corrections

3. **Algorithmic Components and Combinations**
   - IS/RS aggregation levels (token, sequence, geometric)
   - Loss functions (PPO vs policy gradient)
   - Safety mechanisms (veto, batch normalization)

4. **Off-Policy Diagnostic Metrics**
   - KL divergence, perplexity, chi-squared divergence

5. **Summary and Decision Guide**
   - Method comparison table
   - Scenario-based recommendations

[Read the full documentation →](https://verl.readthedocs.io/en/latest/algo/rollout_corr_math.html)

## Related Resources

- [Rollout Correction Usage Guide](https://verl.readthedocs.io/en/latest/algo/rollout_corr.html) - Practical implementation guide
- [VeRL GitHub Repository](https://github.com/volcengine/verl)

