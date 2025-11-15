---
title: 'When Speed Kills Stability: Demystifying RL Collapse from the Training-Inference Mismatch'
subtitle: 'How the inference-training gap causes catastrophic failures in LLM reinforcement learning'
summary: 'The relentless push for faster inference creates a dangerous training-inference mismatch that silently kills RL with LLMs. We reveal the vicious cycle—particularly acute in reasoning and agentic RL—and show that sequence-level importance sampling is the principled solution.'
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
categories:
  - Research
  - Theory
date: '2025-09-17T00:00:00Z'
lastmod: '2025-09-17T00:00:00Z'
featured: true
draft: false
external_link: 'https://yingru.notion.site/When-Speed-Kills-Stability-Demystifying-RL-Collapse-from-the-Training-Inference-Mismatch-271211a558b7808d8b12d403fd15edda'

# Featured image
image:
  caption: 'Training-Inference Mismatch in LLM-RL'
  focal_point: ''
  preview_only: false

# Projects (optional)
projects: []
---

**Co-First Authors: Jiacai Liu and Yingru Li**

**Corresponding Authors: Yingru Li and Yu Shen**

## TL;DR

The relentless push for faster inference has created a dangerous "training-inference mismatch" that can silently kill reinforcement learning with LLMs. Our investigation reveals a vicious cycle that is particularly acute in modern reasoning and agentic RL:

- **OOD Contexts Drive Low-Probability Sampling:** Agentic workflows expose models to external inputs and dynamic environments, forcing frequent generation of low-probability tokens that are essential for novel reasoning, tool calls, and adaptive responses.
- **Low-Probability Tokens Amplify Training Collapse:** These tokens become the weakest link—the training-inference mismatch is most severe for them, causing catastrophically large gradients that lead to silent degradation and sudden training failure.
- **Hardware Variability Complicates the Problem:** Different GPU architectures exacerbate the mismatch unpredictably, meaning the same agentic training setup can succeed on one machine and catastrophically fail on another.
- **Sequence-Level IS is the Principled Solution:** Sequence-level Importance Sampling emerges as the theoretically grounded fix, restoring training stability across different hardware and complex tasks.

[Read the full article on Notion →](https://yingru.notion.site/When-Speed-Kills-Stability-Demystifying-RL-Collapse-from-the-Training-Inference-Mismatch-271211a558b7808d8b12d403fd15edda)
