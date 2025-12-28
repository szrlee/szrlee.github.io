---
title: 'The Optimal Token Baseline'
subtitle: 'Variance Reduction for Long-Horizon LLM-RL'
summary: 'RL training for LLMs frequently suffers from training collapse due to exploding gradient variance in long-horizon tasks. We derive the Optimal Token Baseline (OTB) from first principles, proving that updates should be weighted inversely to their accumulated uncertainty (Realized Energy). Our computationally free Logit-Gradient Proxy eliminates training collapse, matches N=32 performance with just N=4, and reduces token consumption by 62-66%.'
authors:
  - admin
  - Jiawei Xu
  - Ziniu Li
  - Jiacai Liu
  - Yuxuan Tong
  - Wei Liu
  - Longtao Zheng
  - Zhenghai Xue
  - Yaxiang Zhang
  - Tianle Cai
  - Ge Zhang
  - Qian Liu
  - Baoxiang Wang
tags:
  - Reinforcement Learning
  - Language Models
  - Deep Learning
  - Variance Reduction
  - Optimization
categories:
  - Research
  - Theory
date: '2025-12-20T00:00:00Z'
lastmod: '2025-12-20T00:00:00Z'
featured: true
draft: false
external_link: 'https://yingru.notion.site/The-Optimal-Token-Baseline-399211a558b782cfa936014c0d42dfb8'

# Featured image
image:
  caption: 'Optimal Token Baseline for LLM-RL'
  focal_point: ''
  preview_only: false

# Projects (optional)
projects: []
---

**Project Lead: Yingru Li**

**Co-First Authors: Yingru Li and Jiawei Xu**

## TL;DR

- **The Problem**: RL training for LLMs frequently suffers from "training collapse" due to exploding gradient variance in long-horizon tasks. Standard baselines (like Group Mean) fail because they treat all tokens and sequences as equally "noisy."
- **The Insight**: Gradient noise is heterogeneous. We derive the **Optimal Token Baseline (OTB)** from first principles, proving that updates should be weighted inversely to their accumulated uncertainty (Realized Energy).
- **The Solution**: We introduce a **computationally free Logit-Gradient Proxy**. This allows us to approximate the true gradient norm using only forward-pass probabilities—requiring **zero** additional backward passes.
- **The Impact**:
  - **Stability**: Eliminates training collapse by stabilizing gradient norms.
  - **Efficiency**: Matches the performance of group size N=32 with just N=4.
  - **Savings**: Reduces token consumption by **62%** on Single-turn Reasoning and **66%** on Multi-turn Tool-Integrated Reasoning (TIR).

[Read the full article on Notion →](https://yingru.notion.site/The-Optimal-Token-Baseline-399211a558b782cfa936014c0d42dfb8)

[[Code](https://github.com/volcengine/verl/pull/4678)] [[Dataset](https://huggingface.co/datasets/Jiawei415/DPAO_filter)]
