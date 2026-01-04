---
title: 'Beyond Precision: Why Training-Inference Mismatch is an Optimization Problem and How Simple LR Scheduling Fixes It'
subtitle: 'A learning rate scheduling approach to stabilize LLM-RL training'
summary: 'RL training for LLMs is notoriously unstable. While recent studies attribute this to training-inference mismatch from hybrid engines, we show this is not merely a static numerical issue, but a dynamic problem coupled with the model's optimization trajectory. We propose a specialized Learning Rate Scheduler that decays LR as gradient noise rises, using response length surge as a reliable early indicator of impending instability.'
authors:
  - Yaxiang Zhang
  - admin
  - Jiacai Liu
  - Ziniu Li
  - Jiawei Xu
  - Qian Liu
tags:
  - Reinforcement Learning
  - Language Models
  - Deep Learning
  - Optimization
  - Learning Rate Scheduling
categories:
  - Research
  - Theory
date: '2025-12-20T00:00:00Z'
lastmod: '2025-12-20T00:00:00Z'
featured: true
draft: false
external_link: 'https://yingru.notion.site/Beyond-Precision-Why-Training-Inference-Mismatch-is-an-Optimization-Problem-and-How-Simple-LR-Sched-2d9211a558b780f1a710f99dbdc403d3'

# Featured image
image:
  caption: 'Learning Rate Scheduling for LLM-RL Stability'
  focal_point: ''
  preview_only: false

# Projects (optional)
projects: []
---

**Corresponding Author: Yingru Li**

**Co-First Authors: Yaxiang Zhang and Yingru Li**

## TL;DR

- **The Problem:** Reinforcement Learning (RL) training for LLMs is notoriously unstable. While recent studies attribute this to "training-inference mismatch" (caused by hybrid engines), standard fixes like Importance Sampling might fail during longer training runs.
- **The Insight:** We analyze this instability through an optimization lens. We find that as training progresses, gradient noise and training-inference mismatch increases simultaneously. This suggests that the "mismatch" is **not merely a static numerical issue, but a dynamic problem coupled with the model's optimization trajectory.**
- **The Solution:** A specialized **Learning Rate (LR) Scheduler**.
  - **Mechanism:** By decaying the learning rate as gradient noise rises, we can consistently stabilize RL training and keep the training-inference mismatch at a safe level.
  - **Heuristic:** We propose a novel method to time this decay based on **Response Length**. The surge in response length serves as a reliable early indicator of impending instability, signaling exactly when to reduce the learning rate.

[Read the full article on Notion →](https://yingru.notion.site/Beyond-Precision-Why-Training-Inference-Mismatch-is-an-Optimization-Problem-and-How-Simple-LR-Sched-2d9211a558b780f1a710f99dbdc403d3)
