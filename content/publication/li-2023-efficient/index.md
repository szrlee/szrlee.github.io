---
title: Efficient and scalable reinforcement learning via hypermodel
authors:
- Yingru Li
- Jiawei Xu
- Zhiquan Luo
date: '2023-01-01'
publishDate: '2024-02-21T16:25:24.145987Z'
publication_types:
- paper-conference
publication: '*NeurIPS 2023 Workshop on Adaptive Experimental Design and Active Learning in the Real World*'


abstract: We design efficient and scalable RL algorithms for complex environments with hypermodel and approximate Thompson sampling, which demonstrates significant efficiency gain in DRL benchmark problems (e.g. only 15\% data consumption and 5\% model parameters compared to SOTAs in Arcade Learning Environment. We developed new probability tools for the sequential random projection and sequential subspace embedding via stopping-time argument and self-normalized martingale, which can be regard as a non-trivial extension to the renowned Johnson–Lindenstrauss (JL) lemma. The tools are then applied to the regret analysis of hypermodel-based TS-type algorithms in bandit and RL environments, achieving the same regret order of RLSVI and PSRL with cheap computation.

# Summary. An optional shortened abstract.
summary: Addressing efficiency challenge in RL with theoretical advancements and practical algorithm designs

tags:
  - Efficiency
  - Reinforcement Learning
  - Algorithm
  - Deep Learning
  - Regret
  - Random Projection
  - Martingale
  - Application

# Display this page in the Featured widget?
# featured: true

# Custom links (uncomment lines below)
links:
- name: Workshop
  url: https://neurips.cc/virtual/2023/78739

url_pdf: 'uploads/paper/HyperFQI_camera_ready.pdf'
# url_code: 'https://github.com/szrlee/HyperFQI'
# url_dataset: 'https://github.com/HugoBlox/hugo-blox-builder'
url_poster: 'uploads/slides/HyperFQI_Short.pdf'
# url_project: ''
url_slides: 'uploads/slides/AGI_humanity_talk_1021_2023.pdf'
# url_source: 'https://github.com/HugoBlox/hugo-blox-builder'
# url_video: 'https://youtube.com'

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
image:
  caption: 'Significant gain in data and computation efficiency'
  focal_point: ''
  preview_only: false

# Associated Projects (optional).
#   Associate this publication with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `internal-project` references `content/project/internal-project/index.md`.
#   Otherwise, set `projects: []`.
projects:
  - Efficient RL

# Slides (optional).
#   Associate this publication with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides: "example"` references `content/slides/example/index.md`.
#   Otherwise, set `slides: ""`.
# slides: example
---

Data-efficient reinforcement learning(RL) requires deep exploration.
Thompson sampling is a principled method for deep exploration in reinforcement learning.
However, Thompson sampling need to track the degree of uncertainty by maintaining the posterior distribution of models, which is computationally feasible only in simple environments with restrictive assumptions.
A key problem in modern RL is how to develop data and computation efficient algorithm that is scalable to large-scale complex environments.
We develop a principled framework, called HyperFQI, to tackle both the computation and data efficiency issues.
HyperFQI can be regarded as approximate Thompson sampling for reinforcement learning based on hypermodel. Hypermodel in this context serves as the role for uncertainty estimation of action-value function.
HyperFQI demonstrates its ability for efficient and scalable deep exploration in DeepSea benchmark with large state space.
HyperFQI also achieves super-human performance in Atari benchmark with 2M interactions with low computation costs.
We also give a rigorous performance analysis for the proposed method, justifying its computation and data efficiency.
To the best of knowledge, this is the first principled RL algorithm that is provably efficient and also practically scalable to complex environments such as Arcade learning environment that requires deep networks for pixel-based control.
We believe this work serves as a bridge for theory and practice in reinforcement learning.
---
