---
title: "Adaptive Foundation Models for Online Decisions: HyperAgent with Fast Incremental Uncertainty Estimation"
authors:
- Yingru Li
- Jiawei Xu
- Zhi-Quan Luo
date: '2024-07-21'
publishDate: '2024-05-03T17:13:02.195766Z'
publication_types:
- manuscript
publication: '*Preprint. Presentation at **ICML** 2024 Workshops: (1) "Aligning Reinforcement Learning Experimentalists and Theorists"; (2) "Automated Reinforcement Learning: Exploring Meta-Learning, AutoML, and LLMs"*'

abstract: Foundation models often struggle with uncertainty when faced with new situations in online decision-making, necessitating scalable and efficient exploration to resolve this uncertainty. We introduce GPT-HyperAgent, an augmentation of GPT with HyperAgent for uncertainty-aware, scalable exploration in contextual bandits, a fundamental online decision problem involving natural language input. We prove that HyperAgent achieves fast incremental uncertainty estimation with $\tilde{O}(\log T)$ per-step computational complexity over $T$ periods under the linear realizable assumption. Our analysis demonstrates that HyperAgent's regret order matches that of exact Thompson sampling in linear contextual bandits, closing a significant theoretical gap in scalable exploration. Empirical results in real-world contextual bandit tasks, such as automated content moderation with human feedback, validate the practical effectiveness of GPT-HyperAgent for safety-critical decisions. Our code is open-sourced at \url{https://github.com/szrlee/GPT-HyperAgent/}.


# Summary. An optional shortened abstract.
summary: We prove HyperAgent closes a theoretical gap in scalable exploration. Further, GPT-HyperAgent addresses challenges in human-Al pipeline for automated content moderation with human feedback.

tags:
  - agent
  - foundation-model
  - application
  - theory
  - probability
  - bandit
  - regret

# Display this page in the Featured widget?
featured: true

# Custom links (uncomment lines below)
# links:
# - name: Custom Link
#   url: http://example.org

# url_pdf: '#comming-soon'
url_pdf: 'https://arxiv.org/abs/2407.13195'
# url_code: 'https://github.com/HugoBlox/hugo-blox-builder'
# url_dataset: 'https://github.com/HugoBlox/hugo-blox-builder'
url_poster: 'https://icml.cc/virtual/2024/search?query=yingru'
# url_poster: ''
# url_project: ''
url_slides: 'uploads/slides/HyperAgent_RLCN.pdf'
# url_source: 'https://github.com/HugoBlox/hugo-blox-builder'
# url_video: 'https://youtube.com'

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
# image:
#   caption: 'Significant gain in data and computation efficiency'
#   focal_point: ''
#   preview_only: false

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


### Summary of technical contributions of HyperAgent

1. New probability tools in high-dimensional probability and statistics
- The [first probability tool](/publication/li-2024-probability) for sequential random projection, a non-trivial martingale extension of Johnson-Lindenstrauss (JL) for adaptively sampled data due to the sequential nature of RL;
- A [unified and simple analysis](/publication/li-2024-simple) for JL via high-dimension extension of Hanson-Wright.

2. Methodology for sequential-decision making
- Hypermodel: efficient incremental approximation of the posterior (uncertainty quantification) over complex models without leveraging conjugacy as encountering more data;
- Index sampling: approximate posterior sampling for data-efficient sequential decision-making.

3. Results for sequential-decision making
- Practically, our developed HyperAgent demonstrates its robust performance in large-scale deep RL benchmarks with significant efficiency gain in terms of both data and computation;
- Theoretically, the first method to achieve logarithmic per-step computation and sublinear under [tabular episodic RL](/publication/li-2024-hyperagent/) and [linear contextual bandit](/publication/li-2024-scaling/) setups among practically scalable algorithms. At the heart of the analysis is the sequential incremental posterior approximation argument, made possible by the our developed first probability tool for sequential random projection.
