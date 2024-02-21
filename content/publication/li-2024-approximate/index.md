---
title: Approximate Thompson sampling via Hypermodel and Index sampling
authors:
- Yingru Li
- Jiawei Xu
- Zhi-Quan Luo
date: '2024-01-01'
publishDate: '2024-02-21T16:25:24.114942Z'
publication_types:
- manuscript

tags:
  - Efficiency
  - Bandit
  - Algorithm
  - Regret
  - Random Projection
  - Martingale


# abstract: We design efficient and scalable RL algorithms for complex environments with hypermodel and approximate Thompson sampling, which demonstrates significant efficiency gain in DRL benchmark problems (e.g. only 15\% data consumption and 5\% model parameters compared to SOTAs in Arcade Learning Environment. We developed new probability tools for the sequential random projection and sequential subspace embedding via stopping-time argument and self-normalized martingale, which can be regard as a non-trivial extension to the renowned Johnson–Lindenstrauss (JL) lemma. The tools are then applied to the regret analysis of hypermodel-based TS-type algorithms in bandit and RL environments, achieving the same regret order of RLSVI and PSRL with cheap computation.

# Summary. An optional shortened abstract.
summary: Hypermodel for efficient incremental approximation of the posterior (uncertainty estimation) over complex models without leveraging conjugacy as encountering more data; Index sampling for approximate posterior sampling for data-efficient sequential decision-making. This approach is provably superior than ensemble sampling and Langevin Monte-carlo.

# Display this page in the Featured widget?
featured: true

# Custom links (uncomment lines below)
# links:
# - name: Custom Link
#   url: http://example.org

# url_pdf: '#comming-soon'
# url_code: 'https://github.com/HugoBlox/hugo-blox-builder'
# url_dataset: 'https://github.com/HugoBlox/hugo-blox-builder'
# url_poster: ''
# url_project: ''
# url_slides: ''
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


### Summary of contributions:

1. New probability tools in high-dimensional probability and statistics
- a non-trivial martingale extension of Johnson-Lindenstrauss (JL) for adaptively sampled data due to the sequential nature of RL;
- a unified and simple analysis for JL via high-dimension extension of Hanson-Wright.

2. Methodology for sequential-decision making
- hypermodel for efficient incremental approximation of the posterior (uncertainty estimation) over complex models without leveraging conjugacy as encountering more data;
- index sampling for approximate posterior sampling for data-efficient sequential decision-making.

3. Results for sequential-decision making
- Practically, our developed HyperAgent demonstrates its robust performance in large-scale deep RL benchmarks with significant efficiency gain in terms of both data and computation;
- Theoretically, first method to achieve logarithmic per-step computation and sublinear under tabular episodic RL among practically scalable algorithms.
