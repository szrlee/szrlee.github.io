---
title: Provably scalable and near-optimal Thompson sampling via hypermodel with applications
  in decision-making language agents
authors:
- Yingru Li
- Jiawei Xu
- Zhi-Quan Luo
date: '2024-01-01'
publishDate: '2024-05-03T17:13:02.195766Z'
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
summary: Hypermodel for efficient incremental approximation of the posterior (uncertainty quantification) over complex models without leveraging conjugacy as encountering more data; Index sampling for approximate posterior sampling for data-efficient sequential decision-making. This approach is provably superior than ensemble sampling and Langevin Monte-carlo.

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


### Summary of contributions

1. New probability tools in high-dimensional probability and statistics
- The [first probability tool](/publication/li-2024-probability) for sequential random projection, a non-trivial martingale extension of Johnson-Lindenstrauss (JL) for adaptively sampled data due to the sequential nature of RL;
- A [unified and simple analysis](/publication/li-2024-simple) for JL via high-dimension extension of Hanson-Wright.

2. Methodology for sequential-decision making
- Hypermodel: efficient incremental approximation of the posterior (uncertainty quantification) over complex models without leveraging conjugacy as encountering more data;
- Index sampling: approximate posterior sampling for data-efficient sequential decision-making.

3. Results for sequential-decision making
- Practically, our developed HyperAgent demonstrates its robust performance in large-scale deep RL benchmarks with significant efficiency gain in terms of both data and computation;
- Theoretically, the first method to achieve logarithmic per-step computation and sublinear under [tabular episodic RL](/publication/li-2024-hyperagent/) and [linear bandit](/publication/li-2024-approximate/) setups among practically scalable algorithms. At the heart of the analysis is the sequential incremental posterior approximation argument, made possible by the our developed first probability tool for sequential random projection.
