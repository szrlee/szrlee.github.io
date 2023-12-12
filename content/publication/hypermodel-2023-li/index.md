---
title: An analysis of hypermodel and approximate Thompson sampling
authors:
- Yingru Li
- Zhi-Quan Luo
date: '2023-12-31'
publishDate: '2023-12-07T12:23:02.222401Z'
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
summary: We have developed a novel random projection tool for sequentially dependent data, which extends the Johnson–Lindenstrauss lemma in a non-trivial way. Wi th this novel analytical tool, we prove approximate Thompson sampling (TS) with hypermodel can achieve the same regret order of TS with exponential smaller computation requirement than the ensemble sampling. This result effectively addresses data and computation efficiency challenges in online decision-making (bandit and RL), bridging theory and practice.

# Display this page in the Featured widget?
featured: true

# Custom links (uncomment lines below)
# links:
# - name: Custom Link
#   url: http://example.org

url_pdf: '#comming-soon'
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
