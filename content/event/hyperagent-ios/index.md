---
title: HyperAgent - A Simple, Efficient, Scalable and Provable RL Framework

event: Informs Optimization Society Conference 2024
event_url: https://ios2024.rice.edu

location: Rice University
address:
  street: 6100 Main St.
  city: Houston
  region: TX
  # postcode: '94305'
  country: US

summary: Practically and provably efficient RL under resource constraints!
abstract: Under resource constraints, reinforcement learning (RL) agents need to be simple, efficient and scalable with (1) large state space and (2) increasingly accumulated data of interactions when deploying in complex environments. We propose the HyperAgent, a RL framework with hypermodel, index sampling schemes and incremental update mechanism, enabling computation-efficient sequential posterior approximation and data-efficient action selection under general value function approximation beyond conjugacy. The implementation of HyperAgent is simple as it only add one module and a line of code additional to DDQN. Practically, HyperAgent demonstrates its robust performance in large-scale deep RL benchmarks with significant efficiency gain in terms of both data and computation. Theoretically, among the practically scalable algorithms, HyperAgent is the first achieving provably scalable per-step computational complexity as well as sublinear regret under tabular RL. The core of our theoretical analysis is the sequential posterior approximation argument. This is made possible by the first analytical tool for sequential random projection, a non-trivial martingale extension of the Johnson-Lindenstrauss lemma, which is of independent interest. This work bridges the theoretical and practical realms of RL, establishing a new benchmark for RL algorithms design.



# Talk start and end times.
#   End time can optionally be hidden by prefixing the line with `#`.
date: '2024-03-23T13:30:00Z'
# date_end: '2024-01-13T13:50:00Z'
all_day: false

# Schedule page publish date (NOT talk date).
publishDate: '2017-01-01T00:00:00Z'

draft: false

authors:
  - admin
tags:
  - Efficiency
  - Reinforcement Learning
  - Algorithm
  - Deep Learning
  - Regret
  - Random Projection

# Is this a featured talk? (true/false)
featured: true

# image:
#   caption: 'Image credit: [**Unsplash**](https://unsplash.com/photos/bzdhc5b3Bxs)'
#   focal_point: Right

links:
- icon: twitter
  icon_pack: fab
  name: Follow
  url: https://twitter.com/RichardYRLi
# https://docs.hugoblox.com/getting-started/page-builder/

url_code: ''
url_pdf: ''
url_slides: 'uploads/slides/HyperAgent_slides_ios.pdf'
url_video: 'https://meeting.tencent.com/v2/cloud-record/share?id=IHGtdxajucDxd2ULGfRYB5m_cGzjkUE43Age4ynky0k&from=3&is-single=true&record_type=1'


# Markdown Slides (optional).
#   Associate this talk with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides = "example-slides"` references `content/slides/example-slides.md`.
#   Otherwise, set `slides = ""`.
# slides: example

# Projects (optional).
#   Associate this post with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["internal-project"]` references `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.
projects:
  - Efficient RL
---