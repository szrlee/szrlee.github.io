---
title: HyperAgent - A Simple, Efficient and Scalable RL Framework for Complex Environments

event: Contributed Talk in The third doctoral and postdoctoral Daoyuan academic forum. Best paper award
# event_url: https://example.org

location: Daoyuan Building
address:
  # street: 450 Serra Mall
  city: Shenzhen
  # region: CA
  # postcode: '94305'
  country: China

summary: Practically and provably efficient RL under resource constraints!
abstract: Under resource constraints, reinforcement learning (RL) agents need to be simple, efficient and scalable with (1) large state space and (2) increasingly accumulated data of interactions when deploying in complex environments. We propose the HyperAgent, a RL framework with hypermodel, index sampling schemes and incremental update mechanism, enabling computation-efficient sequential posterior approximation and data-efficient action selection under general value function approximation beyond conjugacy. The implementation of HyperAgent is simple as it only add one module and a line of code additional to DDQN.
Practically, HyperAgent demonstrates its robust performance in large-scale deep RL benchmarks with significant efficiency gain in terms of both data and computation. Theoretically, among the practically scalable algorithms, HyperAgent is the first achieving provably scalable per-step computational complexity as well as sublinear regret under tabular RL. The core of our theoretical analysis is the sequential posterior approximation argument. This is made possible by the first analytical tool for sequential random projection, a non-trivial martingale extension of the Johnson-Lindenstrauss lemma, which is of independent interest. 
This work bridges the theoretical and practical realms of RL, establishing a new benchmark for RL algorithms design.



# Talk start and end times.
#   End time can optionally be hidden by prefixing the line with `#`.
date: '2024-1-13T13:20:00Z'
date_end: '2024-1-13T13:50:00Z'
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
url_code: ''
url_pdf: ''
url_slides: 'uploads/slides/HyperAgent_slides_0113_2024.pdf'
url_video: ''
- icon: newspaper
  icon_pack: fab
  name: News
  url: https://mp.weixin.qq.com/s/erfgIgYJCjYg2aRTnqseuQ



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
<!-- 
We embark on a compelling journey towards Artificial General Intelligence (AGI) and emphasize its profound impact on humanity. We begin by defining AGI and its transformative potential, underlining the central role of Reinforcement Learning (RL) in achieving this aspiration. We explore the real-world applications of RL, from plasma control to ChatGPT, shedding light on the pressing need for efficient RL algorithms. Enter HyperFQI, an innovative solution to RL efficiency challenges we developed, boasting generality and scalability. Witness its remarkable efficiency in benchmark results, particularly in Atari video games. Discover the practical integration of HyperFQI, adapting seamlessly into existing RL frameworks. Delve into the theoretical guarantees of HyperFQI in tabular settings, featuring rigorous mathematical probability tools we developed. This presentation bridges theory and practice, elucidating HyperFQI’s pivotal role in the expedition toward AGI, with a direct impact on realizing AGI’s potential for the betterment of humanity. The talk concludes by underscoring the transformative potential of efficient RL agents and their promise for the future of AGI and, indeed, humanity. -->

<!-- 
{{% callout note %}}
Click on the **Slides** button above to view the built-in slides feature.
{{% /callout %}}

Slides can be added in a few ways:

- **Create** slides using Hugo Blox Builder's [_Slides_](https://docs.hugoblox.com/reference/content-types/) feature and link using `slides` parameter in the front matter of the talk file
- **Upload** an existing slide deck to `static/` and link using `url_slides` parameter in the front matter of the talk file
- **Embed** your slides (e.g. Google Slides) or presentation video on this page using [shortcodes](https://docs.hugoblox.com/reference/markdown/).

Further event details, including [page elements](https://docs.hugoblox.com/reference/markdown/) such as image galleries, can be added to the body of this page. -->