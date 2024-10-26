---
title: 'Q-Star Meets Scalable Posterior Sampling: Bridging Theory and Practice via HyperAgent'
authors:
- Yingru Li
- Jiawei Xu
- Lei Han
- Zhiquan Luo
date: '2024-07-21'
publishDate: '2024-02-21T16:25:24.139610Z'
publication_types:
- paper-conference
publication: '*International Conference on Machine Learning (**ICML**)*'
# links:
# - name: paper
#   url: https://arxiv.org/abs/2402.10228

url_code: 'https://github.com/szrlee/HyperAgent'
url_video: 'https://meeting.tencent.com/v2/cloud-record/share?id=IHGtdxajucDxd2ULGfRYB5m_cGzjkUE43Age4ynky0k&from=3&is-single=true&record_type=1'


url_pdf: 'https://arxiv.org/abs/2402.10228'
# url_code: 'https://github.com/HugoBlox/hugo-blox-builder'
# url_dataset: 'https://github.com/HugoBlox/hugo-blox-builder'
url_poster: 'https://icml.cc/virtual/2024/poster/34173'
# url_project: ''
url_slides: 'uploads/slides/HyperAgent_RLCN.pdf'
# url_source: 'https://github.com/HugoBlox/hugo-blox-builder'
# url_video: 'https://youtube.com''

featured: true
abstract: We propose HyperAgent, a reinforcement learning (RL) algorithm based on the hypermodel framework for exploration in RL. HyperAgent allows for the efficient incremental approximation of posteriors associated with an optimal action-value function ($Q^\star$) without the need for conjugacy and follows the greedy policies w.r.t. these approximate posterior samples. We demonstrate that HyperAgent offers robust performance in large-scale deep RL benchmarks. It can solve Deep Sea hard exploration problems with episodes that optimally scale with problem size and exhibits significant efficiency gains in the Atari suite. Implementing HyperAgent requires minimal code addition to well-established deep RL frameworks like DQN. We theoretically prove that, under tabular assumptions, HyperAgent achieves logarithmic per-step computational complexity while attaining sublinear regret, matching the best known randomized tabular RL algorithm.

# Summary. An optional shortened abstract.
summary: Addressing data and computation efficiency challenges in real-world deployments of RL Agents. It achieves significant efficiency gains in deep RL benchmarks as well as theoretical milestones.

projects:
  - Efficient RL

tags:
  - agent
  - probability
  - theory
  - reinforcement-learning
  - selected
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
