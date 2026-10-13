---
# Leave the homepage title empty to use the site title
title: ''
date: 2023-12-07
type: landing

# Blocks live in layouts/partials/blocks/home.*.html; their content comes from data/research.yaml.
sections:
  - block: home.hero
    id: about
    content:
      username: admin
      kicker: Member of Technical Staff · xAI
      title: Reinforcement learning, efficient in data and in compute
      text: |-
        I work on large-scale reinforcement learning at [xAI](https://x.ai/): stability, efficiency, and the science of scaling.
        Before that I led research on RL training stability and efficiency for LLM reasoning and agents at ByteDance.

        I received my Ph.D. from The Chinese University of Hong Kong in 2025, advised by [Prof. Zhi-Quan (Tom) Luo](https://tomluo123.github.io/), with [Prof. Benjamin Van Roy](https://web.stanford.edu/~bvr/) on my thesis committee.
        I also had the privilege of collaborating with [Prof. Tong Zhang](https://tongzhang-ml.org/) and [Prof. John Hopcroft](https://www.cs.cornell.edu/jeh/).

  - block: home.framework
    id: framework

  - block: home.questions
    id: research
    content:
      label: Where my work sits
      title: Three questions, one quantity
      text: >-
        Information per sample is decided by which experience the learner collects. The fraction
        each update extracts is lost in two ways: an update that points the wrong way (bias) or
        one that is drowned in noise (variance).

  - block: home.timeline
    id: timeline

  - block: home.outlook
    id: outlook

  - block: home.talks
    id: talks
    content:
      label: Talks
      title: Selected talks

  - block: home.writing
    id: writing
    content:
      label: Writing
      title: Derivations and notes, grouped by question

  - block: contact
    id: contact
    content:
      title: Contact
      text: |-
        szrlee [at] gmail [dot] com
      appointment_url: 'https://calendly.com/yrli'
      autolink: true
      form:
        provider: formspree
        formspree:
          id: szrlee@gmail.com
    design:
      columns: '2'
---
