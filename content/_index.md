---
# Leave the homepage title empty to use the site title
title: ''
date: 2026-10-22
type: landing

# Blocks live in layouts/partials/blocks/home.*.html; their content comes from data/home.yaml.
sections:
  - block: home.hero
    id: about
    content:
      username: admin

  - block: home.highlights
    id: highlights
    content:
      title: Highlights

  - block: home.experience
    id: experience
    content:
      title: Experience

  - block: home.selected
    id: publications
    content:
      title: Selected publications

  - block: home.contact
    id: contact
    content:
      title: Get in touch
---
