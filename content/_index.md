---
# Leave the homepage title empty to use the site title
title: ''
date: 2023-12-07
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

  - block: contact
    id: contact
    content:
      title: Get in touch
      text: |-
        Open to conversations on research collaboration, advising, and speaking.

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
