// MathJax configuration (overrides the Hugo Blox default to add macros).
// \ps and \pt are the sampler and trainer policies, colored by CSS (.mjx-ps, .mjx-pt)
// to match the step-through figures in both themes; \smu and \pmodel are the same two
// colors for the sampling distribution mu and the model's next-token distribution p.
window.MathJax = {
  options: {
    // Don't render math in mindmaps as Markmap has its own math renderer.
    ignoreHtmlClass: 'markmap',
  },
  tex: {
    inlineMath: [
      ['$', '$'],
      ['\\(', '\\)'],
    ],
    displayMath: [
      ['$$', '$$'],
      ['\\[', '\\]'],
    ],
    processEscapes: false,
    packages: {'[+]': ['noerrors', 'html']},
    macros: {
      ps: '{\\class{mjx-ps}{\\pi_{\\text{sampler}}}}',
      pt: '{\\class{mjx-pt}{\\pi_{\\text{trainer}}}}',
      smu: '{\\class{mjx-ps}{\\mu}}',
      pmodel: '{\\class{mjx-pt}{p}}',
    },
  },
  loader: {
    load: ['[tex]/noerrors', '[tex]/html'],
  },
};
