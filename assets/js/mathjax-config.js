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
      // The vLLM sampler and the FSDP trainer in "When Speed Kills Stability".
      pvllm: '{\\class{mjx-ps}{\\pi_{\\theta}^{\\text{vllm}}}}',
      pfsdp: '{\\class{mjx-pt}{\\pi_{\\theta}^{\\text{fsdp}}}}',
    },
  },
  loader: {
    load: ['[tex]/noerrors', '[tex]/html'],
  },
  startup: {
    ready: function () {
      MathJax.startup.defaultReady();
      MathJax.startup.promise.then(function () {
        glueMathPunctuation(document.body);
      });
    },
  },
};

// A browser may wrap a line between an inline formula and the punctuation next to it,
// which leaves a comma at the start of a line or an opening bracket at the end of one.
// Wrap the formula and that punctuation in a span that does not break (.math-glue).
function glueMathPunctuation(root) {
  var closing = /^[.,;:!?)\]}\u201d\u2019]+/;
  var opening = /[(\[{\u201c\u2018]+$/;
  var phrasing = /^(STRONG|EM|B|I|A)$/;
  // The text next to a node in reading order, looking through bold, italic and link elements.
  function neighbour(node, forward) {
    var n = node, s;
    while (!(s = forward ? n.nextSibling : n.previousSibling)) {
      n = n.parentNode;
      if (!n || !phrasing.test(n.nodeName)) return null;
    }
    while (s.nodeType === 1 && phrasing.test(s.nodeName) && s.firstChild) {
      s = forward ? s.firstChild : s.lastChild;
    }
    return s.nodeType === 3 ? s : null;
  }
  root.querySelectorAll('mjx-container:not([display="true"])').forEach(function (el) {
    if (el.closest('.math-glue')) return;
    var next = neighbour(el, true), prev = neighbour(el, false);
    var after = next ? closing.exec(next.nodeValue) : null;
    var before = prev ? opening.exec(prev.nodeValue) : null;
    if (!after && !before) return;
    var glue = document.createElement('span');
    glue.className = 'math-glue';
    el.parentNode.insertBefore(glue, el);
    glue.appendChild(el);
    // Punctuation right beside the formula moves into the span. Punctuation on the other side
    // of a bold, italic or link boundary stays where it is, held by a word joiner (U+2060).
    if (before) {
      if (prev === glue.previousSibling) {
        prev.nodeValue = prev.nodeValue.slice(0, before.index);
        glue.insertBefore(document.createTextNode(before[0]), el);
      } else {
        glue.insertBefore(document.createTextNode('\u2060'), el);
      }
    }
    if (after) {
      if (next === glue.nextSibling) {
        next.nodeValue = next.nodeValue.slice(after[0].length);
        glue.appendChild(document.createTextNode(after[0]));
      } else {
        glue.appendChild(document.createTextNode('\u2060'));
      }
    }
  });
}
