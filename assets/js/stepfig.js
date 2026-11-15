// Step-through figures (see layouts/shortcodes/stepfig.html).
// A figure is a list of frames; each frame reveals its layers by build level.
// Steps are (frame, level) pairs; captions come from the steps JSON.
// A figure plays by itself once, when most of it has scrolled into view; touching its controls
// ends that for good, and "reduce motion" turns it off.
(function () {
  'use strict';
  var MIN_W = 560;
  var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  function init(fig) {
    var steps = JSON.parse(fig.querySelector('.stepfig-steps').textContent);
    var frames = Array.prototype.slice.call(fig.querySelectorAll('.stepfig-frame'));
    var viewport = fig.querySelector('.stepfig-viewport');
    var sizer = fig.querySelector('.stepfig-sizer');
    var stage = fig.querySelector('.stepfig-stage');
    var caption = fig.querySelector('.stepfig-caption');
    var note = fig.querySelector('.stepfig-note');
    var count = fig.querySelector('.stepfig-count');
    var prev = fig.querySelector('.stepfig-prev');
    var next = fig.querySelector('.stepfig-next');
    var play = fig.querySelector('.stepfig-play');
    var i = 0, timer = null, visible = false;
    var auto = steps.length > 1 && !reduce;   // still owes its one automatic run
    var inView = false;

    fig.classList.add('js');
    fig.tabIndex = 0;

    function captionHtml(s) {
      return (s.title ? '<p class="stepfig-title">' + escapeHtml(s.title) + '</p>' : '') + s.text.join('');
    }

    function syncEmbeds() {
      var s = steps[i];
      frames.forEach(function (fr, f) {
        fr.querySelectorAll('iframe[data-src]').forEach(function (ifr) {
          var want = visible && !reduce && f === s.f && (!ifr.dataset.step || +ifr.dataset.step <= s.k);
          if (want && !ifr.getAttribute('src')) ifr.setAttribute('src', ifr.dataset.src);
          if (!want && ifr.getAttribute('src')) ifr.removeAttribute('src');
        });
      });
    }

    function render() {
      var s = steps[i];
      frames.forEach(function (fr, f) {
        var on = f === s.f;
        fr.classList.toggle('is-on', on);
        fr.setAttribute('aria-hidden', on ? 'false' : 'true');
        fr.querySelectorAll('[data-step]').forEach(function (el) {
          el.classList.toggle('is-on', on && +el.dataset.step <= s.k);
        });
      });
      caption.innerHTML = captionHtml(s);
      note.textContent = s.note || '';
      count.textContent = (i + 1) + ' / ' + steps.length;
      prev.disabled = i === 0;
      next.disabled = i === steps.length - 1;
      syncEmbeds();
    }

    function go(j) {
      i = Math.max(0, Math.min(steps.length - 1, j));
      render();
    }

    function stop() {
      if (timer) { clearTimeout(timer); timer = null; }
      play.textContent = '\u25B6';
      play.setAttribute('aria-label', 'Play');
      caption.setAttribute('aria-live', 'polite');
    }

    // A step stays up long enough to take in its caption: 1.6 s for a short one, up to 4.5 s.
    function dwell() {
      return Math.max(1600, Math.min(4500, 1000 + 18 * caption.textContent.length));
    }

    function tick() {
      timer = setTimeout(function () {
        go(i + 1);
        if (i >= steps.length - 1) { auto = false; stop(); } else tick();
      }, dwell());
    }

    function start() {
      if (i === steps.length - 1) go(0);
      play.textContent = '\u275A\u275A';
      play.setAttribute('aria-label', 'Pause');
      caption.setAttribute('aria-live', 'off');
      tick();
    }

    // The reader took over: no automatic run from here on.
    function take() { auto = false; stop(); }

    function autoplay() {
      if (auto && inView && !timer) start();
    }

    prev.addEventListener('click', function () { take(); go(i - 1); });
    next.addEventListener('click', function () { take(); go(i + 1); });
    play.addEventListener('click', function () { if (timer) take(); else { auto = false; start(); } });
    fig.addEventListener('keydown', function (e) {
      if (e.key === 'ArrowRight') { take(); go(i + 1); e.preventDefault(); }
      if (e.key === 'ArrowLeft') { take(); go(i - 1); e.preventDefault(); }
    });

    function fit() {
      var baseW = parseFloat(stage.style.width);
      var baseH = parseFloat(stage.style.height);
      var minW = parseFloat(fig.getAttribute('data-min')) || MIN_W;
      var scale = Math.min(1, Math.max(viewport.clientWidth, minW) / baseW);
      stage.style.transform = 'scale(' + scale + ')';
      sizer.style.width = (baseW * scale) + 'px';
      sizer.style.height = (baseH * scale) + 'px';
      reserve();
    }

    // Room for the tallest caption and note, so the page below does not move from step to step.
    function reserve() {
      var shown = caption.innerHTML, shownNote = note.textContent, tallest = 0, tallestNote = 0;
      caption.style.minHeight = '';
      note.style.minHeight = '';
      note.style.display = '';
      steps.forEach(function (s) {
        caption.innerHTML = captionHtml(s);
        tallest = Math.max(tallest, Math.ceil(caption.getBoundingClientRect().height));
        note.textContent = s.note || '';
        tallestNote = Math.max(tallestNote, Math.ceil(note.getBoundingClientRect().height));
      });
      caption.innerHTML = shown;
      note.textContent = shownNote;
      if (tallest) caption.style.minHeight = tallest + 'px';
      // An empty note is hidden by the stylesheet; keep its line when other steps have one.
      if (tallestNote) { note.style.minHeight = tallestNote + 'px'; note.style.display = 'block'; }
    }

    render();
    fit();
    if ('ResizeObserver' in window) new ResizeObserver(fit).observe(viewport);
    else window.addEventListener('resize', fit);
    window.addEventListener('load', reserve);
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(reserve);

    if ('IntersectionObserver' in window) {
      new IntersectionObserver(function (entries) {
        visible = entries[0].isIntersecting;
        if (!visible) stop();
        syncEmbeds();
      }, {rootMargin: '100px'}).observe(fig);
      // In view: 60% of the figure shows, or it fills 60% of the window (figures taller than it).
      new IntersectionObserver(function (entries) {
        var e = entries[entries.length - 1];
        var room = e.rootBounds ? e.rootBounds.height : window.innerHeight;
        inView = e.isIntersecting && (e.intersectionRatio >= 0.6 || e.intersectionRect.height >= 0.6 * room);
        autoplay();
      }, {threshold: [0, 0.2, 0.4, 0.6, 0.8, 1]}).observe(fig);
    } else {
      visible = true;
      syncEmbeds();
    }
  }

  function escapeHtml(t) {
    return t.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  function boot() {
    document.querySelectorAll('.stepfig').forEach(init);
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
})();
