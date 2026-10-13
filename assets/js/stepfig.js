// Step-through figures (see layouts/shortcodes/stepfig.html).
// A figure is a list of frames; each frame reveals its layers by build level.
// Steps are (frame, level) pairs; captions come from the steps JSON.
(function () {
  'use strict';
  var MIN_W = 560, PLAY_MS = 1600;
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
      if (timer) { clearInterval(timer); timer = null; }
      play.textContent = '\u25B6';
      play.setAttribute('aria-label', 'Play');
    }

    function start() {
      if (i === steps.length - 1) go(0);
      play.textContent = '\u275A\u275A';
      play.setAttribute('aria-label', 'Pause');
      timer = setInterval(function () {
        if (i >= steps.length - 1) { stop(); return; }
        go(i + 1);
      }, PLAY_MS);
    }

    prev.addEventListener('click', function () { stop(); go(i - 1); });
    next.addEventListener('click', function () { stop(); go(i + 1); });
    play.addEventListener('click', function () { if (timer) stop(); else start(); });
    fig.addEventListener('keydown', function (e) {
      if (e.key === 'ArrowRight') { stop(); go(i + 1); e.preventDefault(); }
      if (e.key === 'ArrowLeft') { stop(); go(i - 1); e.preventDefault(); }
    });

    function fit() {
      var baseW = parseFloat(stage.style.width);
      var baseH = parseFloat(stage.style.height);
      var scale = Math.min(1, Math.max(viewport.clientWidth, MIN_W) / baseW);
      stage.style.transform = 'scale(' + scale + ')';
      sizer.style.width = (baseW * scale) + 'px';
      sizer.style.height = (baseH * scale) + 'px';
    }
    fit();
    if ('ResizeObserver' in window) new ResizeObserver(fit).observe(viewport);
    else window.addEventListener('resize', fit);

    if ('IntersectionObserver' in window) {
      new IntersectionObserver(function (entries) {
        visible = entries[0].isIntersecting;
        if (!visible) stop();
        syncEmbeds();
      }, {rootMargin: '100px'}).observe(fig);
    } else {
      visible = true;
    }

    render();
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
