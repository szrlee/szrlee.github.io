// Slider: two responses with one logit each. Policy gradient on a reward gap versus a
// demonstration of the rewarded response; time to reach probability one half.
// draw and readout are inlined into the page, where these helpers are globals.
import {C, f, text, line, rect, dot, path} from '../lib.mjs';

export default {
  name: 's-amplify',
  title: 'Time to amplify a rare response',
  params: [
    {id: 'eps', label: 'starting probability ε', min: -4, max: -0.5, step: 0.05, value: 0.01, slider: -2,
      toValue: (x) => Math.pow(10, x), toSlider: (v) => Math.log10(v), show: (v) => (v < 0.01 ? v.toPrecision(1) : v.toFixed(2))},
    {id: 'gap', label: 'reward gap Δ', min: 0.1, max: 2, step: 0.1, value: 1, show: (v) => v.toFixed(1)},
  ],
  draw: function (p) {
    function lg(q) { return Math.log(q / (1 - q)); }
    function sg(l) { return 1 / (1 + Math.exp(-l)); }
    var l0 = lg(p.eps), tpg = (Math.exp(-l0) - 2 * l0 - Math.exp(l0)) / (2 * p.gap), tsf = (1 - l0 - Math.exp(l0)) / 2;
    var TM = Math.max(8, Math.min(2.4e4, 1.25 * tpg));
    var X0 = 96, XW = 820, YT = 40, YB = 300, LMIN = Math.min(-1, Math.log10(p.eps) - 0.3);
    function X(t) { return X0 + XW * Math.min(t, TM) / TM; }
    function Y(q) { return YT + (YB - YT) * (Math.log10(q) / LMIN); }
    function run(rate) {
      var l = l0, t = 0, d = 'M' + f(X(0)) + ' ' + f(Y(sg(l))), r, dt;
      while (t < TM && sg(l) < 0.999) {
        r = rate(sg(l));
        dt = Math.min(TM / 400, 0.02 / Math.max(1e-9, r));
        l += r * dt; t += dt;
        d += 'L' + f(X(t)) + ' ' + f(Y(sg(l)));
      }
      return d + 'L' + f(X(TM)) + ' ' + f(Y(sg(l)));
    }
    var o = rect(0, 0, 960, 350, {rx: 16, fill: C.panel});
    o += line(X0, YB, X0 + XW, YB, {stroke: C.rule, w: 2}) + line(X0, YT - 10, X0, YB, {stroke: C.rule, w: 2});
    var k, q;
    for (k = 0; k >= Math.ceil(LMIN); k--) {
      q = Math.pow(10, k);
      o += line(X0 - 6, Y(q), X0, Y(q), {stroke: C.rule, w: 2}) + text(X0 - 10, Y(q) + 5, String(+q.toPrecision(1)), {size: 14, anchor: 'end', fill: C.mute});
    }
    o += line(X0, Y(0.5), X0 + XW, Y(0.5), {stroke: C.soft, w: 1.5, dash: '6 6'});
    o += text(X0 + XW, Y(0.5) - 8, 'one half', {size: 14, anchor: 'end', fill: C.soft});
    o += text(X0 + XW, YB + 28, 'time, 0 to ' + Math.round(TM) + ' (step size \u00d7 steps)', {size: 14, anchor: 'end', fill: C.mute});
    o += path(run(function (s) { return 2 * p.gap * s * (1 - s); }), {stroke: C.bl, w: 3.5});
    o += path(run(function (s) { return 2 * (1 - s); }), {stroke: C.or, w: 3.5});
    if (tpg <= TM) o += dot(X(tpg), Y(0.5), 6, {fill: C.bl});
    o += dot(X(tsf), Y(0.5), 6, {fill: C.or});
    o += text(X0 + 14, 30, 'policy gradient', {size: 16, weight: 600, fill: C.bl}) + text(X0 + 150, 30, 'demonstration', {size: 16, weight: 600, fill: C.or});
    return '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 350" role="img" aria-label="Probability of the rewarded response over time">' + o + '</svg>';
  },
  readout: function (p) {
    var l0 = Math.log(p.eps / (1 - p.eps));
    var tpg = (Math.exp(-l0) - 2 * l0 - Math.exp(l0)) / (2 * p.gap), tsf = (1 - l0 - Math.exp(l0)) / 2;
    return 'Time to reach one half: policy gradient <b>' + (tpg < 100 ? tpg.toFixed(1) : Math.round(tpg)) +
      '</b>, demonstration <b>' + tsf.toFixed(1) + '</b>, a ratio of <b>' + Math.round(tpg / tsf) + '\u00d7</b>. ' +
      'Policy gradient grows like 1/(<i>ε</i>Δ); the demonstration like log(1/<i>ε</i>).';
  },
};
