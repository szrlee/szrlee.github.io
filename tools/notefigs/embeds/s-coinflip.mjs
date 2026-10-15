// Slider: the output-layer Fisher at one prefix, as the sampler's probabilities of
// tea and coffee move. The remaining mass is spread over 100,000 other tokens.
// draw and readout are inlined into the page, where these helpers are globals.
import {C, f, text, line, rect, dot, path} from '../lib.mjs';

export default {
  name: 's-coinflip',
  title: 'Output-layer Fisher at one prefix',
  params: [
    {id: 'tea', label: 'μ(tea)', min: 0, max: 1, step: 0.01, value: 0.5, show: (v) => v.toFixed(2)},
    {id: 'coffee', label: 'μ(coffee)', min: 0, max: 1, step: 0.01, value: 0.4, show: (v) => v.toFixed(2)},
  ],
  clamp: function (p, src) {
    if (p.tea + p.coffee > 1) {
      if (src === 'coffee') p.tea = Math.round((1 - p.coffee) * 100) / 100;
      else p.coffee = Math.round((1 - p.tea) * 100) / 100;
    }
    return p;
  },
  draw: function (p) {
    var N = 100000, rest = Math.max(0, 1 - p.tea - p.coffee);
    var ft = p.tea * (1 - p.tea), fc = p.coffee * (1 - p.coffee), fr = rest - rest * rest / N, tot = ft + fc + fr;
    var o = rect(0, 0, 960, 380, {rx: 16, fill: C.panel}), i, x;
    o += rect(24, 20, 420, 282, {rx: 14, fill: C.card}) + rect(468, 20, 468, 282, {rx: 14, fill: C.card});
    o += text(44, 54, 'sampler *μ* after \u201cShe drank a cup of\u201d', {size: 18, weight: 600, fill: C.or});
    var B = 262, H = 170;
    o += line(44, B, 424, B, {stroke: C.rule, w: 2});
    [['tea', p.tea, 84], ['coffee', p.coffee, 184]].forEach(function (t) {
      o += rect(t[2], B - H * t[1], 64, H * t[1], {rx: 6, fill: C.or});
      o += text(t[2] + 32, B + 26, t[0], {size: 17, anchor: 'middle'});
      o += text(t[2] + 32, B - H * t[1] - 10, t[1].toFixed(2), {size: 16, anchor: 'middle', fill: C.soft});
    });
    for (i = 0; i < 34; i++) { x = 292 + i * 4; o += line(x, B, x, B - 2 - 2 * ((i * 7) % 3), {stroke: C.or, w: 2, op: 0.7}); }
    o += text(358, B + 26, '100,000 others', {size: 17, anchor: 'middle'});
    o += text(358, B - 24, rest.toFixed(2) + ' in total', {size: 16, anchor: 'middle', fill: C.soft});
    o += text(488, 54, 'row *c*: Fisher \u221d *μ*(*c*)(1 \u2212 *μ*(*c*))', {size: 18, weight: 600, fill: C.bl});
    var X0 = 510, XW = 400, Y0 = 262, YH = 640;
    o += line(X0, Y0, X0 + XW, Y0, {stroke: C.rule, w: 2}) + line(X0, Y0, X0, Y0 - 0.27 * YH, {stroke: C.rule, w: 2});
    var d = '';
    for (i = 0; i <= 100; i++) d += (i ? 'L' : 'M') + f(X0 + XW * i / 100) + ' ' + f(Y0 - YH * (i / 100) * (1 - i / 100));
    o += path(d, {stroke: C.bl, w: 3});
    o += text(X0, Y0 + 24, '0', {size: 15, anchor: 'middle', fill: C.mute}) + text(X0 + XW, Y0 + 24, '1', {size: 15, anchor: 'middle', fill: C.mute});
    o += text(X0 + XW / 2, Y0 + 24, '*μ*(*c*)', {size: 16, anchor: 'middle', fill: C.mute});
    o += dot(X0 + 3, Y0 - 1, 7, {fill: C.or, op: 0.6}) + text(X0 + 14, Y0 - 12, 'others', {size: 14, fill: C.soft});
    [['tea', p.tea, ft], ['coffee', p.coffee, fc]].forEach(function (t, j) {
      var px = X0 + XW * t[1], py = Y0 - YH * t[2];
      o += dot(px, py, 8, {fill: C.or});
      o += text(px + (j ? -12 : 12), py + 26, t[0], {size: 16, anchor: j ? 'end' : 'start'});
    });
    var SX = 24, SW = 912, SY = 324;
    o += text(SX, SY - 8, 'share of the output layer\u2019s Fisher', {size: 15, fill: C.soft});
    var wt = SW * ft / tot, wc = SW * fc / tot;
    o += rect(SX, SY, wt, 30, {fill: C.or}) + rect(SX + wt, SY, wc, 30, {fill: C.orMid}) + rect(SX + wt + wc, SY, SW - wt - wc, 30, {fill: C.track});
    if (wt + wc > 120) o += text(SX + 10, SY + 21, 'tea + coffee ' + Math.round(100 * (ft + fc) / tot) + '%', {size: 16, weight: 600, fill: '#fff'});
    return '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 380" role="img" aria-label="Output-layer Fisher at one prefix">' + o + '</svg>';
  },
  readout: function (p) {
    var N = 100000, rest = Math.max(0, 1 - p.tea - p.coffee);
    var ft = p.tea * (1 - p.tea), fc = p.coffee * (1 - p.coffee), fr = rest - rest * rest / N, tot = ft + fc + fr;
    return 'Row Fisher per \u2016<i>h</i>\u2016\u00b2: tea <b>' + ft.toFixed(3) + '</b>, coffee <b>' + fc.toFixed(3) +
      '</b>, all other rows <b>' + fr.toFixed(3) + '</b>, of <b>' + tot.toFixed(3) + '</b> in total. Tea and coffee hold <b>' +
      (tot > 0 ? Math.round(100 * (ft + fc) / tot) : 0) + '%</b>.';
  },
};
