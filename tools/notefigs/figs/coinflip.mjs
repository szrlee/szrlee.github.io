import {C, f, text, line, rect, dot, path, rng} from '../lib.mjs';

// Output layer: row c's Fisher is a coin flip, mu(c)(1 - mu(c)); summed over a
// response it tracks how often the sampler emits c.
const P = {tea: 0.5, coffee: 0.4, rest: 0.1};
const B = 300, BH = 300;
let bars = rect(24, 24, 420, 330, {rx: 14, fill: C.card});
bars += text(44, 60, 'sampler *μ* after \u201cShe drank a cup of\u201d', {size: 19, weight: 600, fill: C.or});
bars += line(44, B, 424, B, {stroke: C.rule, w: 2});
[['tea', P.tea, 84], ['coffee', P.coffee, 184]].forEach(([n, v, x]) => {
  bars += rect(x, B - BH * v * 0.62, 64, BH * v * 0.62, {rx: 6, fill: C.or});
  bars += text(x + 32, B + 28, n, {size: 18, anchor: 'middle'});
  bars += text(x + 32, B - BH * v * 0.62 - 10, v.toFixed(1), {size: 17, anchor: 'middle', fill: C.soft});
});
for (let i = 0; i < 34; i++) { const x = 292 + i * 4; bars += line(x, B, x, B - 2 - 2 * ((i * 7) % 3), {stroke: C.or, w: 2, op: 0.7}); }
bars += text(358, B + 28, '100,000 others', {size: 18, anchor: 'middle'});
bars += text(358, B - 24, '0.1 in total', {size: 17, anchor: 'middle', fill: C.soft});

const X0 = 510, XW = 400, Y0 = 300, YH = 760;
let axes = rect(468, 24, 468, 330, {rx: 14, fill: C.card});
axes += line(X0, Y0, X0 + XW, Y0, {stroke: C.rule, w: 2}) + line(X0, Y0, X0, Y0 - 0.27 * YH, {stroke: C.rule, w: 2});
axes += text(X0, Y0 + 26, '0', {size: 16, anchor: 'middle', fill: C.mute}) + text(X0 + XW, Y0 + 26, '1', {size: 16, anchor: 'middle', fill: C.mute});
axes += text(X0 + XW / 2, Y0 + 26, '*μ*(*c*)', {size: 18, anchor: 'middle', fill: C.mute});

let d = '';
for (let i = 0; i <= 100; i++) d += (i ? 'L' : 'M') + f(X0 + XW * i / 100) + ' ' + f(Y0 - YH * (i / 100) * (1 - i / 100));
const curve = path(d, {stroke: C.bl, w: 3.5}) + text(490, 60, 'row *c*: Fisher \u221d *μ*(*c*)(1 \u2212 *μ*(*c*))', {size: 19, weight: 600, fill: C.bl});

let pts = dot(X0 + 3, Y0 - 1, 8, {fill: C.or, op: 0.65}) + text(X0 + 16, Y0 - 14, '100,000 others', {size: 16, fill: C.soft});
[['tea', P.tea, 14, 'start'], ['coffee', P.coffee, -14, 'end']].forEach(([n, v, dx, an]) => {
  const px = X0 + XW * v, py = Y0 - YH * v * (1 - v);
  pts += dot(px, py, 9, {fill: C.or}) + text(px + dx, py + 30, n, {size: 17, anchor: an});
});

const ft = P.tea * (1 - P.tea), fc = P.coffee * (1 - P.coffee), tot = ft + fc + P.rest;
const SX = 24, SW = 912, SY = 404;
let share = text(SX, SY - 10, 'share of the output layer\u2019s Fisher at this prefix', {size: 17, fill: C.soft});
share += rect(SX, SY, SW * ft / tot, 40, {fill: C.or}) + rect(SX + SW * ft / tot, SY, SW * fc / tot, 40, {fill: C.orMid}) +
  rect(SX + SW * (ft + fc) / tot, SY, SW * P.rest / tot, 40, {fill: C.track});
share += text(SX + 14, SY + 27, 'tea + coffee: 0.49 of 0.59 = ' + Math.round(100 * (ft + fc) / tot) + '%', {size: 19, weight: 600, fill: '#fff'});
share += text(SX + SW - 12, SY + 27, 'all others', {size: 17, anchor: 'end', fill: C.soft});

// Frame 1: the rows of the output layer; lit rows are the tokens this task emits.
const r = rng(11), COLS = 48, ROWS = 9, CW = 18;
let grid = text(480, 62, 'rows of the output layer, one square per token', {size: 21, weight: 600, anchor: 'middle'});
grid += text(480, 92, 'lit: tokens the sampler emits on this task; darker: more often', {size: 18, anchor: 'middle', fill: C.or});
const GX = 480 - COLS * CW / 2;
for (let row = 0; row < ROWS; row++) {
  for (let col = 0; col < COLS; col++) {
    const u = r(), lit = u < 0.07, w = lit ? 0.25 + 0.75 * Math.pow(r(), 2.5) : 0;
    grid += rect(GX + col * CW, 128 + row * CW, CW - 3, CW - 3, {rx: 3, fill: lit ? C.or : C.track, op: lit ? f(0.3 + 0.7 * w) : 1});
  }
}
grid += text(480, 330, 'a row\u2019s Fisher, summed over a response, is at most \u2016*h*\u2016\u00b2 times the expected count of its token', {size: 18, anchor: 'middle', fill: C.soft});
grid += text(480, 360, 'a vocabulary has over a hundred thousand tokens; a task uses a few thousand', {size: 18, anchor: 'middle', fill: C.soft});

export default {
  name: 'coinflip',
  aria: 'Output-layer Fisher at one prefix: tea and coffee rows sit near the top of the coin-flip curve and hold most of the Fisher; across a task only the rows of emitted tokens light up',
  frames: [
    {base: bars + axes, layers: [curve, pts, share]},
    {base: grid, layers: []},
  ],
  steps: [
    {f: 0, k: 0, text: 'At one prefix the sampler puts 0.5 on tea, 0.4 on coffee, and 0.1 on everything else.'},
    {f: 0, k: 1, text: 'For row *c* the next token is a coin flip, *c* or not *c*, so its Fisher follows *μ*(1 \u2212 *μ*).'},
    {f: 0, k: 2, text: 'Tea and coffee sit near the top of the curve; the hundred thousand other rows sit at zero.'},
    {f: 0, k: 3, text: 'So the tea and coffee rows hold 83% of the output layer\u2019s Fisher at this prefix.'},
    {f: 1, k: 0, text: 'Summed over a response, a row\u2019s Fisher tracks how often the sampler emits its token: rows the task never uses get almost nothing.'},
  ],
};
