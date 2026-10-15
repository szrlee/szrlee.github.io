import {C, f, text, line, rect, dot, arrow, path} from '../lib.mjs';

// Hidden space: candidate output vectors w_a, their mean (removed by centering),
// and their covariance C_mu (the directions that tell likely candidates apart).
const W = {tea: [3.0, 2.2], coffee: [3.4, 1.6], water: [2.2, 2.8], milk: [3.8, 2.6]};
const MU = {tea: 0.55, coffee: 0.44, water: 0.006, milk: 0.004};
const O = [90, 470], S = 110;
const px = (v) => [O[0] + S * v[0], O[1] - S * v[1]];

const names = Object.keys(W);
const mean = [0, 1].map((j) => names.reduce((t, n) => t + MU[n] * W[n][j], 0));
let cxx = 0, cxy = 0, cyy = 0;
for (const n of names) {
  const dx = W[n][0] - mean[0], dy = W[n][1] - mean[1];
  cxx += MU[n] * dx * dx; cxy += MU[n] * dx * dy; cyy += MU[n] * dy * dy;
}
const tr = cxx + cyy, det = cxx * cyy - cxy * cxy, l1 = tr / 2 + Math.sqrt(tr * tr / 4 - det), l2 = tr / 2 - Math.sqrt(tr * tr / 4 - det);
const ang = Math.atan2(l1 - cxx, cxy);
const e1 = [Math.cos(ang), Math.sin(ang)], e2 = [-Math.sin(ang), Math.cos(ang)];

let base = text(40, 54, 'hidden space at one prefix', {size: 21, weight: 600});
base += text(40, 82, 'logit of token *a*: *z*_{*a*} = *w*_{*a*} \u00b7 *h*', {size: 19, fill: C.soft});
base += dot(O[0], O[1], 5, {fill: C.ink}) + text(O[0] - 12, O[1] + 6, '0', {size: 16, fill: C.mute, anchor: 'end'});
for (const n of names) {
  const [x, y] = px(W[n]);
  base += arrow(O[0], O[1], x, y, {stroke: C.gray, w: 2, head: 12, op: 0.8});
  base += dot(x, y, 4 + 14 * Math.sqrt(MU[n]), {fill: C.ink, op: 0.8});
  const dx = n === 'coffee' || n === 'milk' ? 16 : -16, an = dx > 0 ? 'start' : 'end';
  base += text(x + dx, y + (n === 'coffee' ? 22 : -8), '*w*_{' + n + '}', {size: 18, anchor: an});
}
base += text(40, 108, 'dot size: how likely the sampler finds the token', {size: 16, fill: C.mute});

const [mx, my] = px(mean);
let meanL = line(O[0], O[1], mx, my, {stroke: C.or, w: 16, op: 0.14, cap: 'round'});
meanL += arrow(O[0], O[1], mx, my, {stroke: C.or, w: 4, head: 16});
meanL += text(612, 150, 'mean *w̄*_{*μ*}: along it every likely', {size: 19, fill: C.or, weight: 600});
meanL += text(612, 176, 'logit rises together; centering', {size: 19, fill: C.or, weight: 600});
meanL += text(612, 202, 'removes this direction', {size: 19, fill: C.or, weight: 600});

const K = 2.2 * S;
let ell = '';
let d = '';
for (let i = 0; i <= 72; i++) {
  const t = 2 * Math.PI * i / 72, a = Math.sqrt(l1) * Math.cos(t), b = Math.sqrt(l2) * Math.sin(t);
  d += (i ? 'L' : 'M') + f(mx + K * (a * e1[0] + b * e2[0])) + ' ' + f(my - K * (a * e1[1] + b * e2[1]));
}
ell += path(d + 'Z', {fill: C.or, op: 0.18, stroke: C.or, w: 2.5});
ell += text(612, 262, '*C*_{*μ*}: the spread of the likely', {size: 19, fill: C.soft, weight: 600});
ell += text(612, 288, 'candidates, nearly a line: two', {size: 19, fill: C.soft});
ell += text(612, 314, 'candidates span one direction', {size: 19, fill: C.soft});

const U = 1.25 * S;
let us = arrow(mx, my, mx + U * e1[0], my - U * e1[1], {stroke: C.bl, w: 4, head: 16});
const sg = e2[0] > 0 ? 1 : -1;
us += arrow(mx, my, mx + sg * U * 0.8 * e2[0], my - sg * U * 0.8 * e2[1], {stroke: C.gray, w: 4, head: 16});
us += text(612, 380, '*u*_{*i*} along it: Fisher *u*_{*i*}\u1d40*C*_{*μ*}*u*_{*i*} > 0', {size: 19, fill: C.bl, weight: 600});
us += text(612, 410, '*u*_{*j*} across it: Fisher 0', {size: 19, fill: C.mute, weight: 600});

export default {
  name: 'subspace',
  aria: 'Hidden space: candidate output vectors, their probability-weighted mean removed by centering, and the thin covariance ellipse of the likely candidates; a direction across it has zero Fisher',
  frames: [{base, layers: [meanL, ell, us]}],
  steps: [
    {k: 0, text: 'Each candidate token has an output vector *w*_{*a*}; its logit is the inner product with the hidden state *h*.'},
    {k: 1, text: 'Moving *h* along the mean vector raises every likely logit together, which changes no probability: centering removes this direction.'},
    {k: 2, text: 'What is left is the spread of the likely candidates, here almost a line: *n* candidates span at most *n* \u2212 1 directions.'},
    {k: 3, text: 'A parameter that moves *h* along that spread has Fisher; one that moves it across has none.'},
  ],
};
