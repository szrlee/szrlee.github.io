import {C, f, text, line, rect, path, rng, gauss} from '../lib.mjs';

// Running sums of per-position score terms. Centered: zero-mean terms, the sum
// spreads like sqrt(T). Uncentered: the same terms plus a shift m, the sum drifts like T.
const T = 60, M = 0.35, NW = 5;
const X0 = 110, XW = 790, Y0 = 290, SY = 6.0;
const X = (t) => X0 + XW * t / T, Y = (v) => Y0 - SY * v;
const r = rng(5);
const walks = [];
for (let w = 0; w < NW; w++) {
  const z = [0];
  for (let t = 1; t <= T; t++) z.push(z[t - 1] + gauss(r));
  walks.push(z);
}
const poly = (z, m) => z.map((v, t) => (t ? 'L' : 'M') + f(X(t)) + ' ' + f(Y(v + m * t))).join('');

let base = line(X0, Y0, X0 + XW, Y0, {stroke: C.rule, w: 2}) + line(X0, 130, X0, 450, {stroke: C.rule, w: 2});
base += text(X0 + XW / 2, 488, 'position *t* along the response \u2192', {size: 17, anchor: 'middle', fill: C.mute});
base += text(X0 - 12, 300, 'sum', {size: 17, fill: C.mute, anchor: 'end'});
let env = '';
for (let t = 0; t <= T; t++) env += (t ? 'L' : 'M') + f(X(t)) + ' ' + f(Y(2 * Math.sqrt(t)));
for (let t = T; t >= 0; t--) env += 'L' + f(X(t)) + ' ' + f(Y(-2 * Math.sqrt(t)));
base += path(env + 'Z', {fill: C.bl, op: 0.1});
for (const z of walks) base += path(poly(z, 0), {stroke: C.bl, w: 2, sop: 0.7});
base += text(X0 + XW - 4, 458, 'centered: spreads like \u221a*T*', {size: 18, anchor: 'end', fill: C.bl, weight: 600});

let drift = line(X(0), Y(0), X(T), Y(M * T), {stroke: C.or, w: 2.5, dash: '9 7'});
for (const z of walks) drift += path(poly(z, M), {stroke: C.or, w: 2, sop: 0.75});
drift += text(X0 + XW - 4, 46, 'uncentered: drifts like *mT*', {size: 18, anchor: 'end', fill: C.or, weight: 600});

let moments = rect(130, 26, 380, 84, {rx: 12, fill: C.card});
moments += text(150, 60, 'second moment, centered: \u2248 *Tσ*\u00b2', {size: 19, fill: C.bl, weight: 600});
moments += text(150, 92, 'uncentered: \u2248 *Tσ*\u00b2 + *T*\u00b2*m*\u00b2', {size: 19, fill: C.or, weight: 600});

// Frame 1: one output row under a demonstration the model gives probability 1%.
const p = 0.01, capv = Math.sqrt(p * (1 - p)), push = 1 - p, BY = 420, BH = 280;
let row = text(480, 60, 'output row of a demonstrated token *y*\u22c6 with *p*(*y*\u22c6) = 0.01', {size: 21, weight: 600, anchor: 'middle'});
row += line(220, BY, 740, BY, {stroke: C.rule, w: 2});
row += rect(290, BY - BH * capv, 120, BH * capv, {rx: 6, fill: C.bl});
row += rect(550, BY - BH * push, 120, BH * push, {rx: 6, fill: C.or});
row += text(350, BY - BH * capv - 14, capv.toFixed(2), {size: 18, anchor: 'middle', fill: C.soft});
row += text(610, BY - BH * push - 14, push.toFixed(2), {size: 18, anchor: 'middle', fill: C.soft});
row += text(350, BY + 30, 'on-policy cap: \u221a(*p*(1 \u2212 *p*))', {size: 18, anchor: 'middle', fill: C.bl, weight: 600});
row += text(610, BY + 30, 'SFT push: 1 \u2212 *p*', {size: 18, anchor: 'middle', fill: C.or, weight: 600});
row += text(480, BY + 62, 'both per \u2016*h*\u2016 and per unit of signal', {size: 16, anchor: 'middle', fill: C.mute});

export default {
  name: 'meanshift',
  aria: 'Running sums of per-position terms: centered sums spread like the square root of length, uncentered sums drift linearly; and one output row where a demonstration pushes ten times the on-policy cap',
  frames: [
    {base, layers: [drift, moments]},
    {base: row, layers: []},
  ],
  steps: [
    {f: 0, k: 0, text: 'With centering, each position adds a zero-mean term, and the running sum spreads like \u221a*T*.'},
    {f: 0, k: 1, text: 'Without it, every position also adds the same shift *m*, and the sum drifts like *T*.'},
    {f: 0, k: 2, text: 'So the second moment grows like *Tσ*\u00b2 with centering and like *Tσ*\u00b2 + *T*\u00b2*m*\u00b2 without.'},
    {f: 1, k: 0, text: 'In the output layer, a demonstrated token the model gives 1% gets a push ten times its row\u2019s on-policy cap.'},
  ],
};
