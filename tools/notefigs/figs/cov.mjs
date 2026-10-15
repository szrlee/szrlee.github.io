import {C, text, line, rect, dot, path, poly, rng, gauss} from '../lib.mjs';

// Two parameters with the same score spread (same Fisher): one correlated with the
// learning signal, one not.
const r = rng(7);
const clip = (v, m) => Math.max(-m, Math.min(m, v));
function sample(rho) {
  const pts = [];
  for (let n = 0; n < 56; n++) {
    const s = clip(gauss(r), 2.7);
    pts.push([s, clip(0.93 * (rho * s + Math.sqrt(1 - rho * rho) * gauss(r)), 2.5)]);
  }
  return pts;
}
function stats(pts) {
  const n = pts.length, ms = pts.reduce((t, p) => t + p[0], 0) / n, ma = pts.reduce((t, p) => t + p[1], 0) / n;
  const vs = pts.reduce((t, p) => t + (p[0] - ms) ** 2, 0) / n, va = pts.reduce((t, p) => t + (p[1] - ma) ** 2, 0) / n;
  const cv = pts.reduce((t, p) => t + (p[0] - ms) * (p[1] - ma), 0) / n;
  return {sd: Math.sqrt(vs), sa: Math.sqrt(va), slope: cv / vs};
}
const panels = [
  {x0: 30, title: 'parameter *i*', pts: sample(0.8), lab: '*G* = Cov(*A*, score) > 0', tilt: true},
  {x0: 490, title: 'parameter *j*', pts: sample(0), lab: '*G* = 0', tilt: false},
];
const SC = 58, CY = 246, L = 2.9 * SC;

let base = '', spread = '', cov = '', cap = '';
for (const P of panels) {
  const cx = P.x0 + 220, st = stats(P.pts), m = st.sa / st.sd;
  base += rect(P.x0, 50, 440, 390, {rx: 14, fill: C.card});
  base += text(P.x0 + 24, 86, P.title, {size: 22, weight: 600});
  base += line(cx - 190, CY, cx + 190, CY, {stroke: C.rule, w: 2});
  base += line(cx, CY - 118, cx, CY + 150, {stroke: C.bl, w: 2, dash: '6 6'});
  base += text(cx + 196, CY + 6, 'score', {size: 16, fill: C.mute});
  base += text(cx + 10, CY - 104, '*A*', {size: 20, fill: C.mute});
  base += text(cx + 10, CY + 146, 'mean score 0', {size: 16, fill: C.bl});
  for (const [s, a] of P.pts) base += dot(cx + s * SC, CY - a * SC, 4.6, {fill: C.ink, op: 0.55});
  const by = CY + 172;
  spread += line(cx - st.sd * SC, by, cx + st.sd * SC, by, {stroke: C.bl, w: 4, cap: 'round'});
  spread += line(cx - st.sd * SC, by - 9, cx - st.sd * SC, by + 9, {stroke: C.bl, w: 3});
  spread += line(cx + st.sd * SC, by - 9, cx + st.sd * SC, by + 9, {stroke: C.bl, w: 3});
  spread += text(cx + st.sd * SC + 14, by + 7, 'spread: Fisher *F*', {size: 18, fill: C.bl, weight: 600});
  const k = P.tilt ? st.slope : 0;
  cov += line(cx - L, CY + k * L, cx + L, CY - k * L, {stroke: C.or, w: 5, cap: 'round'});
  cov += text(P.x0 + 24, 118, P.lab, {size: 20, fill: C.or, weight: 600});
  const wedge = poly([[cx, CY], [cx + L, CY - m * L], [cx + L, CY + m * L]]) + 'Z' +
    poly([[cx, CY], [cx - L, CY + m * L], [cx - L, CY - m * L]]) + 'Z';
  cap += path(wedge, {fill: C.soft, op: 0.12, stroke: C.soft, w: 1.5, dash: '6 6'});
}
cap += text(480, 474, 'the same cap for both: |*G*| ≤ std(*A*) · √*F*', {size: 21, anchor: 'middle', weight: 600, fill: C.soft});

export default {
  name: 'cov',
  aria: 'Two scatter plots of a parameter score against the learning signal: equal spread, but only one is correlated with the signal',
  frames: [{base, layers: [spread, cov, cap]}],
  steps: [
    {k: 0, text: 'Each dot is a sample: a parameter\u2019s score against the learning signal *A*, with the score averaging to zero.'},
    {k: 1, text: 'Both parameters have the same spread of scores, hence the same Fisher.'},
    {k: 2, text: 'The expected update is the covariance: positive on the left, zero on the right.'},
    {k: 3, text: 'The update line must stay inside the shaded wedge, set by the Fisher: a large Fisher allows a large update but does not force one.'},
  ],
};
