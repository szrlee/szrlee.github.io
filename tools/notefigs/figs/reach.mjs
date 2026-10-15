import {C, f, text, line, dot, arrow, path, star, rng, gauss} from '../lib.mjs';

// Score space: the sampler's outputs as a cloud. The update G points toward the
// rewarded samples; the Fisher is the cloud's spread. A target aligned with the
// cloud moves, one off every direction the cloud covers does not.
const O = [400, 280], S = 66, TH = 0.33;
const e1 = [Math.cos(TH), Math.sin(TH)], e2 = [-Math.sin(TH), Math.cos(TH)];
const P = (a, b) => [O[0] + S * (a * e1[0] + b * e2[0]), O[1] - S * (a * e1[1] + b * e2[1])];
const r = rng(23);
const pts = [];
for (let n = 0; n < 80; n++) {
  const a = Math.max(-2.8, Math.min(2.8, 1.5 * gauss(r))), b = 0.26 * gauss(r);
  const A = Math.max(-1, Math.min(1, 0.42 * a + 0.35 * gauss(r)));
  pts.push({a, b, A});
}
const mA = pts.reduce((t, p) => t + p.A, 0) / pts.length;
let ga = 0, gb = 0;
for (const p of pts) { ga += (p.A - mA) * p.a; gb += (p.A - mA) * p.b; }
const gn = Math.hypot(ga, gb), G = [2.6 * ga / gn, 2.6 * gb / gn];
const shade = (A) => {
  const u = (A + 1) / 2;
  const mix = (x, y) => Math.round(x + (y - x) * u);
  const c1 = [0xF6, 0xD9, 0xC2], c2 = [0x8F, 0x2F, 0x08];
  return '#' + [0, 1, 2].map((i) => mix(c1[i], c2[i]).toString(16).padStart(2, '0')).join('');
};

let base = text(40, 52, 'each dot: a response the sampler drew, placed by its centered score *c*', {size: 20, weight: 600});
base += text(40, 80, 'darker: higher reward', {size: 17, fill: C.or});
base += line(O[0] - 330, O[1], O[0] + 400, O[1], {stroke: C.rule, w: 1.5}) + line(O[0], 110, O[0], 470, {stroke: C.rule, w: 1.5});
for (const p of pts) { const [x, y] = P(p.a, p.b); base += dot(x, y, 6, {fill: shade(p.A)}); }

const g1 = P(G[0], G[1]);
const upd = arrow(O[0], O[1], g1[0], g1[1], {stroke: C.ink, w: 4.5, head: 18}) +
  text(g1[0] - 6, g1[1] + 34, 'update *G* = Cov(*A*, *c*)', {size: 19, weight: 600});

let d = '';
for (let i = 0; i <= 72; i++) {
  const t = 2 * Math.PI * i / 72, [x, y] = P(2 * 1.5 * Math.cos(t), 2 * 0.26 * Math.sin(t));
  d += (i ? 'L' : 'M') + f(x) + ' ' + f(y);
}
const fis = path(d + 'Z', {stroke: C.bl, w: 2.5, fill: C.bl, op: 0.06}) +
  text(40, 110, 'Fisher *F*^{*μ*}: the spread of the cloud', {size: 18, fill: C.bl, weight: 600});

const m = P(3.7, 0.15), mp = P(3.7, 0);
const mat = star(m[0], m[1], 15, {fill: C.bl}) + line(m[0], m[1], mp[0], mp[1], {stroke: C.bl, w: 2, dash: '5 5'}) +
  arrow(m[0] - 26, m[1] + 18, m[0] - 26, m[1] - 34, {stroke: C.bl, w: 3.5, head: 14}) +
  text(m[0] + 24, m[1] - 6, '\u201cmatcha\u201d: never drawn,', {size: 18, fill: C.bl, weight: 600}) +
  text(m[0] + 24, m[1] + 18, 'aligned with *G*: lifted', {size: 18, fill: C.bl, weight: 600});

const q = P(0.5, -2.7);
const iso = star(q[0], q[1], 15, {fill: C.gray}) +
  line(q[0] - 26, q[1] + 24, q[0] + 26, q[1] + 24, {stroke: C.gray, w: 3.5, cap: 'round'}) +
  text(q[0] + 28, q[1] - 4, 'off every direction the cloud covers:', {size: 18, fill: C.mute, weight: 600}) +
  text(q[0] + 28, q[1] + 20, '*v*\u1d40*F*^{*μ*}*v* \u2248 0, barely moves', {size: 18, fill: C.mute, weight: 600});

export default {
  name: 'reach',
  aria: 'Score space: sampled responses form an elongated cloud; the update points toward the rewarded ones; an unsampled response aligned with the cloud is lifted, one off the cloud barely moves',
  frames: [{base, layers: [upd, fis, mat, iso]}],
  steps: [
    {k: 0, text: 'Each dot is a response the sampler drew, placed by its centered score; darker dots earned more.'},
    {k: 1, text: 'The update *G* points toward the high-reward samples.'},
    {k: 2, text: 'The Fisher is the spread of this cloud: the directions the sampler\u2019s own outputs cover.'},
    {k: 3, text: 'A response never drawn but aligned with the rewarded ones, like matcha after tea, moves with them.'},
    {k: 4, text: 'A response off every direction the cloud covers barely moves, whatever its reward.'},
  ],
};
