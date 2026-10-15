import {C, text, line, rect, dot, arrow} from '../lib.mjs';

// At one prefix: the model's probabilities (bars) and the sensitivity of each
// candidate's logit to one parameter (dots). The token Fisher is the p-weighted
// spread of the sensitivities.
const TOK = ['tea', 'coffee', 'water', 'milk', 'juice'];
const XS = [250, 390, 530, 670, 810];
const yk = (k) => 318 - 170 * k;
const BASE = 452, BH = 100;

function state(p, k) {
  const mean = p.reduce((t, v, i) => t + v * k[i], 0);
  const fish = p.reduce((t, v, i) => t + v * (k[i] - mean) ** 2, 0);
  return {p, k, mean, fish};
}
const S0 = state([0.5, 0.4, 0.05, 0.03, 0.02], [0.85, 0.25, 0.6, 0.35, 0.7]);
const S1 = state(S0.p, S0.k.map((v) => v + 0.18));
const S2 = state([0.97, 0.02, 0.005, 0.003, 0.002], S0.k);
const S3 = state(S0.p, [0.62, 0.58, 0.6, 0.35, 0.7]);
const fmt = (v) => v.toPrecision(1);

function frame(S, extra = '') {
  let o = text(40, 54, 'She drank a cup of ___', {size: 24, weight: 600});
  o += text(40, 214, 'sensitivity', {size: 18, fill: C.mute}) + text(40, 238, '*κ*_{*i*,*a*}', {size: 20, fill: C.mute});
  o += text(40, 420, 'probability', {size: 18, fill: C.mute}) + text(40, 444, '*p*(*a*)', {size: 20, fill: C.mute});
  o += line(200, BASE, 880, BASE, {stroke: C.rule, w: 2});
  S.p.forEach((v, i) => {
    o += rect(XS[i] - 30, BASE - BH * v, 60, BH * v, {rx: 4, fill: C.bl, op: 0.85});
    o += text(XS[i], BASE - BH * v - 8, v < 0.01 ? v.toFixed(3) : v.toFixed(2), {size: 15, anchor: 'middle', fill: C.soft});
    o += text(XS[i], BASE + 30, TOK[i], {size: 18, anchor: 'middle'});
  });
  return o + extra;
}
const dots = (S) => S.p.map((v, i) => dot(XS[i], yk(S.k[i]), 6 + 16 * Math.sqrt(v), {fill: C.bl, op: 0.85})).join('');
const mean = (S) => line(200, yk(S.mean), 880, yk(S.mean), {stroke: C.bl, w: 2.5, dash: '9 7'}) +
  text(886, yk(S.mean) + 6, 'mean', {size: 17, fill: C.bl, weight: 600});
const gaps = (S) => S.p.map((v, i) => line(XS[i], yk(S.mean), XS[i], yk(S.k[i]), {stroke: C.ink, w: 2 + 12 * v, op: 0.75})).join('') +
  text(920, 54, '*f*_{*i*} ≈ ' + fmt(S.fish), {size: 24, weight: 600, anchor: 'end'});

const ghost = S0.p.map((v, i) => dot(XS[i], yk(S0.k[i]), 6 + 16 * Math.sqrt(v), {fill: C.gray, op: 0.35}) +
  arrow(XS[i] + 26, yk(S0.k[i]), XS[i] + 26, yk(S1.k[i]) + 4, {stroke: C.or, w: 2.5, head: 12})).join('');

export default {
  name: 'tokfisher',
  aria: 'At one prefix, bars of the model probabilities and dots for the sensitivity of each candidate logit; the token Fisher is their probability-weighted spread',
  frames: [
    {base: frame(S0) + dots(S0), layers: [mean(S0), gaps(S0)]},
    {base: frame(S1, ghost + text(560, 54, 'every *κ* raised by the same amount', {size: 20, fill: C.or, anchor: 'middle'})) + mean(S1) + gaps(S1) + dots(S1), layers: []},
    {base: frame(S2, text(560, 54, 'a confident model', {size: 20, fill: C.or, anchor: 'middle'})) + mean(S2) + gaps(S2) + dots(S2), layers: []},
    {base: frame(S3, text(560, 54, 'tea and coffee react alike', {size: 20, fill: C.or, anchor: 'middle'})) + mean(S3) + gaps(S3) + dots(S3), layers: []},
  ],
  steps: [
    {f: 0, k: 0, text: 'After \u201cShe drank a cup of\u201d: the model\u2019s probabilities (bars), and how strongly *θ*_{*i*} moves each candidate\u2019s logit (dots).'},
    {f: 0, k: 1, text: 'The token score is the drawn token\u2019s sensitivity minus the *p*-weighted mean.'},
    {f: 0, k: 2, text: 'The token Fisher is the *p*-weighted spread of the sensitivities around that mean.'},
    {f: 1, k: 0, text: 'Raising every logit by the same amount changes no probability, so the Fisher does not change.'},
    {f: 2, k: 0, text: 'A confident model leaves little spread: the Fisher is ten times smaller.'},
    {f: 3, k: 0, text: 'So do likely candidates that react alike: over thirty times smaller.'},
  ],
};
