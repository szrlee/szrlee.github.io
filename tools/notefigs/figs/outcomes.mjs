import {C, text, line, arrow, rect, dot} from '../lib.mjs';

// One response of two tokens fixes the token scores psi_1 and psi_2; the advantages
// only choose the combination u = A_1 psi_1 + A_2 psi_2. Counting the distinct
// updates as the reward varies counts the bits that can get through.
const S = 56, PSI1 = [1.0, 0.25], PSI2 = [0.2, 0.9], OY = 300;
const L = {x0: 30, ox: 240}, R = {x0: 490, ox: 710};
const at = (P, a1, a2) => [P.ox + S * (a1 * PSI1[0] + a2 * PSI2[0]), OY - S * (a1 * PSI1[1] + a2 * PSI2[1])];
// Unit normal to psi_1 + psi_2, pointing up and to the left on screen.
const NX = -0.692, NY = -0.722;

function card(P, title, formula) {
  return rect(P.x0, 64, 440, 412, {rx: 14, fill: C.card}) +
    text(P.x0 + 24, 100, title, {size: 21, weight: 600}) +
    text(P.x0 + 24, 128, formula, {size: 18, fill: C.soft});
}

function scores(P, l1, l2) {
  const a = at(P, 1, 0), b = at(P, 0, 1);
  return arrow(P.ox, OY, a[0], a[1], {stroke: C.gray, w: 2.5}) + arrow(P.ox, OY, b[0], b[1], {stroke: C.gray, w: 2.5}) +
    text(a[0] + l1[0], a[1] + l1[1], '*ψ*_{1}', {size: 18, fill: C.mute, anchor: l1[2]}) +
    text(b[0] + l2[0], b[1] + l2[1], '*ψ*_{2}', {size: 18, fill: C.mute, anchor: l2[2]});
}
const arrowsL = scores(L, [6, 22, 'start'], [-8, -4, 'end']);
const arrowsR = scores(R, [0, 28, 'middle'], [3, 30, 'start']);
const head = (s) => text(480, 44, s, {size: 19, anchor: 'middle', fill: C.soft, weight: 600});
const foot = (P, s, fill) => text(P.x0 + 220, 458, s, {size: 19, weight: 600, fill, anchor: 'middle'});
const axis = (P, fill) => line(...at(P, -2.2, -2.2), ...at(P, 2.2, 2.2), {stroke: fill, w: 2, dash: '6 6', op: 0.6});

// Frame 0: a reward in {-1, 0, +1} at each token.
const base0 = head('a reward in {−1, 0, +1} at each of two tokens;  update *u* = *A*_{1}*ψ*_{1} + *A*_{2}*ψ*_{2}') +
  card(L, 'one scalar advantage', '*A* = *r*_{1} + *r*_{2} at both tokens') +
  card(R, 'per-token advantages', '*A*_{1} = *r*_{1} + *r*_{2},  *A*_{2} = *r*_{2}') + arrowsL + arrowsR;

let scalar = axis(L, C.or);
[1, 2, 3, 2, 1].forEach((m, i) => {
  const k = i - 2, [x, y] = at(L, k, k), r = 6 + 3 * m;
  scalar += dot(x, y, r, {fill: C.or, op: 0.9});
  scalar += text(x + NX * (r + 10), y + NY * (r + 10), '×' + m, {size: 16, fill: C.or, weight: 600, anchor: 'end'});
});
scalar += text(262, 352, 'three patterns, one update:', {size: 15, fill: C.soft});
scalar += text(262, 374, '(0, 0)  (+1, −1)  (−1, +1)', {size: 15, fill: C.soft});
scalar += foot(L, '9 patterns → 5 updates: 2.2 bits', C.or);

let token = '';
for (const r2 of [-1, 0, 1]) for (const r1 of [-1, 0, 1]) {
  const [x, y] = at(R, r1 + r2, r2);
  token += dot(x, y, 9, {fill: C.bl, op: 0.9});
}
token += text(698, 292, '(0, 0)', {size: 15, fill: C.soft, anchor: 'end'});
token += text(721, 230, '(−1, +1)', {size: 15, fill: C.soft, anchor: 'middle'});
token += text(712, 356, '(+1, −1)', {size: 15, fill: C.soft});
token += foot(R, '9 patterns → 9 updates: 3.2 bits', C.bl);

// Frame 1: one pass/fail reward at the end.
const base1 = head('one pass/fail reward at the end:  *r*_{1} = 0,  *r*_{2} = ±1') +
  card(L, 'one scalar advantage', '*A* = *r*_{2} at both tokens') +
  card(R, 'per-token advantages', '*A*_{1} = *A*_{2} = *r*_{2}') + arrowsL + arrowsR;

function two(P, fill) {
  let s = axis(P, fill);
  for (const k of [-1, 1]) {
    const [x, y] = at(P, k, k);
    s += dot(x, y, 10, {fill, op: 0.9});
    s += text(x + NX * 20, y + NY * 20, '*r*_{2} = ' + (k > 0 ? '+1' : '−1'), {size: 16, fill, weight: 600, anchor: 'end'});
  }
  return s;
}
const outL = two(L, C.or) + foot(L, '2 patterns → 2 updates: 1 bit', C.or);
const outR = two(R, C.bl) + foot(R, 'the same 2 updates: 1 bit', C.bl);

export default {
  name: 'outcomes',
  aria: 'Possible updates after a two-token response as the reward varies: one scalar advantage puts nine reward patterns on five points along a line, per-token advantages give nine distinct points, and with one outcome reward both give the same two points',
  frames: [{base: base0, layers: [scalar, token]}, {base: base1, layers: [outL, outR]}],
  steps: [
    {f: 0, k: 0, text: 'A response of two tokens fixes the token scores *ψ*_{1} and *ψ*_{2} before any reward arrives; the advantages only choose a combination of them.'},
    {f: 0, k: 1, text: 'One scalar advantage makes every update a multiple of *ψ*_{1} + *ψ*_{2}. The nine reward patterns (*r*_{1}, *r*_{2}) land on five points: 2.2 of 3.2 bits, if all nine are equally likely.'},
    {f: 0, k: 2, text: 'Per-token advantages give each pattern its own update, and all 3.2 bits get through.'},
    {f: 1, k: 1, text: 'With one pass/fail reward at the end there are only two patterns.'},
    {f: 1, k: 2, text: 'Both forms then make the same two updates: 1 bit either way.'},
  ],
};
