import {C, text, line, rect, path} from '../lib.mjs';

// Along a response: one token score per position, each with mean zero given its
// prefix, so in the square of the sum only the diagonal survives.
const TOK = ['tea', 'with', 'milk', 'and', 'honey'];
const TX = (i) => 360 + i * 116;
const FS = [0.9, 0.55, 0.7, 0.35, 0.6];

function bell(cx, cy, w, h) {
  let d = '';
  for (let i = 0; i <= 40; i++) {
    const u = -2.4 + 4.8 * i / 40;
    d += (i ? 'L' : 'M') + (cx + u * w / 4.8).toFixed(1) + ' ' + (cy - h * Math.exp(-u * u / 2)).toFixed(1);
  }
  return path(d, {stroke: C.bl, w: 2.5, fill: C.bl, op: 0.12});
}

let base = rect(40, 34, 270, 50, {rx: 10, fill: C.track}) + text(175, 66, 'She drank a cup of', {size: 20, anchor: 'middle', fill: C.soft});
TOK.forEach((t, i) => {
  base += rect(TX(i) - 48, 34, 96, 50, {rx: 10, fill: C.card, stroke: C.bl, sw: 2});
  base += text(TX(i), 66, t, {size: 20, anchor: 'middle'});
  base += text(TX(i) - 52, 30, String(i + 1), {size: 13, fill: C.mute});
  base += bell(TX(i), 158, 92, 48);
  base += text(TX(i), 186, '*s*^{(' + (i + 1) + ')}', {size: 18, anchor: 'middle', fill: C.bl});
});
base += text(40, 140, 'one token score', {size: 18, fill: C.mute}) + text(40, 164, 'per position', {size: 18, fill: C.mute});

let mean = '';
TOK.forEach((t, i) => { mean += line(TX(i), 104, TX(i), 166, {stroke: C.or, w: 3}); });
mean += text(40, 206, 'mean 0 given its prefix', {size: 18, fill: C.or, weight: 600});

const G0x = 520, G0y = 228, CS = 48;
let grid = text(40, 262, '(Σ_{*t*} *s*^{(*t*)})² = Σ_{*u*,*t*} *s*^{(*u*)}*s*^{(*t*)}', {size: 20});
let zeros = '';
for (let u = 0; u < 5; u++) {
  grid += text(G0x - 12, G0y + u * CS + 30, String(u + 1), {size: 14, fill: C.mute, anchor: 'end'});
  grid += text(G0x + u * CS + CS / 2, G0y - 8, String(u + 1), {size: 14, fill: C.mute, anchor: 'middle'});
  for (let t = 0; t < 5; t++) {
    const d = u === t;
    grid += rect(G0x + t * CS + 2, G0y + u * CS + 2, CS - 4, CS - 4, {rx: 6, fill: d ? C.bl : C.track, op: d ? 0.85 : 1});
    if (!d) zeros += rect(G0x + t * CS + 2, G0y + u * CS + 2, CS - 4, CS - 4, {rx: 6, fill: C.panel, op: 0.7}) +
      text(G0x + t * CS + CS / 2, G0y + u * CS + 31, '0', {size: 18, anchor: 'middle', fill: C.mute});
  }
}
zeros += text(40, 330, 'for *u* < *t*, *s*^{(*u*)} is fixed', {size: 18, fill: C.soft}) +
  text(40, 356, 'by the prefix of *t*, so each pair', {size: 18, fill: C.soft}) +
  text(40, 382, 'averages to zero', {size: 18, fill: C.soft});

let sum = '';
const SX = 830, SB = 468;
let y = SB;
FS.forEach((v, i) => {
  const h = 46 * v;
  sum += rect(SX, y - h, 70, h - 3, {rx: 4, fill: C.bl, op: 0.85});
  sum += text(SX + 80, y - h / 2 + 6, '*f*(*s*_{' + (i + 1) + '})', {size: 14, fill: C.soft});
  y -= h;
});
sum += line(SX - 10, SB, SX + 80, SB, {stroke: C.rule, w: 2});
sum += text(SX + 35, y - 18, '*F* = E[Σ_{*t*} *f*(*s*_{*t*})]', {size: 19, anchor: 'middle', weight: 600, fill: C.bl});

export default {
  name: 'chain',
  aria: 'Token scores along a response; in the square of their sum, the off-diagonal pairs average to zero and the Fisher is the sum of per-position Fishers',
  frames: [{base, layers: [mean, grid, zeros, sum]}],
  steps: [
    {k: 0, text: 'The score of a response is a sum of one token score per position.'},
    {k: 1, text: 'Given its prefix, each token score averages to zero.'},
    {k: 2, text: 'Squaring the sum gives one term for every pair of positions.'},
    {k: 3, text: 'For *u* < *t* the earlier score is fixed by the later prefix, so the pair averages to zero.'},
    {k: 4, text: 'Only the diagonal survives: the Fisher adds one token Fisher per position.'},
  ],
};
