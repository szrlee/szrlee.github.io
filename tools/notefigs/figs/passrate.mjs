import {C, f, text, line, rect, dot, path} from '../lib.mjs';

// What one pass/fail episode can teach: at most h(p) bits for the pass rate p the
// learner predicts, and 16 h(p) for a group of 16 samples of the prompt.
const h = (p) => (p <= 0 || p >= 1 ? 0 : -p * Math.log2(p) - (1 - p) * Math.log2(1 - p));
const X0 = 80, XW = 460, Y0 = 368, YH = 250;
const X = (p) => X0 + XW * p, Y = (b) => Y0 - YH * b;

let frame = rect(24, 24, 560, 412, {rx: 14, fill: C.card}) + rect(600, 24, 336, 412, {rx: 14, fill: C.card});
frame += text(48, 62, 'what one pass/fail episode can teach', {size: 20, weight: 600});
frame += line(X0, Y0, X0 + XW, Y0, {stroke: C.rule, w: 2}) + line(X0, Y0, X0, Y(1) - 14, {stroke: C.rule, w: 2});
frame += line(X0, Y(1), X0 + XW, Y(1), {stroke: C.rule, w: 1.5, dash: '5 6'});
frame += text(X0 - 10, Y(1) + 6, '1 bit', {size: 16, anchor: 'end', fill: C.mute});
frame += text(X0 - 10, Y0 + 6, '0', {size: 16, anchor: 'end', fill: C.mute});
[0, 0.5, 1].forEach((p) => { frame += text(X(p), Y0 + 26, String(p), {size: 16, anchor: 'middle', fill: C.mute}); });
frame += text(X(0.5), Y0 + 54, 'pass rate *p* predicted from the history', {size: 17, anchor: 'middle', fill: C.mute});
frame += text(624, 62, '16 samples of the prompt', {size: 19, weight: 600});
frame += rect(624, 286, 16, 16, {rx: 4, fill: C.bl}) + text(648, 300, 'pass', {size: 15, fill: C.soft});
frame += rect(712, 287, 14, 14, {rx: 4, fill: C.card, stroke: C.gray, sw: 2}) + text(736, 300, 'fail', {size: 15, fill: C.soft});

let d = '';
for (let i = 0; i <= 200; i++) d += (i ? 'L' : 'M') + f(X(i / 200)) + ' ' + f(Y(h(i / 200)));
const curve = path(d, {stroke: C.bl, w: 3.5}) + text(X(0.2) - 8, Y(h(0.2)) - 12, '*h*(*p*)', {size: 19, anchor: 'end', fill: C.bl, weight: 600});

function marker(p, lab, side, faded) {
  const x = X(p), y = Y(h(p)), op = faded ? 0.35 : 1;
  const tx = side === 'up' ? x : side === 'right' ? x + 14 : x - 14, ty = side === 'up' ? y - 16 : y + 6;
  return line(x, y, x, Y0, {stroke: C.or, w: 2, dash: '4 5', op}) + dot(x, y, 8, {fill: C.or, op}) +
    text(tx, ty, lab, {size: 17, weight: 600, fill: C.or, op, anchor: side === 'up' ? 'middle' : side === 'right' ? 'start' : 'end'});
}

const GX = 674.5, GY = 84, CS = 40, GAP = 9;
function group(pattern, bits, extra) {
  let s = '';
  pattern.forEach((pass, i) => {
    const x = GX + (i % 4) * (CS + GAP), y = GY + Math.floor(i / 4) * (CS + GAP);
    s += pass ? rect(x, y, CS, CS, {rx: 7, fill: C.bl}) : rect(x + 1, y + 1, CS - 2, CS - 2, {rx: 7, fill: C.card, stroke: C.gray, sw: 2});
  });
  s += text(624, 334, 'at most 16 · *h*(*p*) =', {size: 18, fill: C.soft});
  s += text(624, 370, bits, {size: 28, weight: 600, fill: C.or});
  (extra || []).forEach((t, i) => { s += text(624, 400 + 22 * i, t, {size: 15, fill: C.soft}); });
  return s;
}
const P50 = [1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1];
const P90 = [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1];
const P99 = Array(16).fill(1);
const m50 = marker(0.5, '1 bit', 'up'), m90 = marker(0.9, '0.47 bit', 'right');

export default {
  name: 'passrate',
  aria: 'Binary entropy of the predicted pass rate, with a group of sixteen pass/fail samples: a prompt solved half the time can teach one bit per episode, one solved 99% of the time almost nothing',
  height: 460,
  frames: [
    {base: frame, layers: [curve, m50, group(P50, '16 bits')]},
    {base: frame + curve + marker(0.5, '1 bit', 'up', true), layers: [m90 + group(P90, '7.5 bits')]},
    {base: frame + curve + marker(0.5, '1 bit', 'up', true) + marker(0.9, '0.47 bit', 'right', true),
      layers: [marker(0.99, '0.08 bit', 'left') + group(P99, '1.3 bits', ['85% of such groups are all passes:', 'with a group-mean baseline, no update'])]},
  ],
  steps: [
    {f: 0, k: 0, text: 'One prompt, a pass/fail reward, and the pass rate *p* the learner would predict from everything it has seen.'},
    {f: 0, k: 1, text: 'An episode can teach at most *h*(*p*) bits, the entropy of a coin that comes up heads with probability *p*.'},
    {f: 0, k: 2, text: 'A prompt solved half the time: at most 1 bit per episode.'},
    {f: 0, k: 3, text: 'Sixteen samples of it: at most 16 bits.'},
    {f: 1, k: 1, text: 'Solved 90% of the time: at most 0.47 bits per episode, 7.5 per group.'},
    {f: 2, k: 1, text: 'Solved 99% of the time: 0.08 bits per episode, 1.3 per group. Most such groups are all passes, and a group-mean baseline gives them no update at all.'},
  ],
};
