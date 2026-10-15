import {C, f, text, line, path, dot} from '../lib.mjs';

// Two responses with one logit each. Policy gradient raises the rewarded one at a
// rate proportional to its probability; a demonstration's push does not shrink.
const sig = (l) => 1 / (1 + Math.exp(-l));
const logit = (p) => Math.log(p / (1 - p));
const tPG = (eps, gap = 1) => { const l = logit(eps); return (Math.exp(-l) - 2 * l - Math.exp(l)) / (2 * gap); };
const tSFT = (eps) => { const l = logit(eps); return (1 - l - Math.exp(l)) / 2; };
function curve(eps, rate, tmax) {
  const out = [];
  let l = logit(eps), t = 0;
  out.push([0, sig(l)]);
  while (t < tmax && sig(l) < 0.999) {
    const dt = Math.min(0.05, 0.02 / Math.max(1e-6, rate(sig(l))));
    l += rate(sig(l)) * dt; t += dt;
    out.push([t, sig(l)]);
  }
  return out;
}
const pg = (p) => 2 * p * (1 - p), sft = (p) => 2 * (1 - p);
const X0 = 110, XW = 790, TM = 70, YT = 70, YB = 420, LMIN = -3.2;
const X = (t) => X0 + XW * Math.min(t, TM) / TM, Y = (p) => YT + (YB - YT) * (Math.log10(p) / LMIN);
const draw = (pts, o) => path(pts.filter((p) => p[0] <= TM).map((p, i) => (i ? 'L' : 'M') + f(X(p[0])) + ' ' + f(Y(p[1]))).join(''), o);

let base = line(X0, YB, X0 + XW, YB, {stroke: C.rule, w: 2}) + line(X0, YT - 10, X0, YB, {stroke: C.rule, w: 2});
[[1, '1'], [0.1, '0.1'], [0.01, '0.01'], [0.001, '0.001']].forEach(([p, s]) => {
  base += line(X0 - 6, Y(p), X0, Y(p), {stroke: C.rule, w: 2}) + text(X0 - 12, Y(p) + 6, s, {size: 16, anchor: 'end', fill: C.mute});
});
base += line(X0, Y(0.5), X0 + XW, Y(0.5), {stroke: C.soft, w: 1.5, dash: '6 6'}) + text(X0 + XW, Y(0.5) - 10, 'one half', {size: 16, anchor: 'end', fill: C.soft});
base += text(X0 + XW, YB + 34, 'time: step size \u00d7 number of steps', {size: 16, anchor: 'end', fill: C.mute});
base += text(40, 44, 'probability of the rewarded response', {size: 18, fill: C.mute});
const t2 = tPG(0.01);
base += draw(curve(0.01, pg, TM), {stroke: C.bl, w: 3.5});
base += dot(X(t2), Y(0.5), 7, {fill: C.bl}) + text(X(t2) + 12, Y(0.5) + 26, 'from 1%: ' + Math.round(t2), {size: 18, fill: C.bl, weight: 600});

const t1 = tPG(0.1), t3 = tPG(0.001);
let more = draw(curve(0.1, pg, TM), {stroke: C.bl, w: 2.5, sop: 0.6});
more += dot(X(t1), Y(0.5), 6, {fill: C.bl, op: 0.6}) + text(X(t1) + 10, Y(0.5) - 12, 'from 10%: ' + Math.round(t1), {size: 16, fill: C.bl});
more += draw(curve(0.001, pg, TM), {stroke: C.bl, w: 2.5, sop: 0.6});
more += text(X0 + XW, Y(curve(0.001, pg, TM).find((p) => p[0] >= TM)[1]) - 12, 'from 0.1%: ' + Math.round(t3) + ' \u2192', {size: 16, anchor: 'end', fill: C.bl});

let demo = draw(curve(0.01, sft, TM), {stroke: C.or, w: 3.5}) + draw(curve(0.001, sft, TM), {stroke: C.or, w: 2.5, sop: 0.6});
demo += text(X(13), Y(0.004), 'demonstration: ' + tSFT(0.01).toFixed(1) + ' from 1%, ' + tSFT(0.001).toFixed(1) + ' from 0.1%', {size: 18, fill: C.or, weight: 600});

export default {
  name: 'amplify',
  aria: 'Probability of a rewarded response over time: policy gradient takes time proportional to one over the starting probability, a demonstration only logarithmic time',
  note: 'Two responses, one logit each, reward gap 1.',
  frames: [{base, layers: [more, demo]}],
  steps: [
    {k: 0, text: 'Policy gradient lifts a rare rewarded response in proportion to its current probability: from 1% it takes about ' + Math.round(t2) + ' time units to reach one half.'},
    {k: 1, text: 'Start ten times rarer and it takes about ten times longer: ' + Math.round(t3) + ' from 0.1%.'},
    {k: 2, text: 'A demonstration\u2019s push does not shrink with the probability: a few units from either start.'},
  ],
};
