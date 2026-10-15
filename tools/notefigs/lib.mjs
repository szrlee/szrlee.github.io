// SVG helpers shared by the hand-authored figures. Every helper is self-contained
// apart from C, f, esc and each other, so build.mjs can also inline them into the
// slider pages with Function.prototype.toString.

export const C = {
  panel: '#F0EDE4', card: '#FAF8F3', ink: '#161A22', soft: '#3F4654', mute: '#666C78',
  rule: '#C9C2B2', track: '#E4DFD2', gray: '#9AA0AC',
  or: '#C2410C', orSoft: '#F6D9C2', orMid: '#E08A52',
  bl: '#1D4ED8', blSoft: '#DCE5FB', blMid: '#7C9BE8',
};

export function f(v) { return Math.round(v * 10) / 10; }

export function esc(t) { return String(t).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }

// Light markup for labels: *x* italic serif, _{..} subscript, ^{..} superscript.
export function rich(s, size) {
  var segs = [], it = false, pos = 'base', buf = '', i, ch;
  function push() { if (buf) segs.push({t: buf, it: it, pos: pos}); buf = ''; }
  for (i = 0; i < s.length; i++) {
    ch = s[i];
    if (ch === '*') { push(); it = !it; continue; }
    if ((ch === '_' || ch === '^') && s[i + 1] === '{') { push(); pos = ch === '_' ? 'sub' : 'sup'; i++; continue; }
    if (ch === '}' && pos !== 'base') { push(); pos = 'base'; continue; }
    buf += ch;
  }
  push();
  var off = {base: 0, sub: 0.3 * size, sup: -0.38 * size}, cur = 0, out = '';
  segs.forEach(function (g) {
    var a = [], dy = off[g.pos] - cur;
    cur = off[g.pos];
    if (dy) a.push('dy="' + f(dy) + '"');
    if (g.pos !== 'base') a.push('font-size="' + f(size * 0.72) + '"');
    if (g.it) a.push('font-family="Source Serif 4, Georgia, serif" font-style="italic"');
    // No-break spaces: the HTML minifier trims ordinary spaces at tspan boundaries.
    out += '<tspan' + (a.length ? ' ' + a.join(' ') : '') + '>' + esc(g.t).replace(/ /g, '\u00a0') + '</tspan>';
  });
  return out;
}

export function text(x, y, s, o) {
  o = o || {};
  var size = o.size || 22;
  return '<text x="' + f(x) + '" y="' + f(y) + '" font-size="' + size + '" font-weight="' + (o.weight || 500) +
    '" fill="' + (o.fill || C.ink) + '" text-anchor="' + (o.anchor || 'start') + '"' +
    (o.op != null ? ' fill-opacity="' + o.op + '"' : '') + '>' + rich(s, size) + '</text>';
}

export function line(x1, y1, x2, y2, o) {
  o = o || {};
  return '<path d="M' + f(x1) + ' ' + f(y1) + 'L' + f(x2) + ' ' + f(y2) + '" fill="none" stroke="' + (o.stroke || C.ink) +
    '" stroke-width="' + (o.w || 2) + '"' + (o.dash ? ' stroke-dasharray="' + o.dash + '"' : '') +
    (o.op != null ? ' stroke-opacity="' + o.op + '"' : '') + (o.cap ? ' stroke-linecap="' + o.cap + '"' : '') + '/>';
}

export function arrow(x1, y1, x2, y2, o) {
  o = o || {};
  var w = o.w || 3, h = o.head || 4 * w + 6, a = Math.atan2(y2 - y1, x2 - x1);
  var bx = x2 - h * Math.cos(a), by = y2 - h * Math.sin(a), px = -Math.sin(a) * h * 0.42, py = Math.cos(a) * h * 0.42;
  return line(x1, y1, bx, by, {stroke: o.stroke, w: w, dash: o.dash, op: o.op}) +
    '<path d="M' + f(x2) + ' ' + f(y2) + 'L' + f(bx + px) + ' ' + f(by + py) + 'L' + f(bx - px) + ' ' + f(by - py) +
    'Z" fill="' + (o.stroke || C.ink) + '"' + (o.op != null ? ' fill-opacity="' + o.op + '"' : '') + '/>';
}

export function rect(x, y, w, h, o) {
  o = o || {};
  return '<rect x="' + f(x) + '" y="' + f(y) + '" width="' + f(Math.max(0, w)) + '" height="' + f(Math.max(0, h)) + '"' +
    (o.rx ? ' rx="' + o.rx + '"' : '') + ' fill="' + (o.fill || 'none') + '"' +
    (o.op != null ? ' fill-opacity="' + o.op + '"' : '') +
    (o.stroke ? ' stroke="' + o.stroke + '" stroke-width="' + (o.sw || 2) + '"' : '') +
    (o.dash ? ' stroke-dasharray="' + o.dash + '"' : '') + '/>';
}

export function dot(x, y, r, o) {
  o = o || {};
  return '<circle cx="' + f(x) + '" cy="' + f(y) + '" r="' + f(r) + '" fill="' + (o.fill || C.ink) + '"' +
    (o.op != null ? ' fill-opacity="' + o.op + '"' : '') +
    (o.stroke ? ' stroke="' + o.stroke + '" stroke-width="' + (o.sw || 2) + '"' : '') + '/>';
}

export function path(d, o) {
  o = o || {};
  return '<path d="' + d + '" fill="' + (o.fill || 'none') + '"' + (o.op != null ? ' fill-opacity="' + o.op + '"' : '') +
    (o.stroke ? ' stroke="' + o.stroke + '" stroke-width="' + (o.w || 2) + '"' : '') +
    (o.sop != null ? ' stroke-opacity="' + o.sop + '"' : '') +
    (o.dash ? ' stroke-dasharray="' + o.dash + '"' : '') + ' stroke-linejoin="round" stroke-linecap="round"/>';
}

export function poly(pts) {
  return pts.map(function (p, i) { return (i ? 'L' : 'M') + f(p[0]) + ' ' + f(p[1]); }).join('');
}

export function star(x, y, r, o) {
  var pts = [], i, a, rr;
  for (i = 0; i < 10; i++) {
    a = -Math.PI / 2 + i * Math.PI / 5;
    rr = i % 2 ? r * 0.45 : r;
    pts.push([x + rr * Math.cos(a), y + rr * Math.sin(a)]);
  }
  return path(poly(pts) + 'Z', o);
}

export function rng(seed) {
  var s = seed;
  return function () { s = (s * 16807) % 2147483647; return (s - 1) / 2147483646; };
}

export function gauss(r) {
  var u = 0, v = 0;
  while (!u) u = r();
  while (!v) v = r();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export const helpers = [f, esc, rich, text, line, arrow, rect, dot, path, poly, star, rng, gauss];
