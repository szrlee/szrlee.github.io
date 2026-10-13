#!/usr/bin/env node
// Extracts step-through figures from the built deck into the site.
//
//   node tools/deckfigs/extract.mjs [deck-source-dir]
//
// Reads <deck>/project/slides/<id>.html, keeps the figure panel of each slide
// (its layers, labels and data-build-in steps), turns the right-column text of
// each build level into a step caption, and writes:
//   assets/figs/<figure>.html        figure markup, inlined by the stepfig shortcode
//   static/figs/embed/<slide>.html   live animation of a slide (x-embed), run in an iframe
//   static/figs/img/<file>           images listed in figures.json
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const SITE = path.resolve(HERE, '..', '..');
const DECK = path.resolve(process.argv[2] || path.join(os.homedir(), 'Documents', 'deck-source'));
const CFG = JSON.parse(fs.readFileSync(path.join(HERE, 'figures.json'), 'utf8'));
const P = CFG.panel;

const VOID = new Set(['br', 'img', 'hr', 'input', 'meta', 'link', 'source', 'wbr']);

// Top-level elements of an HTML fragment. MathJax output is well-formed, so a
// depth counter over start and end tags is enough; script bodies are skipped raw.
function topLevel(html) {
  const out = [];
  let i = 0, depth = 0, start = -1, tag = '';
  while (i < html.length) {
    const lt = html.indexOf('<', i);
    if (lt < 0) break;
    if (html.startsWith('<!--', lt)) { i = html.indexOf('-->', lt) + 3; continue; }
    const gt = html.indexOf('>', lt);
    const raw = html.slice(lt, gt + 1);
    if (raw[1] === '/') {
      depth--;
      if (depth === 0) out.push({tag, html: html.slice(start, gt + 1)});
      i = gt + 1;
      continue;
    }
    const name = raw.match(/^<([a-zA-Z][\w-]*)/)[1].toLowerCase();
    const selfClosing = raw.endsWith('/>') || VOID.has(name);
    if (depth === 0) { start = lt; tag = name; }
    if (name === 'script') {
      const end = html.indexOf('</script>', gt) + '</script>'.length;
      if (depth === 0) out.push({tag, html: html.slice(start, end)});
      i = end;
      continue;
    }
    if (selfClosing) {
      if (depth === 0) out.push({tag, html: raw});
    } else {
      depth++;
    }
    i = gt + 1;
  }
  return out;
}

const openTag = (el) => el.html.slice(0, el.html.indexOf('>') + 1);
const styleOf = (el) => (openTag(el).match(/style="([^"]*)"/) || [, ''])[1];
const num = (style, key) => {
  const m = style.match(new RegExp(`(?:^|;)${key}:(-?[\\d.]+)px`));
  return m ? parseFloat(m[1]) : null;
};
const level = (el) => {
  const m = openTag(el).match(/data-build-in="[^"\d]*(\d+)/);
  return m ? parseInt(m[1], 10) : 0;
};
function stripSvg(h) {
  const inner = /<svg\b[^>]*>(?:(?!<svg\b)[\s\S])*?<\/svg>/g;
  while (inner.test(h)) h = h.replace(inner, ' ');
  return h;
}
const textOf = (h) => stripSvg(h.replace(/<script[\s\S]*?<\/script>/g, ' '))
  .replace(/<[^>]+>/g, ' ')
  .replace(/&nbsp;/g, ' ').replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&quot;/g, '"').replace(/&#39;/g, "'").replace(/&amp;/g, '&')
  .replace(/\s+/g, ' ').trim();
const esc = (t) => t.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

function shiftOpenTag(el, crop, extra = '') {
  const tag = openTag(el);
  let t = tag
    .replace(/left:(-?[\d.]+)px/, (_, v) => `left:${+(+v - crop.left).toFixed(2)}px`)
    .replace(/top:(-?[\d.]+)px/, (_, v) => `top:${+(+v - crop.top).toFixed(2)}px`)
    .replace(/\sdata-build-in="[^"]*"/, '');
  if (extra) t = t.replace(/^<([\w-]+)/, `<$1${extra}`);
  return t + el.html.slice(tag.length);
}

const blobMap = JSON.parse(fs.readFileSync(path.join(DECK, 'assets', 'map.json'), 'utf8'));
const usedImages = new Set();
function mapBlobs(h) {
  return h.replace(/\/_blob\/([0-9a-f]+)/g, (_, id) => {
    const file = blobMap[id];
    if (!file) throw new Error(`unknown blob ${id}`);
    usedImages.add(file);
    return `/figs/img/${file}`;
  });
}

// Right-column content of one build level, as caption HTML: text stays text,
// formulas keep the deck's MathJax SVG at 0.6 scale.
const MATH_SCALE = 0.6;
function captionItem(el) {
  const svgs = [];
  const re = /<svg\b[^>]*aria-label="[^"]*"[^>]*>/g;
  let m;
  while ((m = re.exec(el.html))) svgs.push(m.index);
  const text = textOf(el.html);
  if (!svgs.length) return text ? `<p>${esc(text)}</p>` : '';
  const start = svgs[0];
  let depth = 0, i = start, end = -1;
  const tagRe = /<(\/?)svg\b[^>]*>/g;
  tagRe.lastIndex = start;
  while ((m = tagRe.exec(el.html))) {
    depth += m[1] ? -1 : 1;
    if (depth === 0) { end = m.index + m[0].length; break; }
  }
  let svg = el.html.slice(start, end);
  svg = svg.replace(/^<svg\b[^>]*>/, (tag) => tag
    .replace(/\swidth="([\d.]+)"/, (_, w) => ` width="${+(w * MATH_SCALE).toFixed(1)}"`)
    .replace(/\sheight="([\d.]+)"/, (_, h) => ` height="${+(h * MATH_SCALE).toFixed(1)}"`));
  if (text) return `<p class="stepfig-inline">${svg} ${esc(text)}</p>`;
  return `<div class="stepfig-math">${svg}</div>`;
}

const embeds = new Map();
// Splits a slide into layers (inside the crop), captions (right of the crop, or
// below it when `belowCaptions`), and a note (level-0 <p> lines below the crop).
function readSlide(id, crop, belowCaptions) {
  const src = fs.readFileSync(path.join(DECK, 'project', 'slides', `${id}.html`), 'utf8');
  const open = src.indexOf('>', src.indexOf('<section')) + 1;
  const body = src.slice(open, src.lastIndexOf('</section>'));
  const right = crop.left + crop.width, bottom = crop.top + crop.height;
  const panel = [], side = [], note = [];
  for (const el of topLevel(body)) {
    if (el.tag === 'aside' || el.tag === 'script') continue;
    const st = styleOf(el);
    if (!st.includes('position:absolute')) continue;
    if (num(st, 'bottom') !== null) continue;
    const left = num(st, 'left'), top = num(st, 'top');
    if (top === null || top < 200) continue;
    if (left >= right) { side.push(el); continue; }
    if (top >= bottom) {
      if (belowCaptions && (el.tag !== 'p' || level(el) > 0)) side.push(el);
      else if (el.tag === 'p') note.push(textOf(el.html));
      continue;
    }
    panel.push(el);
  }

  const levels = Math.max(0, ...panel.map(level), ...side.map(level));
  const layers = panel.map((el) => {
    const k = level(el);
    const extra = k ? ` data-step="${k}"` : '';
    if (el.tag === 'x-embed') {
      const inner = el.html.slice(openTag(el).length, el.html.lastIndexOf('</x-embed>'));
      embeds.set(id, inner);
      const st = styleOf(el);
      const box = ['left', 'top', 'width', 'height'].map((key) => num(st, key));
      return `<iframe class="stepfig-embed"${extra} data-src="/figs/embed/${id}.html" title="Animation" sandbox="allow-scripts" tabindex="-1" style="position:absolute;left:${box[0] - crop.left}px;top:${box[1] - crop.top}px;width:${box[2]}px;height:${box[3]}px;border:0;background:transparent"></iframe>`;
    }
    return mapBlobs(shiftOpenTag(el, crop, extra));
  });

  const isTitle = (el) => el.tag === 'p' && /font-size:(4\d|5\d)px/.test(styleOf(el));
  const title = side.filter((el) => level(el) === 0 && isTitle(el)).map((el) => textOf(el.html))[0] || '';
  const captions = [];
  for (let k = 0; k <= levels; k++) {
    captions.push(side
      .filter((el) => level(el) === k && !isTitle(el))
      .map(captionItem)
      .filter(Boolean));
  }
  const label = (panel.map((el) => (openTag(el) + el.html).match(/aria-label="([^"]+)"/)).find(Boolean) || [, ''])[1];
  return {id, layers, levels, title, captions, note: note.filter(Boolean).join(' '), label};
}

// A figure is either a list of slide ids (standard panel) or
// {frames, crop, wide, min, belowCaptions, captions}. `captions` replaces the
// extracted step captions, one string per step.
function buildFigure(name, spec) {
  const fig = Array.isArray(spec) ? {frames: spec} : spec;
  const crop = {...P, ...(fig.crop || {})};
  const frames = fig.frames.map((id) => readSlide(id, crop, fig.belowCaptions));
  const steps = [];
  frames.forEach((fr, f) => {
    for (let k = 0; k <= fr.levels; k++) {
      steps.push({f, k, title: fr.title, text: fr.captions[k], note: fr.note});
    }
  });
  if (fig.captions) {
    if (fig.captions.length !== steps.length) {
      throw new Error(`${name}: ${fig.captions.length} captions for ${steps.length} steps`);
    }
    steps.forEach((s, i) => { s.title = ''; s.text = [`<p>${esc(fig.captions[i])}</p>`]; });
  }
  const last = steps[steps.length - 1];
  const caption = (s) => `${s.title ? `<p class="stepfig-title">${esc(s.title)}</p>` : ''}${s.text.join('')}`;
  const framesHtml = frames.map((fr, f) =>
    `<div class="stepfig-frame" data-frame="${f}" data-levels="${fr.levels}">\n${fr.layers.join('\n')}\n</div>`).join('\n');
  const json = JSON.stringify(steps).replace(/</g, '\\u003c');
  const cls = fig.wide ? 'stepfig stepfig--wide' : 'stepfig';
  const min = fig.min ? ` data-min="${fig.min}"` : '';
  return `<figure class="${cls}" data-fig="${name}"${min} role="group" aria-label="${esc(frames[0].label || name)}">
<div class="stepfig-viewport"><div class="stepfig-sizer"><div class="stepfig-stage" style="width:${crop.width}px;height:${crop.height}px">
${framesHtml}
</div></div></div>
<div class="stepfig-bar">
<button type="button" class="stepfig-prev" aria-label="Previous step">&#8592;</button>
<button type="button" class="stepfig-play" aria-label="Play">&#9654;</button>
<button type="button" class="stepfig-next" aria-label="Next step">&#8594;</button>
<span class="stepfig-count">${steps.length} / ${steps.length}</span>
</div>
<div class="stepfig-caption" aria-live="polite">${caption(last)}</div>
<div class="stepfig-note">${esc(last.note)}</div>
<script type="application/json" class="stepfig-steps">${json}</script>
</figure>
`;
}

function guard(file, text) {
  for (const bad of CFG.forbidden) {
    if (text.includes(bad)) throw new Error(`forbidden string "${bad}" in ${file}`);
  }
}

const writes = [];
for (const [name, spec] of Object.entries(CFG.figures)) {
  writes.push([path.join(SITE, 'assets', 'figs', `${name}.html`), buildFigure(name, spec)]);
}
for (const [id, inner] of embeds) {
  const html = `<!doctype html><html><head><meta charset="utf-8"><style>html,body{margin:0;padding:0;overflow:hidden;background:transparent}</style></head><body>${inner}</body></html>\n`;
  writes.push([path.join(SITE, 'static', 'figs', 'embed', `${id}.html`), html]);
}
for (const [file, text] of writes) guard(file, text);
for (const [file, text] of writes) {
  fs.mkdirSync(path.dirname(file), {recursive: true});
  fs.writeFileSync(file, text);
}
for (const file of new Set([...usedImages, ...Object.keys(CFG.images)])) {
  const dst = path.join(SITE, 'static', 'figs', 'img', CFG.images[file] || file);
  fs.mkdirSync(path.dirname(dst), {recursive: true});
  fs.copyFileSync(path.join(DECK, 'assets', file), dst);
}
console.log(`wrote ${writes.length} files, ${embeds.size} embeds, ${usedImages.size + Object.keys(CFG.images).length} images`);
