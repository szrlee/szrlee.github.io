#!/usr/bin/env node
// Builds the hand-authored figures. Usage: node tools/notefigs/build.mjs
//   figs/<name>.mjs   -> assets/figs/<name>.html        (step-through, read by assets/js/stepfig.js)
//   embeds/<name>.mjs -> static/figs/embed/<name>.html  (slider pages, shown by the livefig shortcode)
// A figure module exports {name, aria, frames: [{base, layers: [svg...]}], steps: [{f, k, text}], note?}.
// An embed module exports {name, title, width, height, params: [...], draw(p), readout(p)}.
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {C, esc, helpers} from './lib.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const SITE = path.resolve(HERE, '../..');
const FORBIDDEN = JSON.parse(fs.readFileSync(path.join(SITE, 'tools/deckfigs/figures.json'), 'utf8')).forbidden;

const attr = (t) => esc(t).replace(/"/g, '&quot;');

// Caption markup, as in figure labels: *x* italic, _{..} subscript, ^{..} superscript.
function caption(s) {
  return esc(s)
    .replace(/\*([^*]+)\*/g, '<i>$1</i>')
    .replace(/_\{([^}]*)\}/g, '<sub>$1</sub>')
    .replace(/\^\{([^}]*)\}/g, '<sup>$1</sup>');
}

function figureHtml(fig) {
  const W = fig.width || 960, H = fig.height || 500;
  const frames = fig.frames.map((fr, i) => {
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" aria-hidden="true">` +
      `<rect width="${W}" height="${H}" rx="16" fill="${C.panel}"/>` + fr.base +
      fr.layers.map((g, k) => `<g data-step="${k + 1}">${g}</g>`).join('') + '</svg>';
    return `<div class="stepfig-frame" data-frame="${i}" data-levels="${fr.layers.length}">${svg}</div>`;
  }).join('\n');
  const steps = fig.steps.map((s) => ({f: s.f || 0, k: s.k, title: '', text: [`<p>${caption(s.text)}</p>`], note: s.note ?? fig.note ?? ''}));
  const last = steps[steps.length - 1];
  return `<figure class="stepfig${fig.wide ? ' stepfig--wide' : ''}" data-fig="${fig.name}" role="group" aria-label="${attr(fig.aria)}">
<div class="stepfig-viewport"><div class="stepfig-sizer"><div class="stepfig-stage" style="width:${W}px;height:${H}px">
${frames}
</div></div></div>
<div class="stepfig-bar">
<button type="button" class="stepfig-prev" aria-label="Previous step">&#8592;</button>
<button type="button" class="stepfig-play" aria-label="Play">&#9654;</button>
<button type="button" class="stepfig-next" aria-label="Next step">&#8594;</button>
<span class="stepfig-count">${steps.length} / ${steps.length}</span>
</div>
<div class="stepfig-caption" aria-live="polite">${last.text.join('')}</div>
<div class="stepfig-note">${esc(last.note)}</div>
<script type="application/json" class="stepfig-steps">${JSON.stringify(steps).replace(/</g, '\\u003c')}</script>
</figure>
`;
}

function embedHtml(em) {
  const p0 = Object.fromEntries(em.params.map((q) => [q.id, q.value]));
  const controls = em.params.map((q) => `<label><span>${q.label}</span>` +
    `<input type="range" id="${q.id}" min="${q.min}" max="${q.max}" step="${q.step}" value="${q.slider ?? q.value}">` +
    `<output id="${q.id}-out">${q.show(q.value)}</output></label>`).join('');
  const code = [
    `var C=${JSON.stringify(C)};`,
    ...helpers.map((h) => h.toString()),
    `var PARAMS=${JSON.stringify(em.params.map((q) => ({id: q.id})))};`,
    `var toValue={${em.params.map((q) => `${JSON.stringify(q.id)}:${(q.toValue || ((x) => x)).toString()}`).join(',')}};`,
    `var toSlider={${em.params.map((q) => `${JSON.stringify(q.id)}:${(q.toSlider || ((x) => x)).toString()}`).join(',')}};`,
    `var show={${em.params.map((q) => `${JSON.stringify(q.id)}:${q.show.toString()}`).join(',')}};`,
    `var clamp=${(em.clamp || ((p) => p)).toString()};`,
    `var draw=${em.draw.toString()};`,
    `var readout=${em.readout.toString()};`,
    `function update(src){var p={};PARAMS.forEach(function(q){p[q.id]=toValue[q.id](+document.getElementById(q.id).value);});` +
      `p=clamp(p,src);PARAMS.forEach(function(q){if(q.id!==src)document.getElementById(q.id).value=toSlider[q.id](p[q.id]);` +
      `document.getElementById(q.id+'-out').textContent=show[q.id](p[q.id]);});` +
      `document.getElementById('fig').innerHTML=draw(p);document.getElementById('read').innerHTML=readout(p);post();}`,
    `function post(){parent.postMessage({livefig:${JSON.stringify(em.name)},h:document.getElementById('wrap').offsetHeight},'*');}`,
    `PARAMS.forEach(function(q){document.getElementById(q.id).addEventListener('input',function(){update(q.id);});});`,
    `window.addEventListener('resize',post);window.addEventListener('load',post);post();`,
  ].join('\n');
  return `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>${esc(em.title)}</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Source+Serif+4:ital,wght@1,400;1,600&display=swap">
<style>
html,body{margin:0;padding:0;background:transparent;color:${C.ink};font-family:"IBM Plex Sans",Arial,sans-serif}
#fig{overflow-x:auto}
#fig svg{display:block;width:100%;min-width:560px;height:auto}
.ctl{display:flex;flex-wrap:wrap;gap:10px 28px;margin:12px 4px 0}
.ctl label{display:flex;align-items:center;gap:10px;font-size:15px;color:${C.soft}}
.ctl input{width:200px;accent-color:${C.or}}
.ctl output{min-width:3.6em;font-variant-numeric:tabular-nums;color:${C.ink};font-weight:500}
#wrap{display:block;padding-bottom:2px}
#read{margin:10px 4px 2px;font-size:15px;line-height:1.5;color:${C.soft}}
#read b{color:${C.ink};font-weight:600}
i{font-family:"Source Serif 4",Georgia,serif}
</style></head><body><main id="wrap">
<div id="fig">${em.draw(p0)}</div>
<div class="ctl">${controls}</div>
<div id="read" aria-live="polite">${em.readout(p0)}</div>
</main>
<script>
${code}
</script>
</body></html>
`;
}

async function load(dir) {
  const full = path.join(HERE, dir);
  if (!fs.existsSync(full)) return [];
  const files = fs.readdirSync(full).filter((n) => n.endsWith('.mjs')).sort();
  return Promise.all(files.map(async (n) => (await import(pathToFileURL(path.join(full, n)).href)).default));
}

const writes = [];
for (const fig of await load('figs')) writes.push([path.join(SITE, 'assets/figs', `${fig.name}.html`), figureHtml(fig)]);
for (const em of await load('embeds')) writes.push([path.join(SITE, 'static/figs/embed', `${em.name}.html`), embedHtml(em)]);
for (const [file, text] of writes) {
  for (const bad of FORBIDDEN) if (text.includes(bad)) throw new Error(`forbidden string "${bad}" in ${file}`);
}
for (const [file, text] of writes) {
  fs.mkdirSync(path.dirname(file), {recursive: true});
  fs.writeFileSync(file, text);
}
console.log(writes.map(([file]) => path.relative(SITE, file)).join('\n'));
