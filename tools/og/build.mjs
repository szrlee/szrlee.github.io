// Renders the social preview cards (1200x630): assets/media/og-default.png for the site,
// and og.png in every published post bundle, from tools/og/card.html in headless Chrome.
// Usage: node tools/og/build.mjs [slug ...]
import {spawn} from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const CHROME = process.env.CHROME || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const W = 1200, H = 630;
const URL_TEXT = 'richardli.xyz';
const MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];

function unquote(v) {
  v = v.trim();
  if (v.startsWith("'") && v.endsWith("'")) return v.slice(1, -1).replace(/''/g, "'");
  if (v.startsWith('"') && v.endsWith('"')) return v.slice(1, -1).replace(/\\"/g, '"');
  return v;
}

// Enough YAML for this site's front matter: one-line scalars and lists of scalars.
function frontMatter(file) {
  const m = fs.readFileSync(file, 'utf8').match(/^---\n([\s\S]*?)\n---/);
  const out = {};
  let list = null;
  for (const line of (m ? m[1] : '').split('\n')) {
    const item = line.match(/^\s+-\s+(.*)$/);
    if (item && list) { out[list].push(unquote(item[1])); continue; }
    const kv = line.match(/^([A-Za-z_]+):\s*(.*)$/);
    list = null;
    if (!kv) continue;
    if (kv[2] === '') { out[kv[1]] = []; list = kv[1]; } else out[kv[1]] = unquote(kv[2]);
  }
  return out;
}

function escapeRe(s) { return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); }

function monthYear(date) {
  const m = String(date || '').match(/^(\d{4})-(\d{2})/);
  return m ? `${MONTHS[+m[2] - 1]} ${m[1]}` : 'Writing';
}

function homeCard() {
  const y = fs.readFileSync(path.join(ROOT, 'data/home.yaml'), 'utf8');
  const get = (k) => unquote((y.match(new RegExp(`^  ${k}:\\s*(.*)$`, 'm')) || ['', ''])[1]);
  return {
    kind: 'default',
    kicker: get('kicker').replace(/<[^>]+>/g, ''),
    name: get('display_name'),
    pitch: get('pitch'),
    url: URL_TEXT,
  };
}

function postCards(only) {
  const dir = path.join(ROOT, 'content/post');
  const posts = fs.readdirSync(dir)
    .filter((slug) => fs.existsSync(path.join(dir, slug, 'index.md')))
    .map((slug) => ({slug, fm: frontMatter(path.join(dir, slug, 'index.md'))}))
    .filter((p) => String(p.fm.draft) !== 'true');
  const count = {};
  for (const p of posts) { const s = (p.fm.series || [])[0]; if (s) count[s] = (count[s] || 0) + 1; }
  return posts.filter((p) => !only.length || only.includes(p.slug)).map(({slug, fm}) => {
    const series = (fm.series || [])[0];
    let title = fm.title || slug;
    let sub = fm.subtitle || '';
    let kicker = monthYear(fm.date);
    if (series) {
      kicker = fm.series_order ? `${series} · Part ${fm.series_order} of ${count[series]}` : series;
      title = title.replace(/^Part\s+\d+\s*[:·—-]\s*/i, '');
      sub = sub.replace(new RegExp(`^${escapeRe(series)},?\\s*Part\\s+\\d+\\s*:\\s*`, 'i'), '');
    }
    if (sub) sub = sub[0].toUpperCase() + sub.slice(1);
    return {out: path.join(dir, slug, 'og.png'), data: {kind: 'post', kicker, title, sub, name: 'Yingru Li', url: URL_TEXT}};
  });
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const only = process.argv.slice(2);
const jobs = [...(only.length ? [] : [{out: path.join(ROOT, 'assets/media/og-default.png'), data: homeCard()}]), ...postCards(only)];

const port = 9400 + Math.floor(Math.random() * 400);
const chrome = spawn(CHROME, ['--headless=new', `--remote-debugging-port=${port}`, '--disable-gpu', '--hide-scrollbars',
  '--allow-file-access-from-files', `--user-data-dir=/tmp/og-chrome-${port}`, `--window-size=${W},${H}`, 'about:blank'], {stdio: 'ignore'});
let ws, id = 0;
const pending = new Map();
const send = (method, params = {}) => new Promise((res, rej) => {
  const i = ++id; pending.set(i, {res, rej}); ws.send(JSON.stringify({id: i, method, params}));
});
try {
  let list;
  for (let t = 0; t < 50 && !list; t++) { try { list = await (await fetch(`http://127.0.0.1:${port}/json`)).json(); } catch { await sleep(200); } }
  ws = new WebSocket(list.find((p) => p.type === 'page').webSocketDebuggerUrl);
  await new Promise((r) => ws.addEventListener('open', r));
  ws.addEventListener('message', (e) => {
    const m = JSON.parse(e.data);
    if (m.id && pending.has(m.id)) { const p = pending.get(m.id); pending.delete(m.id); m.error ? p.rej(new Error(m.error.message)) : p.res(m.result); }
  });
  await send('Page.enable');
  await send('Emulation.setDeviceMetricsOverride', {width: W, height: H, deviceScaleFactor: 1, mobile: false});
  const page = pathToFileURL(path.join(ROOT, 'tools/og/card.html')).href;
  for (const job of jobs) {
    await send('Page.navigate', {url: `${page}?d=${encodeURIComponent(JSON.stringify(job.data))}`});
    let ready = null;
    for (let t = 0; t < 100 && !ready; t++) {
      await sleep(150);
      ready = (await send('Runtime.evaluate', {expression: 'window.__ready || null', returnByValue: true})).result.value;
    }
    if (!ready) throw new Error(`card did not render: ${job.out}`);
    const {data} = await send('Page.captureScreenshot', {format: 'png', clip: {x: 0, y: 0, width: W, height: H, scale: 1}});
    fs.writeFileSync(job.out, Buffer.from(data, 'base64'));
    console.log(`${path.relative(ROOT, job.out)}  title ${ready.size}px${ready.fits ? '' : '  (does not fit)'}`);
  }
} finally {
  try { ws && ws.close(); } catch {}
  chrome.kill('SIGKILL');
}
