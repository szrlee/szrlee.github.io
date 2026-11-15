#!/usr/bin/env python3
"""Step-through figures for the post "When Speed Kills Stability" (rl-collapse-training-inference).

    python3 tools/postfigs/rl_collapse.py

Writes assets/figs/rlc-*.html, in the markup that assets/js/stepfig.js drives and the
{{< stepfig "name" >}} shortcode inlines. The figures are plain SVG drawn here, with no build step:
a figure is a list of frames, a frame is one SVG whose <g data-step="k"> groups appear from step k
on, and a step is (frame, level, caption).
"""
import html
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, 'assets', 'figs')

BG, PANEL, INK, SOFT, MUTE, RULE = '#F0EDE4', '#FAF8F3', '#161A22', '#3F4654', '#666C78', '#C9C3B5'
OR, BL, PU, BAR = '#C2410C', '#1D4ED8', '#7C3AED', '#B8AF9C'   # sampler / MIS, trainer, TIS, neutral bars
MATH = "font-family=\"Georgia, 'Times New Roman', serif\" font-style=\"italic\""


def t(x, y, s, size=18, fill=INK, weight=400, anchor='start', extra=''):
    # The site's HTML minifier drops a plain space next to a tag, so those become no-break spaces.
    s = re.sub(r' (?=<tspan)', '&#160;', s)
    s = re.sub(r'(?<=</tspan>) ', '&#160;', s)
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" font-weight="{weight}" text-anchor="{anchor}" {extra}>{s}</text>'


def m(sym, sub='', size=None):
    """A math symbol in italic serif with an optional subscript, for use inside <text>."""
    fs = f' font-size="{size}"' if size else ''
    out = f'<tspan {MATH}{fs}>{sym}</tspan>'
    if sub:
        # the zero-width space carries the shift back up (an empty tspan would be dropped)
        out += f'<tspan dy="5" font-size="68%">{sub}</tspan><tspan dy="-5">&#8203;</tspan>'
    return out


def arrow(points, color=INK, width=2.5, dash=''):
    """Polyline with an arrowhead at its last point."""
    d = 'M' + ' L'.join(f'{x} {y}' for x, y in points)
    (x1, y1), (x2, y2) = points[-2], points[-1]
    dx, dy = x2 - x1, y2 - y1
    n = (dx * dx + dy * dy) ** 0.5 or 1
    ux, uy = dx / n, dy / n
    a, b = 11, 5.5
    head = f'M{x2} {y2} L{x2 - a * ux - b * uy:.1f} {y2 - a * uy + b * ux:.1f} L{x2 - a * ux + b * uy:.1f} {y2 - a * uy - b * ux:.1f} Z'
    da = f' stroke-dasharray="{dash}"' if dash else ''
    return (f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round"{da}/>'
            f'<path d="{head}" fill="{color}"/>')


def box(x, y, w, h, stroke=INK, fill=PANEL, width=3, dash='', rx=12):
    da = f' stroke-dasharray="{dash}"' if dash else ''
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"{da}/>'


def g(step, *parts):
    return f'<g data-step="{step}">' + ''.join(parts) + '</g>'


def svg(w, h, *parts):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'style="position:absolute;left:0;top:0;display:block" font-family="\'IBM Plex Sans\', Arial, sans-serif">'
            f'<rect x="0" y="0" width="{w}" height="{h}" rx="16" fill="{BG}"/>' + ''.join(parts) + '</svg>')


def figure(name, label, w, h, frames, steps, note='', wide=False):
    """frames: list of SVG strings; steps: list of (frame, level, caption)."""
    data = [{'f': f, 'k': k, 'title': '', 'text': ['<p>' + html.escape(c, quote=False) + '</p>'], 'note': note} for f, k, c in steps]
    body = ''.join(f'<div class="stepfig-frame" data-frame="{i}">{fr}</div>\n' for i, fr in enumerate(frames))
    cls = 'stepfig stepfig--wide' if wide else 'stepfig'
    out = f'''<figure class="{cls}" data-fig="{name}" role="group" aria-label="{html.escape(label)}">
<div class="stepfig-viewport"><div class="stepfig-sizer"><div class="stepfig-stage" style="width:{w}px;height:{h}px">
{body}</div></div></div>
<div class="stepfig-bar">
<button type="button" class="stepfig-prev" aria-label="Previous step">&#8592;</button>
<button type="button" class="stepfig-play" aria-label="Play">&#9654;</button>
<button type="button" class="stepfig-next" aria-label="Next step">&#8594;</button>
<span class="stepfig-count">{len(steps)} / {len(steps)}</span>
</div>
<div class="stepfig-caption" aria-live="polite"><p>{html.escape(steps[-1][2], quote=False)}</p></div>
<div class="stepfig-note">{html.escape(note, quote=False)}</div>
<script type="application/json" class="stepfig-steps">{json.dumps(data, ensure_ascii=False).replace('<', '\\u003c')}</script>
</figure>
'''
    path = os.path.join(OUT, name + '.html')
    with open(path, 'w', encoding='utf8') as f:
        f.write(out)
    print('wrote', os.path.relpath(path, ROOT), f'({len(steps)} steps)')


# ---------------------------------------------------------------------------------------------
# 1. The whole post in one picture: two engines, the gap, the loop, and where the mask sits.
# ---------------------------------------------------------------------------------------------
def loop():
    W, H = 960, 510
    cells = ''.join(f'<rect x="{345 + i * 32}" y="117" width="26" height="26" rx="5" fill="{PANEL}" stroke="{INK}" stroke-width="2"/>' for i in range(6))
    base = [
        # shared weights
        f'<path d="M165 80 V52 H795 V80" fill="none" stroke="{MUTE}" stroke-width="2"/>',
        f'<rect x="388" y="36" width="184" height="32" rx="16" fill="{BG}"/>',
        t(480, 59, f'same weights {m("θ")}', 20, INK, 600, 'middle'),
        # sampler
        box(40, 80, 250, 100, OR),
        t(165, 122, 'Sampler', 26, OR, 600, 'middle'),
        t(165, 153, 'vLLM · fast kernels', 18, MUTE, 400, 'middle'),
        # rollout
        arrow([(290, 130), (337, 130)]),
        cells,
        t(438, 172, f'rollout: a response {m("y")}', 17, MUTE, 400, 'middle'),
        arrow([(539, 130), (662, 130)]),
        # trainer
        box(670, 80, 250, 100, BL),
        t(795, 122, 'Trainer', 26, BL, 600, 'middle'),
        t(795, 153, 'FSDP · faithful kernels', 18, MUTE, 400, 'middle'),
    ]
    s1 = g(1,
           f'<rect x="330" y="232" width="300" height="158" rx="12" fill="{PANEL}"/>',
           t(480, 257, 'probability of the sampled token', 16.5, MUTE, 400, 'middle'),
           f'<rect x="356" y="268" width="12" height="12" fill="{OR}"/>', t(374, 280, 'sampler', 15.5, SOFT),
           f'<rect x="456" y="268" width="12" height="12" fill="{BL}"/>', t(474, 280, 'trainer', 15.5, SOFT),
           # likely token: the two agree
           f'<rect x="372" y="294" width="24" height="62" fill="{OR}"/>', f'<rect x="400" y="296" width="24" height="60" fill="{BL}"/>',
           # unlikely token: trainer far below sampler
           f'<rect x="532" y="338" width="24" height="18" fill="{OR}"/>', f'<rect x="560" y="353" width="24" height="3" fill="{BL}"/>',
           f'<path d="M350 356 H610" stroke="{RULE}" stroke-width="2"/>',
           t(398, 379, 'likely token', 15.5, SOFT, 400, 'middle'), t(558, 379, 'unlikely token', 15.5, SOFT, 400, 'middle'))
    s2 = g(2,
           box(40, 250, 250, 70, MUTE, PANEL, 2.5, '7 6'),
           t(165, 281, 'tool response', 20, INK, 600, 'middle'),
           t(165, 305, 'out of distribution', 17, MUTE, 400, 'middle'),
           arrow([(165, 250), (165, 189)], MUTE),
           t(178, 227, 'more unlikely tokens', 17, SOFT),
           t(480, 414, 'the size of the gap depends on the GPU', 16.5, MUTE, 400, 'middle'))
    s3 = g(3,
           arrow([(795, 180), (795, 242)]),
           box(670, 250, 250, 70, INK),
           t(795, 281, 'gradient spike', 20, INK, 600, 'middle'),
           t(795, 305, 'biased, noisy update', 17, MUTE, 400, 'middle'))
    s4 = g(4,
           arrow([(795, 320), (795, 466), (698, 466)], MUTE, 2.5, '7 6'),
           box(270, 434, 420, 64, MUTE, PANEL, 2.5, '7 6'),
           t(480, 461, 'weights move to where rounding matters more', 17, INK, 600, 'middle'),
           t(480, 484, 'the gap widens (hypothesis)', 16.5, MUTE, 400, 'middle'),
           arrow([(270, 466), (18, 466), (18, 52), (157, 52)], MUTE, 2.5, '7 6'))
    s5 = g(5,
           f'<rect x="566" y="104" width="68" height="52" rx="10" fill="{OR}"/>',
           t(600, 137, 'mask', 20, '#fff', 600, 'middle'),
           t(600, 210, 'drop responses outside the trust region', 17, OR, 600, 'middle'))
    figure('rlc-loop',
           'The post in one picture: a sampler and a trainer share weights but disagree on unlikely tokens, which leads to gradient spikes and a hypothesized feedback loop; the mask sits between rollout and trainer',
           W, H, [svg(W, H, *base, s1, s2, s3, s4, s5)], [
               (0, 0, 'One set of weights, two implementations. The sampler generates the rollouts; the trainer scores them and computes the gradient.'),
               (0, 1, 'The two engines agree on likely tokens. On unlikely tokens the trainer\'s probability can be orders of magnitude below the sampler\'s (Finding 2).'),
               (0, 2, 'Out-of-distribution tool responses make the model sample unlikely tokens more often, and the size of the gap depends on the GPU (Findings 3 and 4).'),
               (0, 3, 'On those batches the trainer\'s perplexity and the gradient norm spike, and the update is biased and noisy (Finding 1).'),
               (0, 4, 'Hypothesis: the update moves the weights to where rounding matters more, the gap widens, and the loop closes (Finding 5).'),
               (0, 5, 'The fix: mask whole responses whose importance ratio leaves the trust region, before they reach the loss (MIS and Geo-MIS, §4).'),
           ], note='Schematic. Bar heights are illustrative.')


# ---------------------------------------------------------------------------------------------
# 2. What a token-level weight corrects, and what a sequence-level weight corrects.
# ---------------------------------------------------------------------------------------------
def tokseq():
    W, H = 960, 380
    ratios = [0.9, 1.1, 0.8, 1.0, 0.7, 1.1]
    X0, CW, GAP, Y, CH = 120, 110, 12, 96, 64

    def row(hi=(), color=BL):
        out = [t(40, 56, f'ratio of token {m("t")}:  {m("ρ", "t")} = trainer probability / sampler probability', 18, SOFT)]
        for i, r in enumerate(ratios):
            x = X0 + i * (CW + GAP)
            on = i in hi
            out.append(f'<rect x="{x}" y="{Y}" width="{CW}" height="{CH}" rx="10" fill="{PANEL}" stroke="{color if on else INK}" stroke-width="{5 if on else 2}"/>')
            out.append(t(x + CW / 2, Y + 42, m('y', str(i + 1)), 28, INK, 400, 'middle'))
            out.append(t(x + CW / 2, Y + CH + 30, f'{m("ρ", str(i + 1))} = {r}', 18, color if on else MUTE, 600 if on else 400, 'middle'))
        return ''.join(out)

    def brace(i0, i1, y, color):
        x0 = X0 + i0 * (CW + GAP)
        x1 = X0 + i1 * (CW + GAP) + CW
        xm = (x0 + x1) / 2
        return (f'<path d="M{x0} {y} v8 H{xm - 10} l10 9 l10 -9 H{x1} v-8" fill="none" stroke="{color}" stroke-width="2.5" stroke-linejoin="round"/>')

    f0 = svg(W, H, row(),
             g(1,
               f'<rect x="{X0 + 4 * (CW + GAP)}" y="{Y}" width="{CW}" height="{CH}" rx="10" fill="none" stroke="{BL}" stroke-width="5"/>',
               t(X0 + 4 * (CW + GAP) + CW / 2, Y - 14, 'token-level weight', 16, BL, 600, 'middle'),
               t(40, 300, f'Token-level: the gradient term of {m("y", "5")} is weighted by {m("ρ", "5")} = 0.7.', 20, BL, 600)),
             g(2,
               brace(0, 3, Y + CH + 44, OR),
               t(X0 + 2 * (CW + GAP) - GAP / 2, Y + CH + 86, 'prefix', 16, OR, 600, 'middle'),
               t(40, 340, f'Left out: the prefix ratio {m("ρ", "1")}{m("ρ", "2")}{m("ρ", "3")}{m("ρ", "4")} = 0.79.', 20, OR, 600)))
    prod = 1.0
    for r in ratios:
        prod *= r
    f1 = svg(W, H, row(range(6)),
             brace(0, 5, Y + CH + 44, BL),
             t(40, 300, f'Sequence-level: every term is weighted by {m("ρ")} = {m("ρ", "1")}{m("ρ", "2")}{m("ρ", "3")}{m("ρ", "4")}{m("ρ", "5")}{m("ρ", "6")} = {prod:.2f}.', 20, BL, 600),
             t(40, 340, 'Prefix and token are both corrected.', 20, INK, 400))
    figure('rlc-tokseq',
           'A response of six tokens with per-token ratios. A token-level weight uses one ratio and leaves out the ratio of the prefix; a sequence-level weight uses the product of all of them',
           W, H, [f0, f1], [
               (0, 0, 'A response of six tokens. Each token has a ratio: the trainer\'s probability of that token over the sampler\'s.'),
               (0, 1, 'Token-level importance sampling weights the gradient term of each token by that token\'s ratio alone. It corrects how likely the token is, given its prefix.'),
               (0, 2, 'The prefix was drawn by the sampler too. Under the trainer it is 0.79 times as likely, and the token-level weight leaves that factor out. The factor is a product, so the error compounds with length.'),
               (1, 0, 'Sequence-level importance sampling weights every term by the product over the whole response, 0.61 here. The estimator is unbiased; the price is a weight whose variance grows with length.'),
           ], note='Illustrative ratios.')


# ---------------------------------------------------------------------------------------------
# 3. Weight as a function of the ratio: IS, TIS, MIS, then Geo-MIS.
# ---------------------------------------------------------------------------------------------
def weights():
    W, H = 960, 462
    X0, X1, Y0, Y1 = 90, 610, 370, 70

    def axes(xlabel, ylabel):
        return (f'<rect x="56" y="36" width="590" height="406" rx="12" fill="{PANEL}"/>'
                + arrow([(X0, Y0), (X1 + 22, Y0)], INK, 2) + arrow([(X0, Y0), (X0, Y1 - 22)], INK, 2)
                + t((X0 + X1) / 2, Y0 + 60, xlabel, 18, SOFT, 400, 'middle') + t(X0 - 10, Y1 - 28, ylabel, 18, SOFT))

    # frame 0: product form, rho in [0, 5], threshold C = 2
    sx = (X1 - X0) / 5.0
    sy = (Y0 - Y1) / 5.0
    X = lambda r: X0 + r * sx
    Yw = lambda w: Y0 - w * sy
    C, R = 2.0, 4.0
    f0 = svg(W, H,
             axes(f'ratio {m("ρ")}', 'weight'),
             f'<path d="M{X(C)} {Y0} v8" stroke="{INK}" stroke-width="2"/>', t(X(C), Y0 + 30, m('C'), 20, INK, 400, 'middle'),
             f'<path d="M{X(R)} {Y0} v8" stroke="{INK}" stroke-width="2"/>', t(X(R), Y0 + 30, f'2{m("C")}', 20, INK, 400, 'middle'),
             f'<path d="M{X(0)} {Yw(0)} L{X(C)} {Yw(C)}" stroke="{INK}" stroke-width="4.5" stroke-linecap="round"/>',
             f'<path d="M{X(R)} {Y0} V{Yw(R) - 6}" stroke="{RULE}" stroke-width="2" stroke-dasharray="4 5"/>',
             f'<rect x="676" y="36" width="254" height="406" rx="12" fill="{PANEL}"/>',
             t(696, 76, f'a response with {m("ρ")} = 2{m("C")}', 18, INK, 600),
             t(696, 104, 'gets the weight', 17, MUTE),
             # IS
             f'<path d="M{X(C)} {Yw(C)} L{X(5)} {Yw(5)}" stroke="{MUTE}" stroke-width="3.5" stroke-dasharray="9 7"/>',
             f'<circle cx="{X(R)}" cy="{Yw(R)}" r="7" fill="{MUTE}"/>',
             t(X(5) - 6, Yw(5) + 34, 'IS', 19, MUTE, 600, 'end'),
             t(696, 158, 'IS', 20, MUTE, 600), t(910, 158, f'2{m("C")}', 22, MUTE, 600, 'end'),
             g(1,
               f'<path d="M{X(C)} {Yw(C)} H{X(5)}" stroke="{PU}" stroke-width="4.5"/>',
               f'<circle cx="{X(R)}" cy="{Yw(C)}" r="7" fill="{PU}"/>',
               t(X(5), Yw(C) - 12, 'TIS', 19, PU, 600, 'end'),
               t(696, 208, 'TIS: capped', 20, PU, 600), t(910, 208, m('C'), 22, PU, 600, 'end')),
             g(2,
               f'<path d="M{X(C)} {Yw(C)} V{Y0}" stroke="{OR}" stroke-width="2.5" stroke-dasharray="5 5"/>',
               f'<path d="M{X(C)} {Y0} H{X(5)}" stroke="{OR}" stroke-width="6"/>',
               f'<circle cx="{X(C)}" cy="{Yw(C)}" r="6.5" fill="{OR}"/>',
               f'<circle cx="{X(R)}" cy="{Y0}" r="7" fill="{OR}"/>',
               t(X(5), Y0 - 14, 'MIS', 19, OR, 600, 'end'),
               t(696, 258, 'MIS: dropped', 20, OR, 600), t(910, 258, '0', 22, OR, 600, 'end')))

    # frame 1: geometric form, rho_geo in [0.6, 1.6], band [0.8, 1.25]
    gx = (X1 - X0) / 1.0
    gy = (Y0 - Y1) / 1.6
    GX = lambda r: X0 + (r - 0.6) * gx
    GY = lambda w: Y0 - w * gy
    lo, hi = 0.8, 1.25
    f1 = svg(W, H,
             axes(f'per-token geometric mean {m("ρ", "geo")}', 'weight'),
             f'<rect x="{GX(lo)}" y="{Y1}" width="{GX(hi) - GX(lo)}" height="{Y0 - Y1}" fill="{OR}" fill-opacity="0.08"/>',
             ''.join(f'<path d="M{GX(v)} {Y0} v8" stroke="{INK}" stroke-width="2"/>' for v in (lo, 1.0, hi)),
             t(GX(lo), Y0 + 30, m('C', 'low'), 20, INK, 400, 'middle'), t(GX(1.0), Y0 + 30, '1', 19, INK, 400, 'middle'), t(GX(hi), Y0 + 30, m('C', 'high'), 20, INK, 400, 'middle'),
             f'<path d="M{X0} {Y0} H{GX(lo)}" stroke="{OR}" stroke-width="6"/>', f'<path d="M{GX(hi)} {Y0} H{X1}" stroke="{OR}" stroke-width="6"/>',
             f'<path d="M{GX(lo)} {Y0} V{GY(lo)}" stroke="{OR}" stroke-width="2.5" stroke-dasharray="5 5"/>',
             f'<path d="M{GX(hi)} {Y0} V{GY(hi)}" stroke="{OR}" stroke-width="2.5" stroke-dasharray="5 5"/>',
             f'<path d="M{GX(lo)} {GY(lo)} L{GX(hi)} {GY(hi)}" stroke="{OR}" stroke-width="4.5" stroke-linecap="round"/>',
             f'<circle cx="{GX(lo)}" cy="{GY(lo)}" r="6.5" fill="{OR}"/>', f'<circle cx="{GX(hi)}" cy="{GY(hi)}" r="6.5" fill="{OR}"/>',
             t((GX(lo) + GX(hi)) / 2, Y1 + 26, 'kept', 18, OR, 600, 'middle'),
             t((X0 + GX(lo)) / 2, Y0 - 16, 'dropped', 17, OR, 600, 'middle'), t((GX(hi) + X1) / 2, Y0 - 16, 'dropped', 17, OR, 600, 'middle'),
             f'<rect x="676" y="36" width="254" height="406" rx="12" fill="{PANEL}"/>',
             t(696, 76, 'same per-token drift,', 18, INK, 600), t(696, 100, 'two lengths', 18, INK, 600),
             t(696, 132, f'log {m("ρ", "t")} = 0.01 at every token', 16, MUTE),
             f'<path d="M696 150 H910" stroke="{RULE}" stroke-width="2"/>',
             t(696, 184, f'{m("T")} = 100', 19, INK, 600),
             t(696, 212, f'{m("ρ")} = 2.7', 18, SOFT), t(910, 212, f'{m("ρ", "geo")} = 1.01', 18, OR, 600, 'end'),
             f'<path d="M696 232 H910" stroke="{RULE}" stroke-width="2"/>',
             t(696, 266, f'{m("T")} = 1000', 19, INK, 600),
             t(696, 294, f'{m("ρ")} ≈ 22,000', 18, SOFT), t(910, 294, f'{m("ρ", "geo")} = 1.01', 18, OR, 600, 'end'),
             f'<path d="M696 314 H910" stroke="{RULE}" stroke-width="2"/>',
             t(696, 348, 'One band judges both', 17, INK), t(696, 372, 'the same way.', 17, INK))
    figure('rlc-weights',
           'Weight given to a response as a function of its importance ratio: importance sampling grows without bound, truncation caps it at C, masking sets it to zero above C, and Geo-MIS keeps a two-sided band of the per-token geometric mean',
           W, H, [f0, f1], [
               (0, 0, 'Sequence-level importance sampling (IS) weights a response by its ratio. It is unbiased, but one response with a large ratio can dominate the batch.'),
               (0, 1, 'Truncation (TIS) caps the weight at C. The response stays in the batch.'),
               (0, 2, 'Masking (MIS) removes it: the weight is the ratio up to C and zero beyond.'),
               (1, 0, 'Geo-MIS applies the mask to the per-token geometric mean, on both sides. Responses of very different lengths with the same per-token drift get the same verdict.'),
           ])


# ---------------------------------------------------------------------------------------------
# 4. Mean KL and max KL along one response: what each criterion catches.
# ---------------------------------------------------------------------------------------------
def kl():
    W, H = 960, 430
    N, X0, PITCH, BW, Y0 = 20, 126, 25, 17, 350
    SCALE = 1400.0                    # px per nat
    D_MAX, D_AVG = 0.10, 0.03
    Yv = lambda v: Y0 - v * SCALE

    def lcg(seed):
        x = seed
        while True:
            x = (1103515245 * x + 12345) % 2147483648
            yield x / 2147483648.0

    r = lcg(7)
    low = [0.004 + 0.014 * next(r) for _ in range(N)]
    spike = list(low)
    spike[13] = 0.16
    r = lcg(11)
    drift = [0.032 + 0.028 * next(r) for _ in range(N)]

    def frame(vals, verdict_mean, verdict_max, outcome, color):
        mean, mx = sum(vals) / N, max(vals)
        imax = vals.index(mx)
        bars = ''.join(f'<rect x="{X0 + i * PITCH}" y="{Yv(v):.1f}" width="{BW}" height="{v * SCALE:.1f}" fill="{OR if (i == imax and mx > D_MAX) else BAR}"/>' for i, v in enumerate(vals))
        x_end = X0 + N * PITCH
        ok = lambda b: ('✓', INK) if b else ('✗', OR)
        (s1, c1), (s2, c2) = ok(verdict_mean), ok(verdict_max)
        return svg(W, H,
                   f'<rect x="56" y="36" width="668" height="372" rx="12" fill="{PANEL}"/>',
                   t(76, 68, f'token-level KL  {m("D", "t")}  along one response', 18, SOFT),
                   t(76, 94, f'dashed: thresholds on the max ({m("δ")}) and on the mean ({m("δ", "avg")})', 16, MUTE),
                   f'<path d="M{X0 - 8} {Y0} H{x_end + 4}" stroke="{INK}" stroke-width="2"/>',
                   t((X0 + x_end) / 2, Y0 + 30, f'token position {m("t")}', 17, SOFT, 400, 'middle'),
                   bars,
                   # thresholds
                   f'<path d="M{X0 - 8} {Yv(D_MAX)} H{x_end + 4}" stroke="{INK}" stroke-width="2" stroke-dasharray="8 6"/>',
                   t(x_end + 16, Yv(D_MAX) + 7, m('δ'), 22, INK),
                   f'<path d="M{X0 - 8} {Yv(D_AVG)} H{x_end + 4}" stroke="{BL}" stroke-width="2" stroke-dasharray="8 6"/>',
                   t(x_end + 16, Yv(D_AVG) + 7, m('δ', 'avg'), 22, BL),
                   # the mean of this response
                   f'<path d="M{X0 - 8} {Yv(mean):.1f} H{x_end + 4}" stroke="{BL}" stroke-width="4"/>',
                   t(X0 - 16, Yv(mean) + 6, 'mean', 17, BL, 600, 'end'),
                   # verdicts
                   f'<rect x="744" y="36" width="186" height="372" rx="12" fill="{PANEL}"/>',
                   t(764, 96, 'mean KL', 19, BL, 600), t(910, 96, s1, 26, c1, 600, 'end'),
                   t(764, 124, f'at most {m("δ", "avg")}?', 16.5, MUTE),
                   f'<path d="M764 150 H910" stroke="{RULE}" stroke-width="2"/>',
                   t(764, 196, 'max KL', 19, INK, 600), t(910, 196, s2, 26, c2, 600, 'end'),
                   t(764, 224, f'at most {m("δ")}?', 16.5, MUTE),
                   f'<path d="M764 250 H910" stroke="{RULE}" stroke-width="2"/>',
                   t(837, 310, outcome, 26, color, 600, 'middle'))

    figure('rlc-kl',
           'Token-level KL along one response under three patterns: uniformly small (kept), one spike (caught only by the max criterion), many small deviations (caught only by the mean criterion)',
           W, H, [frame(low, True, True, 'kept', INK), frame(spike, True, False, 'masked', OR), frame(drift, False, True, 'masked', OR)], [
               (0, 0, 'A response that stays close to the sampler: the mean of the token-level KLs is under its threshold and so is the largest one. It is kept.'),
               (1, 0, 'One token with a large KL. The mean barely moves and still passes; only the max criterion catches it.'),
               (2, 0, 'Many small deviations. No single token stands out, so the max criterion passes; the mean catches the accumulated drift.'),
           ], note='Illustrative values. Geo-MIS estimates the mean from the sampled tokens; Trust Region Masking computes both from the full logits.')


if __name__ == '__main__':
    loop()
    tokseq()
    weights()
    kl()
