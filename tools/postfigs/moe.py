#!/usr/bin/env python3
"""Interactive figures for the post "Mixture of Experts does not work the way its name says" (how-moe-actually-works).

    python3 tools/postfigs/moe.py

Writes static/figs/embed/moe-*.html, the pages that the {{< livefig "moe-<name>" >}} shortcode frames.
Each page is self-contained: the shared styles and helpers below, then the figure's own markup and script.
A page tells its parent how tall it is with postMessage({h}), as the pages from tools/notefigs/build.mjs do.
Figure 7 replays tools/postfigs/moe_traces.json, recorded from the toy MoE of the post. Figures 1 to 3 are
illustrations, Figure 5 iterates the derived router update, Figure 8 plots the formula above it.
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, 'static', 'figs', 'embed')

# The palette of tools/notefigs/lib.mjs, plus eight series colors for the experts of Figure 2.
C = dict(PANEL='#F0EDE4', CARD='#FAF8F3', INK='#161A22', SOFT='#3F4654', MUTE='#666C78', RULE='#C9C2B2',
         TRACK='#E4DFD2', GRAY='#9AA0AC', OR='#C2410C', ORSOFT='#F6D9C2', BL='#1D4ED8', BLSOFT='#DCE5FB', BLMID='#7C9BE8')
SERIES = ['#257DE0', '#F47230', '#00B894', '#F2A600', '#EE7FAC', '#008A17', '#6A4CE2', '#EC4647']

CSS = r'''
html,body{margin:0;padding:0;background:transparent;color:«INK»;font-family:"IBM Plex Sans",Arial,sans-serif}
#wrap{display:flow-root;padding-bottom:2px}
.fig{overflow-x:auto}
.fig svg{display:block;width:100%;min-width:600px;height:auto}
.ctl{display:flex;flex-wrap:wrap;gap:10px 22px;align-items:center;margin:2px 4px 12px;font-size:15px;color:«SOFT»}
.ctl label{display:flex;align-items:center;gap:10px}
.ctl input[type=range]{width:170px;max-width:42vw;accent-color:«OR»}
.ctl output{min-width:2.2em;font-variant-numeric:tabular-nums;color:«INK»;font-weight:500}
button{font:500 14px "IBM Plex Sans",Arial,sans-serif;color:«INK»;background:«CARD»;border:1px solid «RULE»;border-radius:8px;padding:6px 12px;cursor:pointer}
button:hover{border-color:«OR»;color:«OR»}
button:focus-visible,input:focus-visible{outline:2px solid «OR»;outline-offset:2px}
.seg{display:inline-flex;flex-wrap:wrap;border:1px solid «RULE»;border-radius:8px;overflow:hidden}
.seg button{border:0;border-right:1px solid «RULE»;border-radius:0}
.seg button:last-child{border-right:0}
.seg button[aria-pressed="true"],.seg button[aria-pressed="true"]:hover{background:«INK»;color:«CARD»}
@media (max-width:480px){.seg{flex-direction:column;align-items:stretch}.seg button{border-right:0;border-bottom:1px solid «RULE»;text-align:left}.seg button:last-child{border-bottom:0}}
#read{margin:10px 4px 2px;font-size:15px;line-height:1.5;color:«SOFT»;font-variant-numeric:tabular-nums}
#read b{color:«INK»;font-weight:600}
i,.var{font-family:"Source Serif 4",Georgia,serif;font-style:italic}
.key{display:flex;flex-wrap:wrap;gap:6px 20px;margin:10px 4px 0;font-size:14px;color:«SOFT»}
.key>span{display:inline-flex;align-items:center;gap:7px}
.sw{width:13px;height:13px;border-radius:3px;display:inline-block}
.box{background:«PANEL»;border-radius:16px;padding:14px 16px}
svg text{font-family:"IBM Plex Sans",Arial,sans-serif}
svg .var{font-family:"Source Serif 4",Georgia,serif;font-style:italic}
.lbl{font-size:12.5px;fill:«SOFT»}.lbl.on{fill:«INK»}
.lbl-strong{font-size:13.5px;font-weight:600;fill:«INK»}
.node{fill:«CARD»;stroke:«INK»;stroke-width:1.5}
.fan{stroke:«RULE»;stroke-width:1}
.trk{fill:«TRACK»}
.bar{fill:«GRAY»;transform-box:fill-box;transform-origin:left center;transition:transform .45s ease,fill .25s}
.bar.on{fill:«BL»}
.exp{fill:«CARD»;stroke:«RULE»;stroke-width:1.2;transition:fill .25s,stroke .25s}
.exp.on{fill:«BLSOFT»;stroke:«BL»}
.lnk{stroke:«BL»;stroke-width:2;opacity:0;stroke-dasharray:5 5;transition:opacity .25s}
.lnk.on{opacity:1;animation:dash .9s linear infinite}
.flow{stroke:«INK»;stroke-width:1.6;fill:none;stroke-dasharray:6 5;animation:dash 1.1s linear infinite}
@keyframes dash{to{stroke-dashoffset:-11}}
@media (prefers-reduced-motion: reduce){.lnk.on,.flow{animation:none}.bar,.exp,.u,.tok,.tb,.out{transition:none !important}}
'''

JS = r'''
var RM=!!(window.matchMedia&&window.matchMedia('(prefers-reduced-motion: reduce)').matches);
var NS='http://www.w3.org/2000/svg';
function $(s,r){return (r||document).querySelector(s)}
function S(n,a,p,t){var e=document.createElementNS(NS,n);for(var k in a)e.setAttribute(k,a[k]);if(t!=null)e.textContent=t;if(p)p.appendChild(e);return e}
function H(n,a,p,t){var e=document.createElement(n);for(var k in a){if(k==='class')e.className=a[k];else e.setAttribute(k,a[k])}if(t!=null)e.textContent=t;if(p)p.appendChild(e);return e}
function fnv(s){var h=0x811c9dc5;for(var i=0;i<s.length;i++){h^=s.charCodeAt(i);h=Math.imul(h,0x01000193)}return h>>>0}
function rng(a){return function(){a|=0;a=a+0x6D2B79F5|0;var t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}
function gauss(r){var u=0,v=0;while(u===0)u=r();while(v===0)v=r();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v)}
function softmax(a){var m=Math.max.apply(null,a);var e=a.map(function(x){return Math.exp(x-m)});var z=e.reduce(function(p,c){return p+c},0);return e.map(function(x){return x/z})}
function topk(a,k){return a.map(function(v,i){return i}).sort(function(i,j){return a[j]-a[i]}).slice(0,k)}
function clear(n){while(n.lastChild)n.removeChild(n.lastChild)}
// The rounded panel behind a drawing, as in the other slider figures of the site.
function panel(svg){var b=svg.viewBox.baseVal;return S('rect',{x:b.x,y:b.y,width:b.width,height:b.height,rx:16,fill:'«PANEL»'},svg)}
// VIS follows whether the figure is on screen, so that a figure nobody is looking at stays still.
var VIS=true;
if('IntersectionObserver' in window){new IntersectionObserver(function(es){VIS=es[es.length-1].isIntersecting}).observe(document.documentElement)}
function onView(el,fn){
  if(!el) return;
  if(!('IntersectionObserver' in window)){fn();return}
  var io=new IntersectionObserver(function(es){es.forEach(function(e){if(e.isIntersecting){io.disconnect();fn()}})},{threshold:0.3});
  io.observe(el);
}
function post(){parent.postMessage({livefig:«NAME»,h:document.getElementById('wrap').offsetHeight},'*')}
window.addEventListener('resize',post);window.addEventListener('load',post);
if('ResizeObserver' in window){new ResizeObserver(post).observe(document.getElementById('wrap'))}
if(document.fonts&&document.fonts.ready){document.fonts.ready.then(post)}
'''

FIGS = []


def fig(name, title, css, body, js):
    FIGS.append(dict(name=name, title=title, css=css, body=body, js=js))


# ---------------------------------------------------------------- Figure 1: one token through the layer
fig('moe-forward', 'One token through an MoE layer', r'''
.tokens{display:flex;flex-wrap:wrap;gap:5px;margin:0 4px 10px}
.tk{font-size:14px;padding:2px 7px;border:1px solid «RULE»;border-radius:6px;color:«SOFT»;background:«CARD»}
.tk.cur{background:«INK»;color:«CARD»;border-color:«INK»}
''', r'''
<div class="ctl">
<label for="k"><span>Experts run per token, <i>K</i></span><input type="range" id="k" min="1" max="8" step="1" value="2"><output id="kout" for="k">2</output></label>
<button id="play" type="button">Pause</button>
<button id="next" type="button">Next token</button>
</div>
<div class="tokens" id="tokens" aria-hidden="true"></div>
<div class="fig"><svg id="svg" viewBox="-10 -20 780 344" role="img" aria-label="A token vector x is scored against eight experts by the router. The K highest-scoring experts run and their outputs are summed with gate weights into y. The other experts stay idle."></svg></div>
<div id="read" aria-live="off"></div>
''', r'''
(function(){
  var svg=$('#svg'); panel(svg);
  var defs=S('defs',{},svg), mk=S('marker',{id:'arrow',viewBox:'0 0 10 10',refX:9,refY:5,markerWidth:7,markerHeight:7,orient:'auto'},defs);
  S('path',{d:'M0 0L10 5L0 10z',fill:'«INK»'},mk);
  var M=8, K=2, ti=0, timer=null, playing=!RM;
  var words=['The','router','scores','every','expert',',','runs','a','few',',','and','skips','the','rest','.'];
  var strip=$('#tokens');
  var spans=words.map(function(w){return H('span',{class:'tk'},strip,w)});
  function rowY(i){return 34+i*36}
  S('text',{x:210,y:8,'text-anchor':'middle',class:'lbl'},svg,'router score h');
  S('text',{x:405,y:8,'text-anchor':'middle',class:'lbl'},svg,'experts');
  var rows=[];
  for(var i=0;i<M;i++){
    var y=rowY(i);
    S('line',{x1:76,y1:160,x2:150,y2:y,class:'fan'},svg);
    S('rect',{x:150,y:y-6,width:120,height:12,rx:3,class:'trk'},svg);
    var bar=S('rect',{x:150,y:y-6,width:120,height:12,rx:3,class:'bar'},svg);
    var lnk=S('line',{x1:272,y1:y,x2:328,y2:y,class:'lnk'},svg);
    var out=S('line',{x1:482,y1:y,x2:592,y2:160,class:'lnk'},svg);
    var box=S('rect',{x:330,y:y-13,width:150,height:26,rx:5,class:'exp'},svg);
    S('text',{x:342,y:y+5,class:'lbl-strong'},svg,'E'+(i+1));
    var st=S('text',{x:468,y:y+4,'text-anchor':'end',class:'lbl'},svg,'idle');
    var gt=S('text',{x:494,y:(y<160?y-7:y+17),class:'lbl on'},svg,'');
    rows.push({bar:bar,lnk:lnk,box:box,st:st,out:out,gt:gt});
  }
  S('circle',{cx:56,cy:160,r:20,class:'node'},svg);
  S('text',{x:56,y:165,'text-anchor':'middle',class:'lbl-strong var','font-size':17},svg,'x');
  S('circle',{cx:612,cy:160,r:20,class:'node'},svg);
  S('text',{x:612,y:165,'text-anchor':'middle',class:'lbl-strong'},svg,'Σ');
  S('line',{x1:634,y1:160,x2:692,y2:160,stroke:'«INK»','stroke-width':1.6,'marker-end':'url(#arrow)'},svg);
  S('text',{x:710,y:165,'text-anchor':'middle',class:'lbl-strong var','font-size':17},svg,'y');
  function logits(w){var r=rng(fnv(w)^0x9e3779b9),a=[];for(var i=0;i<M;i++)a.push(1.3*gauss(r));return a}
  function draw(){
    var w=words[ti];
    spans.forEach(function(s,i){s.classList.toggle('cur',i===ti)});
    var h=logits(w), sel=topk(h,K), mn=Math.min.apply(null,h), mx=Math.max.apply(null,h);
    var g=softmax(sel.map(function(i){return h[i]}));
    var parts=[];
    rows.forEach(function(r,i){
      var pos=sel.indexOf(i), on=pos>=0;
      r.bar.style.transform='scaleX('+(0.06+0.94*(h[i]-mn)/(mx-mn)).toFixed(3)+')';
      [r.bar,r.lnk,r.box,r.out].forEach(function(e){e.classList.toggle('on',on)});
      r.st.textContent=on?'runs':'idle'; r.st.classList.toggle('on',on);
      r.gt.textContent=on?('g = '+g[pos].toFixed(2)):'';
    });
    sel.slice().sort(function(a,b){return a-b}).forEach(function(i){parts.push('E'+(i+1)+' '+g[sel.indexOf(i)].toFixed(2))});
    $('#read').textContent='8 experts stored. '+K+' run for “'+w+'”, which is '+K+'/8 of the expert parameters. Gates: '+parts.join(', ')+'.';
  }
  function step(){ti=(ti+1)%words.length;draw()}
  function setPlay(p){playing=p;$('#play').textContent=p?'Pause':'Play';if(timer){clearInterval(timer);timer=null}if(p)timer=setInterval(function(){if(VIS)step()},2400)}
  $('#k').addEventListener('input',function(e){K=+e.target.value;$('#kout').textContent=K;draw()});
  $('#play').addEventListener('click',function(){setPlay(!playing)});
  $('#next').addEventListener('click',function(){setPlay(false);step()});
  draw(); setPlay(playing);
})();
''')

# ---------------------------------------------------------------- Figure 2: routing by subject and by token identity
fig('moe-routes', 'Three passages routed by subject and by token identity', r'''
«SERIESCSS»
.chips{display:flex;flex-wrap:wrap;gap:6px;align-items:center}
.chip{font-size:13px;font-weight:600;border-radius:999px;padding:3px 10px 3px 7px;display:inline-flex;align-items:center;gap:6px}
.chip::before{content:"";width:11px;height:11px;border-radius:3px;background:var(--c)}
.chip[aria-pressed="true"],.chip[aria-pressed="true"]:hover{border-color:«INK»;box-shadow:0 0 0 1px «INK»;color:«INK»}
.card{display:grid;grid-template-columns:104px minmax(0,1fr) 150px;gap:14px;align-items:start;padding:12px 0;border-top:1px solid «RULE»}
.card:first-child{border-top:0;padding-top:2px}
.name{font-size:14px;font-weight:600;color:«SOFT»;padding-top:3px}
.text{font-size:15.5px;line-height:1.8;white-space:pre-wrap;overflow-wrap:anywhere;color:«INK»}
.text.mono{font:400 13.5px/1.8 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
.tok{background:color-mix(in srgb,var(--c) 28%,transparent);box-shadow:inset 0 -2px 0 var(--c);border-radius:2px;padding:0.05em 0.08em;transition:opacity .2s,background .15s}
.tok.dim{background:transparent;box-shadow:none;opacity:0.4}
.tok.pending{background:transparent;box-shadow:none}
.hist{display:grid;grid-template-columns:repeat(8,1fr);gap:2px;align-items:end;height:74px;border-bottom:1px solid «RULE»}
.hcol{display:flex;flex-direction:column;justify-content:flex-end;align-items:center;height:100%}
.hbar{width:100%;background:var(--c);border-radius:3px 3px 0 0;min-height:1px}
.hcol.dim .hbar{opacity:0.25}
.histlab{display:grid;grid-template-columns:repeat(8,1fr);gap:2px;font-size:11.5px;color:«MUTE»;text-align:center;margin-top:2px}
@media (max-width:560px){.card{grid-template-columns:1fr;gap:8px}}
''', r'''
<div class="ctl">
<div class="seg" role="group" aria-label="Routing rule">
<button type="button" id="subject" aria-pressed="false">By subject (the design story)</button>
<button type="button" id="hash" aria-pressed="true">By token identity (a hash)</button>
</div>
<button id="replay" type="button">Replay</button>
</div>
<div class="ctl"><span>Show one expert:</span><div class="chips" id="legend"></div></div>
<div class="box" id="body"></div>
''', r'''
(function(){
  var root=$('#body');
  var P=[
    {name:'Biology',subj:2,mono:false,text:'The enzyme binds the substrate at the active site and lowers the energy of the transition state. A mutation in the binding pocket can change the rate of the reaction.'},
    {name:'Contract law',subj:5,mono:false,text:'The tenant shall pay the rent on the first day of each month. A failure to pay within ten days is a breach of the lease, and the landlord may end the agreement.'},
    {name:'Python',subj:0,mono:true,text:'def route(x, k):\n    scores = router(x)\n    top = scores.topk(k)\n    if top is None:\n        return x\n    return mix(top, x)'}
  ];
  var mode='hash', iso=-1, reveal=Infinity, timer=null;
  function tokens(text){
    var out=[],re=/( {4})|(\n)|([A-Za-z_]+|\d+|[^\sA-Za-z_\d])|(\s)/g,m;
    while((m=re.exec(text))){
      if(m[1]) out.push({kind:'tok',t:'<indent>',raw:m[1]});
      else if(m[2]) out.push({kind:'ws',raw:'\n'});
      else if(m[3]) out.push({kind:'tok',t:m[3],raw:m[3]});
      else out.push({kind:'ws',raw:m[4]});
    }
    return out;
  }
  var legend=$('#legend'), chips=[];
  for(var e=0;e<8;e++){(function(e){
    var b=H('button',{type:'button',class:'chip e'+(e+1),'aria-pressed':'false'},legend,'E'+(e+1));
    b.addEventListener('click',function(){iso=(iso===e)?-1:e;render()});
    chips.push(b);
  })(e)}
  var maxTok=0;
  var cards=P.map(function(p){
    var card=H('div',{class:'card'},root);
    H('div',{class:'name'},card,p.name);
    var txt=H('div',{class:'text'+(p.mono?' mono':'')},card);
    var hw=H('div',{},card);
    var hist=H('div',{class:'hist',role:'img'},hw);
    var lab=H('div',{class:'histlab','aria-hidden':'true'},hw);
    for(var e=1;e<=8;e++)H('span',{},lab,String(e));
    var toks=tokens(p.text), n=toks.filter(function(k){return k.kind==='tok'}).length;
    if(n>maxTok) maxTok=n;
    return {p:p,txt:txt,hist:hist,toks:toks,n:n};
  });
  function expertOf(p,t){return mode==='subject'?p.subj:fnv(t)%8}
  function render(){
    $('#subject').setAttribute('aria-pressed',mode==='subject');
    $('#hash').setAttribute('aria-pressed',mode==='hash');
    chips.forEach(function(c,i){c.setAttribute('aria-pressed',iso===i)});
    cards.forEach(function(c){
      clear(c.txt); clear(c.hist);
      var cnt=[0,0,0,0,0,0,0,0], seen=0, idx=0;
      c.toks.forEach(function(k){
        if(k.kind==='tok'){
          var e=expertOf(c.p,k.t), shown=idx<reveal; idx++;
          if(shown){cnt[e]++; seen++}
          H('span',{class:'tok e'+(e+1)+(shown?'':' pending')+((shown&&iso>=0&&iso!==e)?' dim':''),title:shown?('first choice: expert '+(e+1)):''},c.txt,k.raw);
        } else c.txt.appendChild(document.createTextNode(k.raw));
      });
      var desc=[];
      for(var e=0;e<8;e++){
        var col=H('div',{class:'hcol'+((iso>=0&&iso!==e)?' dim':'')},c.hist);
        var pct=seen?Math.round(100*cnt[e]/seen):0;
        var b=H('div',{class:'hbar e'+(e+1),title:'Expert '+(e+1)+': '+pct+'% of tokens'},col);
        b.style.height=(cnt[e]?Math.max(3,Math.round(70*cnt[e]/seen)):0)+'px';
        desc.push('E'+(e+1)+' '+pct+'%');
      }
      c.hist.setAttribute('aria-label',c.p.name+', share of tokens per expert: '+desc.join(', '));
    });
  }
  function play(){
    if(timer){clearInterval(timer);timer=null}
    if(RM){reveal=Infinity;render();return}
    reveal=0; render();
    timer=setInterval(function(){reveal++;render();if(reveal>=maxTok){clearInterval(timer);timer=null;reveal=Infinity}},55);
  }
  $('#subject').addEventListener('click',function(){mode='subject';play()});
  $('#hash').addEventListener('click',function(){mode='hash';play()});
  $('#replay').addEventListener('click',play);
  render(); onView(root,play);
})();
''')

# ---------------------------------------------------------------- Figure 3: one wide hidden layer, cut into blocks
fig('moe-wide', 'An MoE layer as one wide hidden layer', r'''
.u{fill:«GRAY»;opacity:0.55;transition:fill .35s,opacity .35s}.u.open{fill:«BLMID»;opacity:0.55}.u.fire{fill:«BL»;opacity:1}
.blk{fill:none;stroke:«RULE»;stroke-width:1}.blk.on{stroke:«BL»;stroke-width:1.5}
.wire{stroke:«BL»;stroke-width:1.3;opacity:0.75}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px 20px;margin:12px 4px 0}
.stat b{display:block;font-size:20px;font-weight:600;color:«INK»;font-variant-numeric:tabular-nums}
.stat span{font-size:13.5px;color:«SOFT»}
''', r'''
<div class="ctl">
<div class="seg" role="group" aria-label="Block size">
<button type="button" data-g="1" aria-pressed="true">8 experts, 2 open</button>
<button type="button" data-g="2" aria-pressed="false">16 experts, 4 open</button>
<button type="button" data-g="4" aria-pressed="false">32 experts, 8 open</button>
</div>
<button id="play" type="button">Pause</button>
</div>
<div class="fig"><svg id="svg" viewBox="-10 -8 780 264" role="img" aria-label="A row of 64 hidden units grouped into blocks. For each token a few blocks are open and connected to the input and output. The rest are closed."></svg></div>
<div class="key"><span><i class="sw" style="background:«GRAY»;opacity:0.55"></i>unit in a closed block, not computed</span><span><i class="sw" style="background:«BLMID»;opacity:0.55"></i>unit in an open block</span><span><i class="sw" style="background:«BL»"></i>unit that responds to this token</span></div>
<div class="stats" id="stats" aria-live="off"></div>
''', r'''
(function(){
  var svg=$('#svg');
  var G=1, seed=5, U=64, M=8, K=2, pitch=9, uw=7, bgap=5, y0=104, uh=30, timer=null, playing=false;
  function binom(n,k){var r=1;for(var i=1;i<=k;i++){r=r*(n-k+i)/i}return Math.round(r)}
  function fmt(b){return String(b).replace(/\B(?=(\d{3})+(?!\d))/g,',')}
  var built=-1, units=[], blks=[], wires;
  function build(){
    clear(svg); panel(svg); units=[]; blks=[];
    var blocks=M*G, per=U/blocks, width=U*pitch-(pitch-uw)+(blocks-1)*bgap, x0=(760-width)/2, bw=per*pitch-(pitch-uw);
    wires=S('g',{},svg);
    for(var b=0;b<blocks;b++){
      var bx=x0+b*per*pitch+b*bgap;
      blks.push({el:S('rect',{x:bx-2.5,y:y0-3.5,width:bw+5,height:uh+7,rx:3,class:'blk'},svg),cx:bx+bw/2});
      for(var u=0;u<per;u++) units.push({el:S('rect',{x:bx+u*pitch,y:y0,width:uw,height:uh,rx:1.5,class:'u'},svg),b:b});
    }
    S('circle',{cx:380,cy:226,r:15,class:'node'},svg); S('text',{x:380,y:231,'text-anchor':'middle',class:'lbl-strong var','font-size':16},svg,'x');
    S('circle',{cx:380,cy:26,r:15,class:'node'},svg); S('text',{x:380,y:31,'text-anchor':'middle',class:'lbl-strong var','font-size':16},svg,'y');
    S('text',{x:14,y:18,class:'lbl'},svg,'one hidden layer, '+U+' units, in '+blocks+' blocks');
    var st=$('#stats'); clear(st);
    [[String(blocks),'blocks stored (experts)'],[String(K*G),'blocks opened per token'],['16 of 64','hidden units computed'],[fmt(binom(blocks,K*G)),'distinct sets of open blocks']].forEach(function(p){
      var d=H('div',{class:'stat'},st); H('b',{},d,p[0]); H('span',{},d,p[1]);
    });
    built=G;
  }
  function draw(){
    if(built!==G) build();
    var blocks=M*G, open=K*G, r=rng(seed*7919+G*13), ids=[];
    for(var i=0;i<blocks;i++)ids.push(i);
    for(var i=blocks-1;i>0;i--){var j=Math.floor(r()*(i+1)),t=ids[i];ids[i]=ids[j];ids[j]=t}
    var on={}; ids.slice(0,open).forEach(function(b){on[b]=true});
    clear(wires);
    blks.forEach(function(B,b){
      B.el.setAttribute('class','blk'+(on[b]?' on':''));
      if(on[b]){S('line',{x1:380,y1:212,x2:B.cx,y2:y0+uh+4,class:'wire'},wires);S('line',{x1:B.cx,y1:y0-4,x2:380,y2:40,class:'wire'},wires)}
    });
    units.forEach(function(u){u.el.setAttribute('class','u'+(on[u.b]?(r()<0.5?' fire':' open'):''))});
  }
  function setPlay(p){playing=p;$('#play').textContent=p?'Pause':'Play';if(timer){clearInterval(timer);timer=null}if(p)timer=setInterval(function(){if(VIS){seed++;draw()}},2000)}
  var segs=document.querySelectorAll('.seg button');
  Array.prototype.forEach.call(segs,function(b){
    b.addEventListener('click',function(){G=+b.getAttribute('data-g');
      Array.prototype.forEach.call(segs,function(o){o.setAttribute('aria-pressed',o===b)});draw()});
  });
  $('#play').addEventListener('click',function(){if(!playing){seed++;draw()}setPlay(!playing)});
  draw(); setPlay(false); if(!RM) onView(svg,function(){setPlay(true)});
})();
''')

# ---------------------------------------------------------------- Figure 4: experts named by what they were found to do
fig('moe-found', 'Experts, named by what they were found to do', r'''
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:10px}
.xcard{background:«CARD»;border:1px solid «RULE»;border-radius:10px;padding:10px 12px}
.xcard .id{font-size:11.5px;font-weight:600;color:«MUTE»;letter-spacing:0.04em}
.xcard .job{font-size:16px;font-weight:600;line-height:1.25;margin:4px 0 5px;color:«INK»}
.xcard .how{font-size:13.5px;line-height:1.4;color:«SOFT»}
.xcard.predicted{border-style:dashed;background:transparent}
.ex{font:500 13px/1.7 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;margin-top:8px;padding-top:7px;border-top:1px solid «RULE»;min-height:3.9em;color:«INK»}
.ex mark{background:«BLSOFT»;color:«INK»;padding:0 0.18em;border-radius:2px;box-shadow:inset 0 -2px 0 «BL»}
.ex .arr{color:«MUTE»;margin:0 0.3em}
.ex .out{display:inline-block;font-weight:600;border:1px solid «GRAY»;border-radius:4px;padding:0 0.3em;background:«CARD»;transition:opacity .25s}
.ex .out.hide{opacity:0}
.ex .cap{display:block;color:«MUTE»;font:400 12px/1.5 "IBM Plex Sans",Arial,sans-serif}
''', r'''
<div class="box"><div class="grid" id="grid">
<div class="xcard predicted"><div class="id">WHAT THE NAME PREDICTS</div><div class="job">&ldquo;The biology expert&rdquo;</div><div class="how">One expert per subject. Not found in any of the studies above.</div></div>
<div class="xcard"><div class="id">OLMoE &middot; LAYER 15 &middot; EXPERT 17</div><div class="job">Closes LaTeX delimiters</div><div class="how">Active inside formatting blocks and promotes the closing braces.</div>
<div class="ex" data-ctx="\mathbf{|b|" data-out="}}" data-cap="sees &rarr; promotes"></div></div>
<div class="xcard"><div class="id">ERNIE-4.5 &middot; LAYER 15 &middot; EXPERT 0</div><div class="job">Continues after a conjunction</div><div class="how">Fires on &ldquo;and&rdquo;, &ldquo;or&rdquo;, &ldquo;but&rdquo; and promotes likely completions.</div>
<div class="ex" data-ctx="boots |and| the" data-out="like,alike" data-cap="sees &rarr; promotes"></div></div>
<div class="xcard"><div class="id">OLMoE &middot; LAYER 1 &middot; EXPERT 57</div><div class="job">Glues sub-word pieces</div><div class="how">Fires on a piece of a rare chemical term and promotes sub-word continuations.</div>
<div class="ex" data-ctx="glut|amine|" data-out="iaz,endar,uba" data-cap="sees &rarr; promotes"></div></div>
<div class="xcard"><div class="id">OpenMoE &middot; EXPERT 31</div><div class="job">Modal verbs</div><div class="how">Collects these tokens wherever they occur, whatever the context.</div>
<div class="ex" data-ctx="" data-out="can,will,would" data-cap="tokens routed here"></div></div>
<div class="xcard"><div class="id">OLMoE &middot; LAYER 4 &middot; EXPERT 3</div><div class="job">Patent and legal boilerplate</div><div class="how">The closest thing to a subject expert in the sample. It still works by promoting a document style.</div>
<div class="ex" data-ctx="|patent text|" data-out="patents,applications,inventor" data-cap="sees &rarr; promotes"></div></div>
</div></div>
''', r'''
(function(){
  var exs=document.querySelectorAll('.ex'), items=[];
  Array.prototype.forEach.call(exs,function(el){
    var ctx=el.getAttribute('data-ctx'), outs=el.getAttribute('data-out').split(','), cap=el.getAttribute('data-cap');
    H('span',{class:'cap'},el,cap);
    if(ctx){
      var parts=ctx.split('|'), line=H('span',{},el);
      parts.forEach(function(p,i){if(!p) return; if(i%2===1) H('mark',{},line,p); else line.appendChild(document.createTextNode(p))});
      H('span',{class:'arr'},el,'→');
    }
    var nodes=outs.map(function(o){var s=H('span',{class:'out'},el,o); el.appendChild(document.createTextNode(' ')); return s});
    items.push(nodes);
  });
  if(RM) return;
  var tick=0;
  function frame(){
    if(!VIS) return;
    items.forEach(function(nodes){var n=nodes.length, show=tick%(n+1); nodes.forEach(function(s,i){s.classList.toggle('hide',i>=show)})});
    tick++;
  }
  onView($('#grid'),function(){frame();setInterval(frame,900)});
})();
''')

# ---------------------------------------------------------------- Figure 5: the derived router update, iterated
fig('moe-grad', 'The derived router update, iterated', r'''
.vec{stroke-width:2;fill:none}
''', r'''
<div class="ctl">
<label for="k"><span>Experts selected, <i>K</i></span><input type="range" id="k" min="1" max="5" step="1" value="2"><output id="kout" for="k">2</output></label>
<div class="seg" role="group" aria-label="Size of the router logits" id="scale">
<button type="button" data-s="1" aria-pressed="true">logits × 1</button>
<button type="button" data-s="2" aria-pressed="false">× 2</button>
<button type="button" data-s="3" aria-pressed="false">× 3</button>
</div>
<button id="replay" type="button">Replay</button>
</div>
<div class="fig"><svg id="svg" viewBox="-10 -12 780 360" role="img" aria-label="Left: five expert outputs drawn as arrows, with the direction that lowers the loss and the mixture the layer outputs. Right: the reward of each expert, and the router logits evolving under the derived update. Only selected experts move."></svg></div>
<div id="read" aria-live="off"></div>
''', r'''
(function(){
  var svg=$('#svg');
  var K=2, scale=1, O=[168,236], sc=112, ETA=0.4, TMAX=600, PER=2;
  var d=[0.6,0.8];
  var ang=[110,75,200,30,330], len=[1.0,1.0,0.8,1.3,0.9], h0=[1.2,0.9,0.5,0.1,-0.4];
  var E=ang.map(function(a,i){var t=a*Math.PI/180;return [len[i]*Math.cos(t),len[i]*Math.sin(t)]});
  var r=E.map(function(v){return v[0]*d[0]+v[1]*d[1]});
  var h, t, first4, raf=null, hmax=1, hmin=-1;
  function gatesNow(){var sel=topk(h,K), g=softmax(sel.map(function(i){return h[i]})), gate=[0,0,0,0,0]; sel.forEach(function(i,j){gate[i]=g[j]}); return gate}
  function stepOnce(){
    var gate=gatesNow(), rb=0; r.forEach(function(x,i){rb+=gate[i]*x});
    for(var i=0;i<5;i++) if(gate[i]>0) h[i]+=ETA*gate[i]*(r[i]-rb);
    t++; if(first4<0 && topk(h,K).indexOf(3)>=0) first4=t;
  }
  function reset(){h=h0.map(function(x){return x*scale}); t=0; first4=topk(h,K).indexOf(3)>=0?0:-1}
  function range(){ reset(); hmax=Math.max.apply(null,h); hmin=Math.min.apply(null,h);
    for(var i=0;i<TMAX;i++){stepOnce(); hmax=Math.max(hmax,Math.max.apply(null,h)); hmin=Math.min(hmin,Math.min.apply(null,h))}
    reset(); }
  function P(v){return [O[0]+sc*v[0],O[1]-sc*v[1]]}
  function arrow(v,color,dash){
    var p=P(v), L=Math.hypot(p[0]-O[0],p[1]-O[1]); if(L<1) return;
    var ux=(p[0]-O[0])/L, uy=(p[1]-O[1])/L, bx=p[0]-9*ux, by=p[1]-9*uy;
    var a={x1:O[0],y1:O[1],x2:bx,y2:by,class:'vec',stroke:color}; if(dash) a['stroke-dasharray']='4 4';
    S('line',a,svg);
    S('polygon',{points:[p[0],p[1],bx-4.5*uy,by+4.5*ux,bx+4.5*uy,by-4.5*ux].join(' '),fill:color},svg);
  }
  function label(v,txt,cls){var p=P(v),L=Math.hypot(v[0],v[1])||1;S('text',{x:p[0]+15*v[0]/L,y:p[1]-15*v[1]/L+4,'text-anchor':'middle',class:cls},svg,txt)}
  function draw(){
    clear(svg); panel(svg);
    S('rect',{x:8,y:26,width:352,height:310,rx:14,fill:'«CARD»'},svg);
    S('rect',{x:376,y:26,width:376,height:310,rx:14,fill:'«CARD»'},svg);
    var gate=gatesNow(), y=[0,0]; E.forEach(function(v,i){y[0]+=gate[i]*v[0];y[1]+=gate[i]*v[1]});
    S('text',{x:14,y:14,class:'lbl-strong'},svg,'expert outputs for one token');
    S('circle',{cx:O[0],cy:O[1],r:2.5,fill:'«INK»'},svg);
    E.forEach(function(v,i){var on=gate[i]>0;arrow(v,on?'«BL»':'«GRAY»',!on);label(v,'E'+(i+1),on?'lbl-strong':'lbl')});
    arrow([d[0]*1.42,d[1]*1.42],'«INK»',false);
    var pd=P([d[0]*1.42,d[1]*1.42]); S('text',{x:pd[0]+4,y:pd[1]-9,'text-anchor':'middle',class:'lbl-strong'},svg,'lowers the loss');
    var py=P(y);
    S('circle',{cx:py[0],cy:py[1],r:6.5,fill:'«CARD»',stroke:'«INK»','stroke-width':2},svg);
    S('text',{x:py[0]-11,y:py[1]-9,'text-anchor':'end',class:'lbl-strong var','font-size':16},svg,'y');
    var X0=430, DX=62, bw=30, baseA=116, sa=40, top=204, bot=328;
    var up=Math.max(hmax,0.2), dn=Math.max(-hmin,0.2), unit=Math.min((bot-top-36)/(up+dn),40), baseB=top+19+unit*up;
    S('text',{x:382,y:14,class:'lbl-strong'},svg,'reward of each expert, r (frozen)');
    S('text',{x:X0-36,y:top-8,class:'lbl-strong'},svg,'router logit, h (filled = selected)');
    S('line',{x1:X0-36,y1:baseA,x2:X0+5*DX-20,y2:baseA,stroke:'«GRAY»','stroke-width':1},svg);
    S('line',{x1:X0-36,y1:baseB,x2:X0+5*DX-20,y2:baseB,stroke:'«GRAY»','stroke-width':1},svg);
    for(var i=0;i<5;i++){
      var x=X0+i*DX, on=gate[i]>0, ha=sa*r[i];
      S('text',{x:x+bw/2,y:46,'text-anchor':'middle',class:on?'lbl-strong':'lbl'},svg,'E'+(i+1));
      S('rect',{x:x,y:ha>=0?baseA-ha:baseA,width:bw,height:Math.max(1,Math.abs(ha)),rx:3,fill:on?'«SOFT»':'«RULE»'},svg);
      S('text',{x:x+bw/2,y:ha>=0?baseA-ha-5:baseA+Math.abs(ha)+14,'text-anchor':'middle',class:'lbl'+(on?' on':'')},svg,(r[i]>=0?'+':'−')+Math.abs(r[i]).toFixed(2));
      var hb=unit*h[i];
      var a={x:x,y:hb>=0?baseB-hb:baseB,width:bw,height:Math.max(1.5,Math.abs(hb)),rx:3};
      if(on) a.fill='«BL»'; else {a.fill='none';a.stroke='«GRAY»';a['stroke-width']=1.2;a['stroke-dasharray']='3 3'}
      S('rect',a,svg);
      S('text',{x:x+bw/2,y:hb>=0?baseB-hb-5:baseB+Math.abs(hb)+14,'text-anchor':'middle',class:on?'lbl-strong':'lbl'},svg,(h[i]>=0?'':'−')+Math.abs(h[i]).toFixed(1));
    }
    var sel=topk(h,K).slice().sort(function(a,b){return a-b});
    var line='Step '+t+' of '+TMAX+'. Selected: '+sel.map(function(i){return 'E'+(i+1)+' (gate '+gate[i].toFixed(2)+')'}).join(', ')+'. ';
    if(K===1) line+='With one expert and a renormalized gate the router gets no gradient. Nothing moves.';
    else if(first4===0) line+='E4 is inside the top-K from the start.';
    else if(first4>0) line+='E4 was first tried at step '+first4+'.';
    else line+='E4 has not been tried yet.';
    $('#read').textContent=line;
  }
  function stop(){if(raf){cancelAnimationFrame(raf);raf=null}}
  function play(){
    stop(); range(); draw();
    if(RM||K===1){ if(RM&&K>1){for(var i=0;i<TMAX;i++)stepOnce(); draw()} return }
    function frame(){ for(var i=0;i<PER&&t<TMAX;i++) stepOnce(); draw(); if(t<TMAX) raf=requestAnimationFrame(frame); else raf=null }
    raf=requestAnimationFrame(frame);
  }
  $('#k').addEventListener('input',function(e){K=+e.target.value;$('#kout').textContent=K;play()});
  var segs=document.querySelectorAll('#scale button');
  Array.prototype.forEach.call(segs,function(b){
    b.addEventListener('click',function(){scale=+b.getAttribute('data-s');
      Array.prototype.forEach.call(segs,function(o){o.setAttribute('aria-pressed',o===b)});play()});
  });
  $('#replay').addEventListener('click',play);
  range(); draw(); onView(svg,play);
})();
''')

# ---------------------------------------------------------------- Figure 6: the feedback loop (a diagram, no controls)
fig('moe-loop', 'The feedback loop of top-K routing', '', r'''
<div class="fig"><svg id="svg" viewBox="-10 -6 780 262" role="img" aria-label="A loop: the router selects its top-K experts; only those experts train on the token; they fit it better; their router logits rise; the router selects them again. Experts outside the loop get no data and no signal.">
<rect x="-10" y="-6" width="780" height="262" rx="16" fill="«PANEL»"/>
<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="«INK»"/></marker></defs>
<path class="flow" d="M250 50H506" marker-end="url(#arrow)"/>
<path class="flow" d="M620 78V168" marker-end="url(#arrow)"/>
<path class="flow" d="M510 200H254" marker-end="url(#arrow)"/>
<path class="flow" d="M140 172V82" marker-end="url(#arrow)"/>
<rect class="exp on" x="30" y="24" width="220" height="52" rx="8"/><text class="lbl-strong" x="140" y="46" text-anchor="middle">The router selects</text><text class="lbl-strong" x="140" y="63" text-anchor="middle">its top-K experts</text>
<rect class="exp" x="510" y="24" width="220" height="52" rx="8"/><text class="lbl-strong" x="620" y="46" text-anchor="middle">Only those experts</text><text class="lbl-strong" x="620" y="63" text-anchor="middle">train on the token</text>
<rect class="exp" x="510" y="174" width="220" height="52" rx="8"/><text class="lbl-strong" x="620" y="196" text-anchor="middle">They fit that token</text><text class="lbl-strong" x="620" y="213" text-anchor="middle">better than before</text>
<rect class="exp" x="30" y="174" width="220" height="52" rx="8"/><text class="lbl-strong" x="140" y="196" text-anchor="middle">Their router logits</text><text class="lbl-strong" x="140" y="213" text-anchor="middle">rise for that token</text>
<rect x="278" y="90" width="204" height="70" rx="8" fill="none" stroke="«GRAY»" stroke-dasharray="4 4"/>
<text class="lbl" x="380" y="112" text-anchor="middle">Every other expert:</text><text class="lbl" x="380" y="129" text-anchor="middle">no data from this token,</text><text class="lbl" x="380" y="146" text-anchor="middle">no signal to the router</text>
<text class="lbl" x="378" y="40" text-anchor="middle">top-K</text><text class="lbl" x="632" y="127">gradient</text><text class="lbl" x="382" y="218" text-anchor="middle">higher reward than the mixture</text><text class="lbl" x="128" y="127" text-anchor="end">Δh &gt; 0</text>
</svg></div>
''', '')

# ---------------------------------------------------------------- Figure 7: recorded training runs of the toy
fig('moe-toy', 'One training run of the tiny MoE, three ways', r'''
.panels{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}
@media (max-width:620px){.panels{grid-template-columns:1fr}}
.panel{background:«CARD»;border-radius:12px;padding:10px 12px 6px;display:flex;flex-direction:column}
.panel h4{font-size:15px;font-weight:600;line-height:1.25;margin:0 0 2px;color:«INK»}
.panel .sub{font-size:13px;line-height:1.4;color:«SOFT»;margin-bottom:4px;font-variant-numeric:tabular-nums}
.panel svg{display:block;width:100%;height:auto;margin-top:auto}
.tb{fill:«BL»;transition:y .08s linear,height .08s linear}
.panel .lbl{font-size:11px}.panel .lbl-strong{font-size:12px;fill:«OR»}
''', r'''
<div class="ctl">
<button id="play" type="button">Replay</button>
<label for="t"><span>Step</span><input type="range" id="t" min="0" max="60" step="1" value="60"><output id="tout" for="t">3000</output></label>
</div>
<div class="box"><div class="panels" id="panels"></div></div>
''', r'''
(function(){
  var root=$('#panels');
  var TR=«TRACES»;
  var defs=[['none','No balancing'],['aux','Auxiliary balance loss'],['center','Centered router input, no balancing']];
  var N=TR.none.load.length, ymax=0.3;
  defs.forEach(function(d){TR[d[0]].load.forEach(function(row){row.forEach(function(v){if(v>ymax)ymax=v})})});
  ymax=Math.ceil(ymax*20)/20;
  var W=250,Hh=150,L=36,B=22,T=10, plotH=Hh-B-T, bw=(W-L-6)/8;
  function sy(v){return T+plotH*(1-v/ymax)}
  var panels=defs.map(function(d){
    var p=H('div',{class:'panel'},root); H('h4',{},p,d[1]); var sub=H('div',{class:'sub'},p,'');
    var svg=S('svg',{viewBox:'0 0 '+W+' '+Hh,role:'img','aria-label':d[1]+': share of token slots per expert over training'},p);
    [0,0.125,0.25,0.5].forEach(function(v){if(v<=ymax+1e-9){
      var a={x1:L,y1:sy(v),x2:W-4,y2:sy(v),stroke:(v===0.125?'«SOFT»':'«RULE»'),'stroke-width':1}; if(v===0.125) a['stroke-dasharray']='4 3';
      S('line',a,svg);
      S('text',{x:L-4,y:sy(v)+4,'text-anchor':'end',class:'lbl'},svg,v===0.125?'1/8':String(v))}});
    var bars=[], marks=[];
    for(var i=0;i<8;i++){
      bars.push(S('rect',{x:L+i*bw+2,y:sy(0),width:bw-4,height:0,rx:2,class:'tb'},svg));
      S('text',{x:L+i*bw+bw/2,y:Hh-8,'text-anchor':'middle',class:'lbl'},svg,String(i+1));
      marks.push(S('text',{x:L+i*bw+bw/2,y:sy(0)-4,'text-anchor':'middle',class:'lbl-strong'},svg,''));
    }
    return {key:d[0],sub:sub,bars:bars,marks:marks};
  });
  var slider=$('#t'), outp=$('#tout'), timer=null;
  function draw(f){
    slider.value=f; outp.textContent=String(TR.none.step[f]);
    panels.forEach(function(p){
      var row=TR[p.key].load[f], dead=0;
      row.forEach(function(v,i){var y=sy(v); p.bars[i].setAttribute('y',y); p.bars[i].setAttribute('height',Math.max(0,sy(0)-y));
        var isDead=f>0&&v<0.01; if(isDead)dead++; p.marks[i].textContent=isDead?'×':''});
      p.sub.textContent='loss '+TR[p.key].loss[f].toFixed(2)+' · experts without tokens: '+dead;
    });
  }
  function play(){
    if(timer){clearInterval(timer);timer=null}
    if(RM){draw(N-1);return}
    var f=0; draw(0);
    timer=setInterval(function(){f++;draw(f);if(f>=N-1){clearInterval(timer);timer=null}},90);
  }
  slider.max=N-1;
  slider.addEventListener('input',function(e){if(timer){clearInterval(timer);timer=null}draw(+e.target.value)});
  $('#play').addEventListener('click',play);
  draw(N-1); onView(root,play);
})();
''')

# ---------------------------------------------------------------- Figure 8: the fixed-budget bound against K
fig('moe-bound', 'The fixed-budget convergence bound against K', '', r'''
<div class="ctl">
<button id="play" type="button">Sweep ρ</button>
<label for="rho"><span>Noise ratio <i>ρ</i></span><input type="range" id="rho" min="0" max="15" step="0.25" value="7"><output id="rhoout" for="rho">7</output></label>
<label for="c"><span>Fixed cost <i>c</i></span><input type="range" id="c" min="0" max="16" step="1" value="0"><output id="cout" for="c">0</output></label>
</div>
<div class="fig"><svg id="svg" viewBox="-10 -10 780 330" role="img" aria-label="Line chart of the convergence bound, divided by its own minimum, against K from 1 to 16 with 16 experts stored. With no noise term the bound keeps falling until K equals M. With a large noise term the minimum moves to small K."></svg></div>
<div class="key"><span><svg width="26" height="8" aria-hidden="true"><line x1="0" y1="4" x2="26" y2="4" stroke="«BL»" stroke-width="2.5"/></svg>bound with the chosen noise ratio</span><span><svg width="26" height="8" aria-hidden="true"><line x1="0" y1="4" x2="26" y2="4" stroke="«GRAY»" stroke-width="2" stroke-dasharray="5 4"/></svg><span>same bound with <i>ρ</i> = 0</span></span></div>
<div id="read" aria-live="off"></div>
''', r'''
(function(){
  var svg=$('#svg');
  var M=16, rho=7, c=0, X0=70, X1=716, Y0=34, Y1=262, YMAX=2.6, timer=null;
  function b(K,rh){return (1+M/K+rh)*Math.sqrt(c+K)}
  function sx(K){return X0+(K-1)/(M-1)*(X1-X0)}
  function sy(v){return Y1-(Math.min(v,YMAX)-1)/(YMAX-1)*(Y1-Y0)}
  function curve(rh){var pts=[],mn=Infinity,arg=1;for(var K=1;K<=M+1e-9;K+=0.05){var v=b(K,rh);pts.push([K,v]);if(v<mn){mn=v;arg=K}}return {pts:pts.map(function(p){return [p[0],p[1]/mn]}),arg:arg,min:mn}}
  function path(pts){var d='',pen=false;pts.forEach(function(p){if(p[1]>YMAX){pen=false;return}d+=(pen?'L':'M')+sx(p[0]).toFixed(1)+' '+sy(p[1]).toFixed(1);pen=true});return d}
  var A,B;
  function draw(){
    clear(svg); panel(svg);
    A=curve(rho); B=curve(0);
    [1,1.5,2,2.5].forEach(function(v){
      S('line',{x1:X0,y1:sy(v),x2:X1,y2:sy(v),stroke:'«RULE»','stroke-width':1},svg);
      S('text',{x:X0-8,y:sy(v)+4,'text-anchor':'end',class:'lbl'},svg,v.toFixed(1));
    });
    [1,2,4,8,12,16].forEach(function(k){
      S('line',{x1:sx(k),y1:Y1,x2:sx(k),y2:Y1+5,stroke:'«GRAY»','stroke-width':1},svg);
      S('text',{x:sx(k),y:Y1+19,'text-anchor':'middle',class:'lbl'},svg,String(k));
    });
    S('line',{x1:X0,y1:Y1,x2:X1,y2:Y1,stroke:'«GRAY»','stroke-width':1},svg);
    S('text',{x:(X0+X1)/2,y:Y1+42,'text-anchor':'middle',class:'lbl'},svg,'K, experts run per token (M = 16 stored)');
    S('text',{x:18,y:Y0-18,class:'lbl'},svg,'bound ÷ its own minimum');
    S('path',{d:path(B.pts),fill:'none',stroke:'«GRAY»','stroke-width':2,'stroke-dasharray':'5 4'},svg);
    S('path',{d:path(A.pts),fill:'none',stroke:'«BL»','stroke-width':2.5,'stroke-linejoin':'round'},svg);
    S('line',{x1:sx(A.arg),y1:sy(1),x2:sx(A.arg),y2:Y1,stroke:'«BL»','stroke-width':1,'stroke-dasharray':'3 3'},svg);
    S('circle',{cx:sx(B.arg),cy:sy(1),r:4.5,fill:'«GRAY»',stroke:'«PANEL»','stroke-width':2},svg);
    S('circle',{cx:sx(A.arg),cy:sy(1),r:5.5,fill:'«BL»',stroke:'«PANEL»','stroke-width':2},svg);
    var ax=sx(A.arg), anchor=ax>X1-120?'end':'start';
    S('text',{x:ax+(anchor==='start'?8:-8),y:sy(1)-22,'text-anchor':anchor,class:'lbl-strong'},svg,'best K = '+A.arg.toFixed(1));
    var hov=S('g',{style:'display:none'},svg);
    var hl=S('line',{y1:Y0,y2:Y1,stroke:'«SOFT»','stroke-width':1},hov);
    var hb=S('rect',{width:206,height:58,rx:6,fill:'«CARD»',stroke:'«GRAY»','stroke-width':1},hov);
    var t1=S('text',{class:'lbl-strong'},hov,''), t2=S('text',{class:'lbl on'},hov,''), t3=S('text',{class:'lbl'},hov,'');
    var hit=S('rect',{x:X0,y:Y0,width:X1-X0,height:Y1-Y0,fill:'transparent',style:'cursor:crosshair'},svg);
    function move(ev){
      var box=svg.getBoundingClientRect(), vb=svg.viewBox.baseVal, px=vb.x+(ev.clientX-box.left)*vb.width/box.width;
      var K=Math.max(1,Math.min(M,Math.round(1+(px-X0)/(X1-X0)*(M-1))));
      var va=b(K,rho)/A.min, vb2=b(K,0)/B.min, x=sx(K), bxp=x>X1-226?x-216:x+10;
      hov.style.display=''; hl.setAttribute('x1',x); hl.setAttribute('x2',x);
      hb.setAttribute('x',bxp); hb.setAttribute('y',Y0+6);
      t1.setAttribute('x',bxp+10); t1.setAttribute('y',Y0+25); t1.textContent='K = '+K;
      t2.setAttribute('x',bxp+10); t2.setAttribute('y',Y0+41); t2.textContent='ρ = '+rho+': '+va.toFixed(2)+'× its minimum';
      t3.setAttribute('x',bxp+10); t3.setAttribute('y',Y0+56); t3.textContent='ρ = 0: '+vb2.toFixed(2)+'× its minimum';
    }
    hit.addEventListener('pointermove',move); hit.addEventListener('pointerdown',move);
    hit.addEventListener('pointerleave',function(){hov.style.display='none'});
    var txt='Best K = '+A.arg.toFixed(1)+' of '+M+'. With ρ = 0 it is '+B.arg.toFixed(1)+'.';
    if(c===0) txt+=' Closed form M / (1 + ρ) = '+(M/(1+rho)).toFixed(1)+(rho===7?', the value the paper observes.':'.');
    else txt+=' A fixed cost of '+c+' expert'+(c===1?'':'s')+' per layer makes extra experts relatively cheaper and moves the best K up.';
    $('#read').textContent=txt;
  }
  function setRho(v){rho=v;$('#rho').value=v;$('#rhoout').textContent=v;draw()}
  function sweep(to){
    if(timer){clearInterval(timer);timer=null}
    if(RM){setRho(to);return}
    var v=0; setRho(0);
    timer=setInterval(function(){v=Math.min(to,v+0.25);setRho(v);if(v>=to){clearInterval(timer);timer=null}},70);
  }
  $('#rho').addEventListener('input',function(e){if(timer){clearInterval(timer);timer=null}rho=+e.target.value;$('#rhoout').textContent=rho;draw()});
  $('#c').addEventListener('input',function(e){c=+e.target.value;$('#cout').textContent=c;draw()});
  $('#play').addEventListener('click',function(){sweep(15)});
  draw(); onView(svg,function(){sweep(7)});
})();
''')


def fill(text, name):
    for key, value in C.items():
        text = text.replace('«%s»' % key, value)
    text = text.replace('«NAME»', json.dumps(name))
    text = text.replace('«SERIESCSS»', ''.join('.e%d{--c:%s}' % (i + 1, col) for i, col in enumerate(SERIES)))
    if '«TRACES»' in text:
        with open(os.path.join(HERE, 'moe_traces.json'), encoding='utf-8') as fh:
            text = text.replace('«TRACES»', json.dumps(json.load(fh), separators=(',', ':')))
    assert '«' not in text, text[text.index('«'):][:40]
    return text


def page(f):
    html = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>%s</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Source+Serif+4:ital,wght@1,400;1,600&display=swap">
<style>%s%s</style></head><body><main id="wrap">%s</main>
<script>%s%s</script>
</body></html>
''' % (f['title'], CSS, f['css'], f['body'], JS, f['js'])
    return fill(html, f['name'])


def main():
    with open(os.path.join(ROOT, 'tools', 'deckfigs', 'figures.json'), encoding='utf-8') as fh:
        forbidden = json.load(fh)['forbidden']
    os.makedirs(OUT, exist_ok=True)
    for f in FIGS:
        html = page(f)
        for bad in forbidden:
            assert bad not in html, 'forbidden string "%s" in %s' % (bad, f['name'])
        path = os.path.join(OUT, f['name'] + '.html')
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write(html)
        print(os.path.relpath(path, ROOT))


if __name__ == '__main__':
    main()
