from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


DEFAULT_RESULT = Path("results/latest_malecns_adaptive_training.json")
DEFAULT_HTML = Path("results/latest_malecns_adaptive_training.html")


def _json_for_script(data: object) -> str:
    # Avoid accidental </script> termination inside embedded data.
    return json.dumps(data, ensure_ascii=False).replace("</", "<\\/")


def build_html(report: dict[str, object]) -> str:
    data = _json_for_script(report)
    title = html.escape(str(report.get("experiment", "MaleCNS training")))
    return f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
:root {{ color-scheme: dark; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }}
body {{ margin:0; background:#0d1117; color:#e6edf3; }}
main {{ max-width:1200px; margin:auto; padding:28px 20px 60px; }}
h1 {{ margin:0 0 4px; font-size:28px; }}
.sub {{ color:#8b949e; margin-bottom:22px; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; }}
.card,.panel {{ background:#161b22; border:1px solid #30363d; border-radius:12px; padding:16px; }}
.card .k {{ color:#8b949e; font-size:12px; text-transform:uppercase; letter-spacing:.08em; }}
.card .v {{ font-size:26px; font-weight:700; margin-top:5px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(360px,1fr)); gap:14px; margin-top:14px; }}
.panel h2 {{ font-size:15px; margin:0 0 12px; color:#c9d1d9; }}
canvas {{ width:100%; height:240px; display:block; background:#0d1117; border-radius:8px; }}
table {{ width:100%; border-collapse:collapse; font-size:13px; }}
th,td {{ border-bottom:1px solid #30363d; padding:7px 6px; text-align:right; }}
th:first-child,td:first-child {{ text-align:left; }}
.badge {{ display:inline-block; padding:4px 8px; border-radius:999px; background:#21262d; margin-right:6px; font-size:12px; }}
.note {{ color:#8b949e; font-size:12px; margin-top:9px; line-height:1.45; }}
</style>
</head>
<body><main>
<h1>MaleCNS learning report</h1>
<div class="sub" id="subtitle"></div>
<div class="cards" id="cards"></div>
<div class="grid">
  <section class="panel"><h2>Rolling training accuracy</h2><canvas id="acc"></canvas></section>
  <section class="panel"><h2>Reward and confidence</h2><canvas id="reward"></canvas></section>
  <section class="panel"><h2>Output spikes and updated edges</h2><canvas id="activity"></canvas></section>
  <section class="panel"><h2>Automatic challenge calibration</h2><canvas id="calibration"></canvas><div class="note">The selected condition aims to start below perfect accuracy while keeping the output population active.</div></section>
  <section class="panel"><h2>Before vs after learning</h2><canvas id="beforeAfter"></canvas></section>
  <section class="panel"><h2>Plastic synapse state</h2><div id="plasticity"></div></section>
</div>
<section class="panel" style="margin-top:14px"><h2>Recent training trials</h2><div id="recent"></div></section>
<script>
const R = {data};
const ev = R.evaluation || {{}};
const bt = R.brain_training || {{}};
const rows = bt.rows || [];
const challenge = R.challenge || {{}};
const plast = R.plasticity || {{}};
const pct = x => (100*Number(x||0)).toFixed(1)+'%';
document.getElementById('subtitle').textContent = `${{R.experiment || ''}} · ${{R.connectome?.neuron_count?.toLocaleString?.() || ''}} neurons · ${{R.connectome?.edge_count?.toLocaleString?.() || ''}} edges`;
const cards = [
 ['Before', pct(ev.before_accuracy)], ['After', pct(ev.after_accuracy)], ['Δ accuracy', ((Number(ev.delta_accuracy||0)*100).toFixed(1)+' pp')],
 ['Training', pct(bt.training_accuracy)], ['Changed edges', Number(plast.changed_edges||0).toLocaleString()], ['Plastic edges', Number(plast.plastic_edges||0).toLocaleString()]
];
document.getElementById('cards').innerHTML = cards.map(([k,v])=>`<div class="card"><div class="k">${{k}}</div><div class="v">${{v}}</div></div>`).join('');

function setup(id) {{ const c=document.getElementById(id); const d=devicePixelRatio||1; const r=c.getBoundingClientRect(); c.width=Math.max(300,r.width*d); c.height=240*d; const x=c.getContext('2d'); x.scale(d,d); return [x,r.width,240]; }}
function lineChart(id, series, opts={{}}) {{
 const [x,w,h]=setup(id), pad=30; x.strokeStyle='#30363d'; x.fillStyle='#8b949e'; x.font='11px system-ui';
 x.beginPath(); x.moveTo(pad,8); x.lineTo(pad,h-pad); x.lineTo(w-8,h-pad); x.stroke();
 const all=series.flatMap(s=>s.values).filter(Number.isFinite); if(!all.length)return;
 let lo=opts.min ?? Math.min(...all), hi=opts.max ?? Math.max(...all); if(hi===lo)hi=lo+1;
 for(const s of series) {{ x.beginPath(); x.strokeStyle=s.color; x.lineWidth=1.8; s.values.forEach((v,i)=>{{ const px=pad+(w-pad-10)*(i/Math.max(1,s.values.length-1)); const py=8+(h-pad-12)*(1-(v-lo)/(hi-lo)); if(i)x.lineTo(px,py); else x.moveTo(px,py); }}); x.stroke(); }}
 x.fillStyle='#8b949e'; x.fillText(hi.toFixed(2),3,14); x.fillText(lo.toFixed(2),3,h-pad);
}}
function rolling(vals,n=16) {{ return vals.map((_,i)=>{{const a=Math.max(0,i-n+1), s=vals.slice(a,i+1); return s.reduce((p,q)=>p+q,0)/s.length;}}); }}
const correct=rows.map(r=>r.correct?1:0); lineChart('acc',[{{values:rolling(correct),color:'#3fb950'}}],{{min:0,max:1}});
lineChart('reward',[{{values:rows.map(r=>Number(r.reward||0)),color:'#f85149'}},{{values:rows.map(r=>Number(r.confidence||0)),color:'#58a6ff'}}],{{min:-1,max:1}});
lineChart('activity',[{{values:rows.map(r=>Number(r.output_spikes||0)),color:'#d2a8ff'}},{{values:rows.map(r=>Number(r.edge_updates||0)/1000),color:'#ffa657'}}]);
const cc=challenge.calibration_candidates||[]; lineChart('calibration',[{{values:cc.map(r=>Number(r.accuracy||0)),color:'#58a6ff'}},{{values:cc.map(r=>Math.min(1,Number(r.mean_output_spikes||0)/50)),color:'#8b949e'}}],{{min:0,max:1}});
lineChart('beforeAfter',[{{values:[Number(ev.before_accuracy||0),Number(ev.after_accuracy||0)],color:'#3fb950'}}],{{min:0,max:1}});
const q=plast.multiplier_quantiles||{{}}; document.getElementById('plasticity').innerHTML=`
 <span class="badge">mean ×${{Number(plast.mean_multiplier||1).toFixed(4)}}</span>
 <span class="badge">stability ${{Number(plast.mean_stability||0).toFixed(5)}}</span>
 <table><tr><th>quantile</th><th>multiplier</th></tr>${{Object.entries(q).map(([k,v])=>`<tr><td>${{k}}</td><td>${{Number(v).toFixed(4)}}</td></tr>`).join('')}}</table>`;
const recent=rows.slice(-16); document.getElementById('recent').innerHTML=`<table><tr><th>trial</th><th>target</th><th>prediction</th><th>ok</th><th>confidence</th><th>spikes</th><th>edge updates</th></tr>${{recent.map(r=>`<tr><td>${{r.trial}}</td><td>${{r.target}}</td><td>${{r.prediction}}</td><td>${{r.correct?'✓':'×'}}</td><td>${{Number(r.confidence).toFixed(3)}}</td><td>${{r.output_spikes}}</td><td>${{Number(r.edge_updates).toLocaleString()}}</td></tr>`).join('')}}</table>`;
</script></main></body></html>'''


def render(result_path: Path = DEFAULT_RESULT, html_path: Path = DEFAULT_HTML) -> Path:
    report = json.loads(Path(result_path).read_text(encoding="utf-8"))
    html_path = Path(html_path)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(build_html(report), encoding="utf-8")
    return html_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a standalone MaleCNS learning dashboard.")
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--html", type=Path, default=DEFAULT_HTML)
    args = parser.parse_args()
    out = render(args.result, args.html)
    print(f"saved visualization: {out}")


if __name__ == "__main__":
    main()
