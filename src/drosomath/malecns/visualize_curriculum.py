from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


DEFAULT_RESULT = Path("results/latest_malecns_v1_curriculum.json")
DEFAULT_HTML = Path("results/latest_malecns_v1_curriculum.html")


def build_curriculum_html(report: dict[str, object]) -> str:
    data = json.dumps(report, ensure_ascii=False).replace("</", "<\\/")
    title = html.escape(str(report.get("experiment", "MaleCNS curriculum")))
    return f'''<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>
:root{{color-scheme:dark;font-family:Inter,system-ui,sans-serif}}body{{margin:0;background:#0d1117;color:#e6edf3}}main{{max-width:1250px;margin:auto;padding:28px 18px 60px}}h1{{margin:0}}.sub{{color:#8b949e;margin:5px 0 20px}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px}}.card,.panel{{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:15px}}.k{{font-size:11px;color:#8b949e;text-transform:uppercase;letter-spacing:.08em}}.v{{font-size:24px;font-weight:700;margin-top:5px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:14px;margin-top:14px}}canvas{{width:100%;height:280px;background:#0d1117;border-radius:8px}}table{{width:100%;border-collapse:collapse;font-size:13px}}th,td{{border-bottom:1px solid #30363d;padding:7px;text-align:right}}th:first-child,td:first-child{{text-align:left}}.good{{color:#7ee787}}.bad{{color:#ff7b72}}.muted{{color:#8b949e;font-size:12px}}@media(max-width:700px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><main><h1>DrosoMath · MaleCNS curriculum v1</h1><div class="sub" id="sub"></div><div class="cards" id="cards"></div>
<div class="grid"><section class="panel"><h3>Stage accuracy: before → after</h3><canvas id="stages"></canvas></section><section class="panel"><h3>Retention after each stage</h3><div id="retention"></div></section><section class="panel"><h3>Stage details</h3><div id="details"></div></section><section class="panel"><h3>Final synaptic state</h3><div id="plastic"></div></section></div>
<script>
const R={data};const S=R.stages||[];const F=R.final_plasticity||{{}};const pct=x=>(100*Number(x||0)).toFixed(1)+'%';
document.getElementById('sub').textContent=`${{R.connectome?.neuron_count?.toLocaleString?.()||''}} neurons · ${{R.connectome?.edge_count?.toLocaleString?.()||''}} anatomical edges · checkpoint ${{R.checkpoint||''}}`;
const deltas=S.map(s=>Number(s.delta_accuracy||0));const cards=[['Stages',S.length],['Last Δ',deltas.length?((deltas.at(-1)*100).toFixed(1)+' pp'):'—'],['Changed edges',Number(F.changed_edges||0).toLocaleString()],['Plastic edges',Number(F.plastic_edge_count||0).toLocaleString()],['Plastic fraction',pct(F.plastic_fraction)],['Mean multiplier',Number(F.mean_multiplier||1).toFixed(4)]];document.getElementById('cards').innerHTML=cards.map(([k,v])=>`<div class="card"><div class="k">${{k}}</div><div class="v">${{v}}</div></div>`).join('');
function setup(id){{const c=document.getElementById(id),d=devicePixelRatio||1,r=c.getBoundingClientRect();c.width=Math.max(300,r.width*d);c.height=280*d;const x=c.getContext('2d');x.scale(d,d);return[x,r.width,280]}}
function bars(){{const[x,w,h]=setup('stages'),pad=42,n=Math.max(1,S.length),bw=(w-pad-12)/n; x.strokeStyle='#30363d';x.beginPath();x.moveTo(pad,10);x.lineTo(pad,h-35);x.lineTo(w-8,h-35);x.stroke();S.forEach((s,i)=>{{const before=Number(s.before_accuracy||0),after=Number(s.after_accuracy||0),base=pad+i*bw+8,avail=h-55;x.fillStyle='#6e7681';x.fillRect(base,20+avail*(1-before),Math.max(6,bw*.28),avail*before);x.fillStyle='#3fb950';x.fillRect(base+bw*.31,20+avail*(1-after),Math.max(6,bw*.28),avail*after);x.fillStyle='#8b949e';x.font='10px system-ui';x.fillText(s.stage.slice(0,12),base,h-18)}});x.fillStyle='#8b949e';x.fillText('gray before · green after',pad,12)}} bars();
const H=R.retention_history||[];let names=[...new Set(H.flatMap(h=>Object.keys(h.tasks||{{}})))];let rt='<table><tr><th>after stage</th>'+names.map(n=>`<th>${{n}}</th>`).join('')+'</tr>';for(const h of H)rt+=`<tr><td>${{h.after_stage}}</td>`+names.map(n=>h.tasks?.[n]?`<td>${{pct(h.tasks[n].accuracy)}}<span class="muted"> / silent ${{pct(h.tasks[n].silent_fraction)}}</span></td>`:'<td>—</td>').join('')+'</tr>';rt+='</table>';document.getElementById('retention').innerHTML=rt;
document.getElementById('details').innerHTML='<table><tr><th>stage</th><th>before</th><th>after</th><th>Δ</th><th>train</th><th>silent</th></tr>'+S.map(s=>`<tr><td>${{s.stage}}</td><td>${{pct(s.before_accuracy)}}</td><td>${{pct(s.after_accuracy)}}</td><td class="${{Number(s.delta_accuracy)>=0?'good':'bad'}}">${{(100*Number(s.delta_accuracy||0)).toFixed(1)}} pp</td><td>${{pct(s.training_accuracy)}}</td><td>${{pct(s.training_silent_fraction)}}</td></tr>`).join('')+'</table>';
document.getElementById('plastic').innerHTML=`<table><tr><td>Changed edges</td><td>${{Number(F.changed_edges||0).toLocaleString()}}</td></tr><tr><td>Plastic edges</td><td>${{Number(F.plastic_edge_count||0).toLocaleString()}}</td></tr><tr><td>Mean multiplier</td><td>${{Number(F.mean_multiplier||1).toFixed(6)}}</td></tr><tr><td>Mean usage EMA</td><td>${{Number(F.mean_usage_ema||0).toFixed(6)}}</td></tr><tr><td>Mean stability</td><td>${{Number(F.mean_stability||0).toFixed(6)}}</td></tr></table>`;
</script></main></body></html>'''


def render(result: Path = DEFAULT_RESULT, output: Path = DEFAULT_HTML) -> Path:
    report = json.loads(Path(result).read_text(encoding="utf-8"))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_curriculum_html(report), encoding="utf-8")
    return output


def main() -> None:
    p = argparse.ArgumentParser(description="Render MaleCNS curriculum v1 report")
    p.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    p.add_argument("--html", type=Path, default=DEFAULT_HTML)
    a = p.parse_args(); print(f"saved visualization: {render(a.result,a.html)}")


if __name__ == "__main__": main()
