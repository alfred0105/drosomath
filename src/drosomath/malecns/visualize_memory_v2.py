from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from .curriculum_v2_memory import DEFAULT_RESULT


DEFAULT_HTML = Path("results/latest_malecns_v2_memory.html")


def build_memory_v2_html(report: dict[str, object]) -> str:
    data = json.dumps(report, ensure_ascii=False).replace("</", "<\\/")
    title = html.escape(str(report.get("experiment", "MaleCNS memory v2")))
    return f'''<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>
:root{{color-scheme:dark;font-family:Inter,system-ui,sans-serif}}body{{margin:0;background:#0d1117;color:#e6edf3}}main{{max-width:1280px;margin:auto;padding:28px 18px 60px}}h1{{margin:0}}.sub{{color:#8b949e;margin:5px 0 20px}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px}}.card,.panel{{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:15px}}.k{{font-size:11px;color:#8b949e;text-transform:uppercase;letter-spacing:.08em}}.v{{font-size:23px;font-weight:700;margin-top:5px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(440px,1fr));gap:14px;margin-top:14px}}table{{width:100%;border-collapse:collapse;font-size:13px}}th,td{{border-bottom:1px solid #30363d;padding:7px;text-align:right}}th:first-child,td:first-child{{text-align:left}}.good{{color:#7ee787}}.bad{{color:#ff7b72}}.muted{{color:#8b949e;font-size:12px}}.pass{{color:#7ee787;font-weight:700}}.fail{{color:#ff7b72;font-weight:700}}canvas{{width:100%;height:280px;background:#0d1117;border-radius:8px}}@media(max-width:700px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><main><h1>DrosoMath · Phase 1 Memory v2</h1><div class="sub" id="sub"></div><div class="cards" id="cards"></div><div class="grid"><section class="panel"><h3>Stage learning</h3><div id="stages"></div></section><section class="panel"><h3>Final retention · v1 vs v2</h3><div id="compare"></div></section><section class="panel"><h3>Phase 1 gate</h3><div id="gate"></div></section><section class="panel"><h3>Replay / consolidation</h3><div id="memory"></div></section><section class="panel"><h3>Retention history</h3><div id="retention"></div></section><section class="panel"><h3>Final synaptic state</h3><div id="plastic"></div></section></div>
<script>
const R={data},S=R.stages||[],G=R.phase1_gate||{{}},C=R.v1_comparison?.final_retention||{{}},F=R.final_plasticity||{{}};const pct=x=>(100*Number(x||0)).toFixed(1)+'%';const pp=x=>(100*Number(x||0)).toFixed(1)+' pp';
document.getElementById('sub').textContent=`${{R.connectome?.neuron_count?.toLocaleString?.()||''}} neurons · ${{R.connectome?.edge_count?.toLocaleString?.()||''}} edges · protected plasticity + replay + consolidation`;
const cards=[['Phase 1',G.passed?'PASS':'FAIL'],['Changed edges',Number(F.changed_edges||0).toLocaleString()],['Plastic edges',Number(F.plastic_edge_count||0).toLocaleString()],['Mean stability',Number(F.mean_stability||0).toFixed(5)],['Replay interval',R.config?.replay_interval??'—'],['Consolidation top',pct(R.config?.consolidation_top_fraction)]];document.getElementById('cards').innerHTML=cards.map(([k,v],i)=>`<div class="card"><div class="k">${{k}}</div><div class="v ${{i===0?(G.passed?'pass':'fail'):''}}">${{v}}</div></div>`).join('');
document.getElementById('stages').innerHTML='<table><tr><th>stage</th><th>before</th><th>after</th><th>Δ</th><th>train</th><th>replay</th></tr>'+S.map(s=>`<tr><td>${{s.stage}}</td><td>${{pct(s.before_accuracy)}}</td><td>${{pct(s.after_accuracy)}}</td><td class="${{Number(s.delta_accuracy)>=0?'good':'bad'}}">${{pp(s.delta_accuracy)}}</td><td>${{pct(s.training_accuracy)}}</td><td>${{s.replay?.accuracy==null?'—':pct(s.replay.accuracy)}} (${{s.replay?.count||0}})</td></tr>`).join('')+'</table>';
let cr='<table><tr><th>task</th><th>v1</th><th>v2</th><th>Δ</th></tr>';for(const [name,row] of Object.entries(C))cr+=`<tr><td>${{name}}</td><td>${{pct(row.v1_accuracy)}}</td><td>${{pct(row.v2_accuracy)}}</td><td class="${{Number(row.delta_accuracy)>=0?'good':'bad'}}">${{pp(row.delta_accuracy)}}</td></tr>`;cr+='</table>';document.getElementById('compare').innerHTML=Object.keys(C).length?cr:'<div class="muted">v1 baseline unavailable</div>';
let gt=`<div class="${{G.passed?'pass':'fail'}}">${{G.passed?'PASS — move to Phase 2 credit assignment':'FAIL — stay on Phase 1 and tune memory protection'}}</div><table><tr><th>task</th><th>immediate</th><th>final</th><th>ratio</th><th>required</th><th>status</th></tr>`;for(const [name,row] of Object.entries(G.tasks||{{}}))gt+=`<tr><td>${{name}}</td><td>${{pct(row.immediate_accuracy)}}</td><td>${{pct(row.final_retention_accuracy)}}</td><td>${{pct(row.retention_ratio)}}</td><td>${{pct(row.required_accuracy)}}</td><td class="${{row.passed?'good':'bad'}}">${{row.passed?'PASS':'FAIL'}}</td></tr>`;gt+='</table>';document.getElementById('gate').innerHTML=gt;
const cons=R.consolidation_history||[],rep=R.replay_history||[];document.getElementById('memory').innerHTML='<table><tr><th>stage</th><th>consolidated</th><th>stab. gain</th><th>replays</th><th>replay acc.</th></tr>'+cons.map((x,i)=>`<tr><td>${{x.after_stage}}</td><td>${{Number(x.consolidated_edges||0).toLocaleString()}}</td><td>${{Number(x.mean_stability_gain||0).toFixed(4)}}</td><td>${{rep[i]?.count||0}}</td><td>${{rep[i]?.accuracy==null?'—':pct(rep[i].accuracy)}}</td></tr>`).join('')+'</table>';
const H=R.retention_history||[];let names=[...new Set(H.flatMap(h=>Object.keys(h.tasks||{{}})))];let rt='<table><tr><th>after stage</th>'+names.map(n=>`<th>${{n}}</th>`).join('')+'</tr>';for(const h of H)rt+=`<tr><td>${{h.after_stage}}</td>`+names.map(n=>h.tasks?.[n]?`<td>${{pct(h.tasks[n].accuracy)}}<span class="muted"> / silent ${{pct(h.tasks[n].silent_fraction)}}</span></td>`:'<td>—</td>').join('')+'</tr>';rt+='</table>';document.getElementById('retention').innerHTML=rt;
document.getElementById('plastic').innerHTML=`<table><tr><td>Changed edges</td><td>${{Number(F.changed_edges||0).toLocaleString()}}</td></tr><tr><td>Plastic edges</td><td>${{Number(F.plastic_edge_count||0).toLocaleString()}}</td></tr><tr><td>Plastic fraction</td><td>${{pct(F.plastic_fraction)}}</td></tr><tr><td>Mean multiplier</td><td>${{Number(F.mean_multiplier||1).toFixed(6)}}</td></tr><tr><td>Mean usage EMA</td><td>${{Number(F.mean_usage_ema||0).toFixed(6)}}</td></tr><tr><td>Mean stability</td><td>${{Number(F.mean_stability||0).toFixed(6)}}</td></tr></table>`;
</script></main></body></html>'''


def render(result: Path = DEFAULT_RESULT, output: Path = DEFAULT_HTML) -> Path:
    report = json.loads(Path(result).read_text(encoding="utf-8"))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_memory_v2_html(report), encoding="utf-8")
    return output


def main() -> None:
    p = argparse.ArgumentParser(description="Render MaleCNS memory v2 report")
    p.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    p.add_argument("--html", type=Path, default=DEFAULT_HTML)
    a = p.parse_args()
    print(f"saved visualization: {render(a.result, a.html)}")


if __name__ == "__main__":
    main()
