# FASE 8 / Sez. 14: export JSON e viewer HTML degli OBDD.

import argparse
import json
from pathlib import Path

import pandas as pd

from .. import config, forest_io
from ..bdd_backend import make_bdd
from ..compose import compose_forest, declare_vars, win_regions
from ..discretize import discretize_forest


def export_json(name: str, k: int, label: int) -> dict:
    # trasforma la regione certificata in un elenco di nodi e archi pronto per il viewer
    rf = forest_io.load_forest(name)
    dforest = discretize_forest(rf, n_trees=k)
    order = dforest.input_bit_names()
    pos = {v: i for i, v in enumerate(order)}
    feat_names = list(pd.read_csv(
        config.DATASETS_DIR / name / f"{name}_train.csv").columns[:-1])

    bdd = make_bdd()
    declare_vars(bdd, dforest)
    wins = win_regions(bdd, dforest, compose_forest(bdd, dforest))
    root = wins[label]

    nodes, edges, ids = [], [], {}

    def cof(f):
        # restituisce i cofattori raddrizzando gli archi negati
        return (~f.low, ~f.high) if f.negated else (f.low, f.high)

    def visit(f):
        # visita il diagramma e assegna un id a ogni nodo
        if f == bdd.true:
            return "T"
        if f == bdd.false:
            return "F"
        if f in ids:
            return ids[f]
        nid = f"n{len(ids)}"
        ids[f] = nid
        var = f.var  # il nome contiene la feature e il bit
        fi = int(var.split("_")[1])
        bit = int(var.split("_")[2])
        nodes.append({
            "id": nid, "var": var, "feature": fi,
            "feature_name": feat_names[fi] if fi < len(feat_names) else f"f{fi}",
            "bit": bit, "level": pos[var],
        })
        lo, hi = cof(f)
        edges.append({"src": nid, "dst": visit(lo), "branch": 0})
        edges.append({"src": nid, "dst": visit(hi), "branch": 1})
        return nid

    root_id = visit(root)
    feats_present = sorted({n["feature"] for n in nodes})
    thresholds = {
        str(f): [round(float(t), 4) for t in dforest.thresholds[f]]
        for f in feats_present
    }
    return {
        "title": f"{name} — Win_{label} (k={k})",
        "subtitle": f"{len(nodes)} nodi | regione certificata della classe {label}",
        "feature_names": feat_names,
        "features_present": feats_present,
        "feature_thresholds": thresholds,  # soglie reali di ogni feature presente
        "nodes": nodes, "edges": edges, "root": root_id,
        "n_levels": len(order),
    }


HTML = """<!DOCTYPE html>
<html lang="it"><head><meta charset="utf-8"><title>__TITLE__</title>
<style>
 body{margin:0;font-family:system-ui,sans-serif;background:#fafafa}
 #bar{padding:10px 16px;background:#222;color:#fff}
 #bar b{font-size:16px} #bar span{opacity:.8;font-size:13px;margin-left:10px}
 #wrap{display:flex;height:calc(100vh - 46px)}
 svg{flex:1;background:#fff;cursor:grab}
 #info{width:240px;padding:14px;background:#f0f0f3;border-left:1px solid #ccc;font-size:14px}
 #info h3{margin:.2em 0} .k{color:#666}
 .node circle{stroke:#333;stroke-width:1.5px;cursor:pointer}
 .node:hover circle{stroke-width:3px}
 .node text{font-size:10px;pointer-events:none}
 .term rect{stroke:#333} .term text{font-weight:bold;font-size:13px}
 .edge{fill:none;stroke-width:1.6px}
 .hi{stroke:#1f77b4} .lo{stroke:#999;stroke-dasharray:5 4}
 #legend div{margin:3px 0;font-size:13px} #legend i{display:inline-block;width:12px;height:12px;border-radius:50%;margin-right:6px;vertical-align:-1px}
</style></head><body>
<div id="bar"><b>__TITLE__</b><span>__SUBTITLE__ &nbsp;·&nbsp; trascina = pan, rotella = zoom, click nodo = info</span></div>
<div id="wrap"><svg id="svg"></svg>
<div id="info"><h3>Archi</h3>
 <p><span style="color:#1f77b4">———</span> bit = 1 (high)<br>
 <span style="color:#999">- - -</span> bit = 0 (low)<br>
 <b>1</b> = certificato · <b>0</b> = no</p>
 <h3>Feature (colore nodo)</h3><div id="legend"></div>
 <h3>Nodo selezionato</h3><div id="sel" class="k">clicca un nodo…</div></div></div>
<script>
const D = __DATA__;
const NS="http://www.w3.org/2000/svg", svg=document.getElementById("svg");
const DX=90, DY=80, R=22;
const PAL=["#378ADD","#1D9E75","#D85A30","#D4537E","#BA7517","#7F77DD","#639922","#5F5E5A"];
const fcolor={}; D.features_present.forEach((f,i)=>fcolor[f]=PAL[i%PAL.length]);
const byLevel={}; D.nodes.forEach(n=>{(byLevel[n.level]=byLevel[n.level]||[]).push(n)});
const X={}, Y={};
Object.keys(byLevel).forEach(L=>byLevel[L].forEach((n,j)=>{
  X[n.id]=(j-(byLevel[L].length-1)/2)*DX; Y[n.id]=n.level*DY;}));
const termY=(D.n_levels+0.6)*DY;
const TX={F:-DX,T:DX}; const xy=id=> id=="T"||id=="F"?[TX[id],termY]:[X[id],Y[id]];
const g=document.createElementNS(NS,"g"); svg.appendChild(g);
function line(a,b,cls){const [x0,y0]=xy(a),[x1,y1]=xy(b);
  const p=document.createElementNS(NS,"path");
  p.setAttribute("d",`M${x0},${y0+R} C${x0},${(y0+y1)/2} ${x1},${(y0+y1)/2} ${x1},${y1-R}`);
  p.setAttribute("class","edge "+cls); g.appendChild(p);}
D.edges.forEach(e=>line(e.src,e.dst,e.branch?"hi":"lo"));
function thr(f){const t=D.feature_thresholds[String(f)]||[];
  return t.length? "soglie reali: "+t.join(", ") : "nessuna soglia";}
D.nodes.forEach(n=>{const [x,y]=xy(n.id);
  const grp=document.createElementNS(NS,"g"); grp.setAttribute("class","node");
  grp.setAttribute("transform",`translate(${x},${y})`);
  const c=document.createElementNS(NS,"circle"); c.setAttribute("r",R);
  c.setAttribute("fill",fcolor[n.feature]||"#ccc"); c.setAttribute("fill-opacity","0.35"); grp.appendChild(c);
  const t=document.createElementNS(NS,"text"); t.setAttribute("text-anchor","middle");
  t.setAttribute("dy","4"); t.textContent=n.feature_name.slice(0,6)+" b"+n.bit; grp.appendChild(t);
  grp.onclick=()=>{document.getElementById("sel").innerHTML=
    `<b>${n.feature_name}</b><br><span class=k>feature #${n.feature}, bit ${n.bit}</span>`+
    `<br>${thr(n.feature)}`+
    `<br><span class=k>testa il bit ${n.bit} dell'indice di cella di questa feature</span>`;};
  g.appendChild(grp);});
const leg=document.getElementById("legend");
D.features_present.forEach(f=>{const d=document.createElement("div");
  d.innerHTML=`<i style="background:${fcolor[f]}"></i>${D.feature_names[f]}`; leg.appendChild(d);});
["F","T"].forEach((tid,i)=>{const [x,y]=xy(tid);
  const grp=document.createElementNS(NS,"g"); grp.setAttribute("class","term");
  grp.setAttribute("transform",`translate(${x},${y})`);
  const r=document.createElementNS(NS,"rect");
  r.setAttribute("x",-18);r.setAttribute("y",-16);r.setAttribute("width",36);r.setAttribute("height",32);
  r.setAttribute("fill",i?"#cdebcd":"#f3d0d0"); grp.appendChild(r);
  const t=document.createElementNS(NS,"text");t.setAttribute("text-anchor","middle");
  t.setAttribute("dy","5");t.textContent=i?"1":"0";grp.appendChild(t); g.appendChild(grp);});
// pan + zoom
let s=1,tx=420,ty=40,drag=false,px,py;
function upd(){g.setAttribute("transform",`translate(${tx},${ty}) scale(${s})`);}
upd();
svg.onwheel=e=>{e.preventDefault();const f=e.deltaY<0?1.1:0.9;s*=f;upd();};
svg.onmousedown=e=>{drag=true;px=e.clientX;py=e.clientY;svg.style.cursor="grabbing";};
window.onmousemove=e=>{if(!drag)return;tx+=e.clientX-px;ty+=e.clientY-py;px=e.clientX;py=e.clientY;upd();};
window.onmouseup=()=>{drag=false;svg.style.cursor="grab";};
</script></body></html>"""


def write_html(data: dict, out_path: Path) -> None:
    # riempie il template html con i dati del diagramma e lo salva
    html = (HTML.replace("__TITLE__", data["title"])
            .replace("__SUBTITLE__", data["subtitle"])
            .replace("__DATA__", json.dumps(data)))
    out_path.write_text(html, encoding="utf-8")


def _count_nodes(bdd, root) -> int:
    # conta i nodi veri del diagramma tenendo conto degli archi negati
    seen, stack = set(), [root]
    while stack:
        f = stack.pop()
        if f == bdd.true or f == bdd.false:
            continue
        reg = ~f if f.negated else f
        if reg in seen:
            continue
        seen.add(reg)
        stack += [reg.low, reg.high]
    return len(seen)


def pick_best(name: str, ks=(3, 2), max_nodes: int = 50):
    # sceglie il diagramma piu' ricco che resti leggibile e scarta i modelli troppo grandi
    rf = forest_io.load_forest(name)
    if sum(discretize_forest(rf, n_trees=2).bits(f)
           for f in range(rf.n_features_in_)) > 40:
        return None  # gia' con due alberi i bit sono troppi
    best = None
    for k in ks:
        if k > rf.n_estimators:
            continue
        dforest = discretize_forest(rf, n_trees=k)
        bdd = make_bdd()
        declare_vars(bdd, dforest)
        wins = win_regions(bdd, dforest, compose_forest(bdd, dforest))
        for label, w in wins.items():
            n = _count_nodes(bdd, w)
            if 1 <= n <= max_nodes and (best is None or n > best[2]):
                best = (k, label, n)
    return best


def batch_all(max_nodes: int, out_dir: Path) -> list[tuple]:
    # genera i viewer per tutti i modelli leggibili e una pagina indice
    out_dir.mkdir(parents=True, exist_ok=True)
    done = []
    for name in forest_io.iter_dataset_names():
        pick = pick_best(name, max_nodes=max_nodes)
        if pick is None:
            continue
        k, label, n = pick
        data = export_json(name, k, label)
        stem = f"obdd_interactive_{name}_k{k}_win{label}"
        write_html(data, out_dir / f"{stem}.html")
        (out_dir / f"{stem}.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
        done.append((name, k, label, n, f"{stem}.html"))
    done.sort(key=lambda r: r[3])
    rows = "\n".join(
        f'<li><a href="{f}">{name}</a> — k={k}, Win_{l}, {n} nodi</li>'
        for name, k, l, n, f in done)
    (out_dir / "index_obdd.html").write_text(
        "<!DOCTYPE html><meta charset=utf-8><title>OBDD visualizzabili</title>"
        "<h2>OBDD interattivi (modelli con diagramma leggibile)</h2>"
        f"<p>{len(done)} modelli, ordinati per numero di nodi.</p><ul>{rows}</ul>",
        encoding="utf-8")
    return done


def main(argv=None) -> None:
    # genera gli OBDD interattivi in html
    p = argparse.ArgumentParser(description="Esporta un OBDD Win_l in JSON + viewer HTML interattivo")
    p.add_argument("--dataset", default="iris")
    p.add_argument("--trees", type=int, default=3)
    p.add_argument("--label", type=int, default=1)
    p.add_argument("--all", action="store_true",
                   help="genera TUTTI i modelli con OBDD leggibile + index")
    p.add_argument("--max-nodes", type=int, default=50,
                   help="soglia di nodi per --all (default 50)")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    out_dir = Path(config.RESULTS_DIR) / "figures" / "obdd"
    if args.all:
        done = batch_all(args.max_nodes, out_dir)
        print(f"generati {len(done)} OBDD interattivi in {out_dir}")
        for name, k, label, n, _ in done:
            print(f"  {name:<18} k={k} Win_{label} {n} nodi")
        print(f"indice: {out_dir / 'index_obdd.html'}")
        return
    data = export_json(args.dataset, args.trees, args.label)
    out_dir = Path(config.RESULTS_DIR) / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else out_dir / \
        f"obdd_interactive_{args.dataset}_k{args.trees}_win{args.label}.html"
    write_html(data, out)
    json_out = out.with_suffix(".json")
    json_out.write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(f"{data['subtitle']}")
    print(f"viewer: {out}")
    print(f"dati:   {json_out}")


if __name__ == "__main__":
    main()
