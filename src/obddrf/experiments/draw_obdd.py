# FASE 8 / Sez. 14: rendering di piccoli OBDD didattici.

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

from .. import config


def _collect(bdd, root, order):
    # raccoglie i nodi raggiungibili con il loro livello raddrizzando gli archi negati
    pos = {v: i for i, v in enumerate(order)}
    nodes, edges = {}, []
    ids = {}

    def cof(f):
        # restituisce i cofattori raddrizzando gli archi negati
        return (~f.low, ~f.high) if f.negated else (f.low, f.high)

    def visit(f):
        # visita il diagramma e raccoglie nodi e archi
        if f == bdd.true:
            return "T"
        if f == bdd.false:
            return "F"
        if f in ids:
            return ids[f]
        nid = f"n{len(ids)}"
        ids[f] = nid
        nodes[nid] = (f.var, pos[f.var])
        lo, hi = cof(f)
        edges.append((nid, visit(lo), 0))
        edges.append((nid, visit(hi), 1))
        return nid

    root_id = visit(root)
    return root_id, nodes, edges


def draw(bdd, root, order, title, out_path):
    # disegna l'OBDD con matplotlib e lo salva come immagine
    root_id, nodes, edges = _collect(bdd, root, order)
    n_levels = len(order)
    # ogni variabile sta su una riga e i nodi si distribuiscono in orizzontale
    by_level = {}
    for nid, (_, lev) in nodes.items():
        by_level.setdefault(lev, []).append(nid)
    xpos = {}
    for lev, ids in by_level.items():
        for j, nid in enumerate(sorted(ids)):
            xpos[nid] = (j - (len(ids) - 1) / 2) * 2.2
    ypos = {nid: -lev for nid, (_, lev) in nodes.items()}
    # terminali in fondo
    term_y = -(n_levels + 0.5)
    term_x = {"F": -1.2, "T": 1.2}

    fig, ax = plt.subplots(figsize=(7, max(4, n_levels * 1.1 + 2)))

    def xy(nid):
        # restituisce la posizione a schermo di un nodo
        if nid in ("T", "F"):
            return term_x[nid], term_y
        return xpos[nid], ypos[nid]

    for src, dst, branch in edges:
        x0, y0 = xy(src)
        x1, y1 = xy(dst)
        style = "-" if branch else "--"  # ramo alto pieno e ramo basso tratteggiato
        col = "tab:blue" if branch else "tab:gray"
        ax.add_patch(FancyArrowPatch((x0, y0 - 0.18), (x1, y1 + 0.18),
                                     arrowstyle="-|>", mutation_scale=12,
                                     linestyle=style, color=col, lw=1.6,
                                     shrinkA=0, shrinkB=0))
    for nid, (var, _) in nodes.items():
        x, y = xy(nid)
        ax.add_patch(Circle((x, y), 0.26, fc="white", ec="black", lw=1.5, zorder=3))
        ax.text(x, y, var, ha="center", va="center", fontsize=8, zorder=4)
    for t, lab in (("T", "1"), ("F", "0")):
        x, y = term_x[t], term_y
        ax.add_patch(Rectangle((x - 0.25, y - 0.22), 0.5, 0.44,
                               fc="#eee", ec="black", zorder=3))
        ax.text(x, y, lab, ha="center", va="center", fontsize=10,
                fontweight="bold", zorder=4)

    ax.plot([], [], "-", color="tab:blue", label="bit = 1 (high)")
    ax.plot([], [], "--", color="tab:gray", label="bit = 0 (low)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(title, fontsize=11)
    ax.set_xlim(-5, 5)
    ax.set_ylim(term_y - 0.8, 1)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main(argv=None):
    # genera le due figure OBDD didattiche
    import sys

    sys.path.insert(0, "tests")
    import numpy as np

    from obddrf.bdd_backend import make_bdd
    from obddrf.comparator import node_test
    from obddrf.compose import compose_forest, declare_vars, win_regions
    from obddrf.discretize import DiscretizedForest

    out = Path(config.RESULTS_DIR) / "figures"
    out.mkdir(parents=True, exist_ok=True)

    # primo esempio il comparatore su una feature con sei celle
    df1 = DiscretizedForest(1, 1, [0, 1], [np.arange(5, dtype=float)], [], [])
    bdd = make_bdd()
    bdd.declare(*df1.input_bit_names())
    nt = node_test(bdd, df1, 0, 2)  # test indice di cella al massimo due
    draw(bdd, nt, df1.input_bit_names(),
         "Comparator nt(f, tau): \"cell index X_f <= 2\"  (Sec. 5)",
         out / "obdd_1_comparatore.png")

    # secondo esempio la regione certificata di un modellino con due alberi
    from oracle_utils import make_dforest

    dforest = make_dforest([("stump", 0, 0, 1), ("stump", 1, 1, 0)], labels=[0, 1])
    bdd2 = make_bdd()
    declare_vars(bdd2, dforest)
    wins = win_regions(bdd2, dforest, compose_forest(bdd2, dforest))
    draw(bdd2, wins[1], dforest.input_bit_names(),
         "Certified region Win_1 of a toy model (2 trees, 1 feature)",
         out / "obdd_2_win_region.png")

    print("figure OBDD scritte in", out)
    for f in ("obdd_1_comparatore.png", "obdd_2_win_region.png"):
        print(" -", f)


if __name__ == "__main__":
    main()
