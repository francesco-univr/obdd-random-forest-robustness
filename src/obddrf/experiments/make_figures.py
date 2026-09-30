# FASE 8 / Sez. 14: figure derivate dai CSV della campagna.

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # nessuna finestra solo file
import matplotlib.pyplot as plt
import pandas as pd

from .. import config


def _load_grid(path):
    # legge il csv della griglia e sistema i tipi delle colonne
    df = pd.read_csv(path)
    df["completed"] = df["completed"].astype(str) == "True"
    for c in ("n_leaves", "peak_nodes", "n_trees", "n_trees_full"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def fig_scaling_wall(grid, overview, out):
    # figura del muro con la frazione di forest compilata contro i bit di input
    ov = overview.set_index("dataset")["n_input_bits"]
    rows = []
    for name, g in grid.groupby("dataset"):
        gok = g[g.completed]
        full = int(g["n_trees_full"].iloc[0])
        best = int(gok["n_trees"].max()) if len(gok) else 0
        rows.append((ov.get(name, float("nan")), best / full, best == full))
    bits = [r[0] for r in rows]
    frac = [r[1] for r in rows]
    full = [r[2] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter([b for b, f in zip(bits, full) if f],
               [c for c, f in zip(frac, full) if f],
               c="tab:green", label="full forest compiled", s=40)
    ax.scatter([b for b, f in zip(bits, full) if not f],
               [c for c, f in zip(frac, full) if not f],
               c="tab:red", label="prefix only", s=40)
    ax.set_xscale("log")
    ax.set_xlabel("Encoded input bits on a log scale")
    ax.set_ylabel("Compiled forest fraction (k max / D)")
    ax.set_title("Scaling wall as encoded input bits grow")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "1_muro_scalabilita.png", dpi=130)
    plt.close(fig)


def fig_budget(grid, out):
    # figura del compromesso del budget con le fette prodotte per ogni tetto
    spl = grid[grid.completed & (grid.n_leaves > 1)]
    budgets = sorted(spl.budget.unique())
    data = [spl[spl.budget == b]["n_leaves"].values for b in budgets]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot(data, tick_labels=[str(b) for b in budgets], showfliers=False)
    ax.set_yscale("log")
    ax.set_xlabel("Node budget B")
    ax.set_ylabel("Frontier leaves on a log scale")
    ax.set_title("Small budgets create many frontier leaves")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "2_tradeoff_budget.png", dpi=130)
    plt.close(fig)


def fig_selectors(grid, out):
    # figura che confronta i cinque selettori su fette prodotte e tempo
    spl = grid[grid.completed & (grid.n_leaves > 1)]
    sels = sorted(spl.selector.unique())
    med_leaves = [spl[spl.selector == s]["n_leaves"].median() for s in sels]
    med_rt = [spl[spl.selector == s]["runtime_s"].median() for s in sels]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
    a1.bar(sels, med_leaves, color="tab:blue")
    a1.set_ylabel("Median frontier leaves")
    a1.set_title("Leaves by selector")
    a1.tick_params(axis="x", rotation=30)
    a2.bar(sels, med_rt, color="tab:orange")
    a2.set_ylabel("Median runtime (s)")
    a2.set_title("Runtime by selector")
    a2.tick_params(axis="x", rotation=30)
    fig.suptitle("Five selectors on split-active configurations")
    fig.tight_layout()
    fig.savefig(out / "3_confronto_selettori.png", dpi=130)
    plt.close(fig)


def fig_robustness(robust_dir, out):
    # figura con la distribuzione dei raggi di robustezza di tutti i modelli trattabili
    radii = []
    for csvf in Path(robust_dir).glob("*.csv"):
        df = pd.read_csv(csvf)
        # si tengono solo i campioni certificati perche' per gli altri il raggio misura un altro concetto
        sub = df[(df["is_untargeted_min"] == 1) & (df["targeted_radius"] != "inf")
                 & df["certified_class"].notna()]
        # ogni campione conta una volta anche se due rivali pareggiano alla stessa distanza
        sub = sub.drop_duplicates(subset=["sample_idx"])
        radii.extend(pd.to_numeric(sub["targeted_radius"], errors="coerce").dropna())
    if not radii:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    lo, hi = int(min(radii)), int(max(radii))
    ax.hist(radii, bins=range(lo, hi + 2), align="left",
            color="tab:purple", edgecolor="white", rwidth=0.9)
    ax.set_xlabel("Nearest-rival radius in bit flips")
    ax.set_ylabel("Test samples")
    ax.set_title("Robustness distribution across tractable models")
    ax.set_xticks(range(lo, hi + 1))
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "4_distribuzione_robustezza.png", dpi=130)
    plt.close(fig)


def main(argv=None) -> None:
    # genera le quattro figure di studio dai csv
    p = argparse.ArgumentParser(description="Genera le 4 figure di studio (muro, budget, selettori, robustezza) da CSV gia' calcolati")
    p.add_argument("--grid", default=str(config.RESULTS_DIR / "grid_campagna.csv"))
    p.add_argument("--overview", default=str(config.RESULTS_DIR / "datasets_overview.csv"))
    p.add_argument("--robust-dir", default=str(config.RESULTS_DIR / "robustness"))
    p.add_argument("--out", default=str(config.RESULTS_DIR / "figures"))
    args = p.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    grid = _load_grid(args.grid)
    overview = pd.read_csv(args.overview)
    fig_scaling_wall(grid, overview, out)
    fig_budget(grid, out)
    fig_selectors(grid, out)
    fig_robustness(args.robust_dir, out)
    print(f"figure scritte in {out}:")
    for f in sorted(out.glob("*.png")):
        print(" -", f.name)


if __name__ == "__main__":
    main()
