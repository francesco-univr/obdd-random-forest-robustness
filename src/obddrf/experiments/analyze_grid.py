# FASE 8 / Sez. 14: analisi del CSV della griglia.

import argparse
import re
import sys

import pandas as pd

from .. import config


def error_kind(err: str) -> str:
    # traduce il messaggio di errore in una parola sola
    if not err:
        return ""
    if "timeout" in err:
        return "timeout"
    if "hard kill" in err:
        return "hard kill"
    if "manager" in err:
        return "nodi"
    return "altro"


def load(path: str) -> pd.DataFrame:
    # legge il csv e sistema i tipi delle colonne
    df = pd.read_csv(path)
    df["completed"] = df["completed"].astype(str) == "True"
    df["error_kind"] = df["error"].fillna("").map(error_kind)
    return df


def main(argv=None) -> None:
    # stampa il riassunto della griglia con muri budget e selettori
    p = argparse.ArgumentParser(description="Riassume il CSV della griglia: completamento, muri, budget, selettori")
    p.add_argument("csv", nargs="?", default=str(config.RESULTS_DIR / "grid.csv"))
    args = p.parse_args(argv)
    df = load(args.csv)
    ok = df[df.completed]

    print(" PANORAMICA")
    print(f"triple: {len(df)} | completate: {len(ok)} | dataset: {df.dataset.nunique()}")
    print(df[~df.completed].error_kind.value_counts().to_string())

    print("\n PER (BUDGET, SELETTORE)")
    for (b, s), g in df.groupby(["budget", "selector"]):
        gok = g[g.completed]
        full = gok[gok.n_trees == gok.n_trees_full].dataset.nunique()
        print(
            f"B={b:>7} {s:<10} ok {len(gok):>3}/{len(g):>3} triple | "
            f"forest PIENA: {full:>2} dataset | "
            f"mediana foglie {gok.n_leaves.median():>6.0f} | "
            f"mediana peak {gok.peak_nodes.median():>8.0f} | "
            f"mediana t {gok.runtime_s.median():>6.2f}s"
        )

    print("\nMURO PER MODELLO (miglior k completato su qualunque config)")
    walls = []
    for name, g in df.groupby("dataset"):
        gok = g[g.completed]
        full_d = int(g.n_trees_full.iloc[0])
        best_k = int(gok.n_trees.max()) if len(gok) else 0
        fails_above = g[~g.completed & (g.n_trees > best_k)]
        next_fail = int(fails_above.n_trees.min()) if len(fails_above) else None
        walls.append((name, best_k, next_fail, full_d))
    full = sorted(n for n, k, nf, d in walls if k == d)
    partial = sorted((n, k, nf, d) for n, k, nf, d in walls if 0 < k < d)
    zero = sorted(n for n, k, nf, d in walls if k == 0)
    print(f"FOREST PIENA ({len(full)}): {', '.join(full)}")
    print(f"PARZIALI ({len(partial)}):  [muro = ultimo ok -> primo fallito]")
    for n, k, nf, d in partial:
        gap = "" if nf is None else f" -> fallisce a k={nf}" + (
            " (LOCALIZZATO)" if nf == k + 1 else f" (gap {nf - k})"
        )
        print(f"  {n}: ok fino a k={k}{gap}, D={d}")
    print(f"NIENTE (nemmeno k minimo) ({len(zero)}): {', '.join(zero)}")

    def _compare(dim: str, others: list[str], splits_only: bool = False) -> None:
        # confronta budget o selettori solo sulle prove completate da tutti per essere onesti
        # con splits_only restano solo le prove dove almeno un taglio e' avvenuto perche' solo li' il criterio conta
        vals = sorted(df[dim].unique())
        tables = {v: ok[ok[dim] == v].set_index(others) for v in vals}
        common = None
        for t in tables.values():
            common = t.index if common is None else common.intersection(t.index)
        if splits_only and len(common):
            mask = sum((tables[v].loc[common].n_leaves for v in vals)) > len(vals)
            common = common[mask.values]
        label = "con split effettivo" if splits_only else "completate da tutti"
        print(f"  {len(common)} triple {label} ({len(vals)} valori)")
        if not len(common):
            return
        print(f"  {'valore':<12} {'foglie(med)':>12} {'peak(med)':>12} {'runtime(med)':>13}")
        for v in vals:
            sub = tables[v].loc[common]
            print(
                f"  {str(v):<12} {sub.n_leaves.median():>12.0f} "
                f"{sub.peak_nodes.median():>12.0f} {sub.runtime_s.median():>12.2f}s"
            )

    print("\n CONFRONTO BUDGET (a parita' di modello/k/selettore)")
    _compare("budget", ["dataset", "n_trees", "selector"])
    print("  -- solo dove c'e' stato split (foglie > 1):")
    _compare("budget", ["dataset", "n_trees", "selector"], splits_only=True)

    print("\n CONFRONTO SELETTORI (a parita' di modello/k/budget)")
    _compare("selector", ["dataset", "n_trees", "budget"])
    print("  -- solo dove c'e' stato split (foglie > 1), il regime che conta:")
    _compare("selector", ["dataset", "n_trees", "budget"], splits_only=True)
    # quanto spesso ogni selettore porta a termine le prove
    print("  completamento per selettore:")
    for s in sorted(df.selector.unique()):
        g = df[df.selector == s]
        print(f"    {s:<12} {g.completed.sum():>4}/{len(g):<4} ({100*g.completed.mean():.0f}%)")

    print("\n CANDIDATI AL RICONTROLLO (falliti per nodi vicino al tetto)")
    cand = []
    for _, r in df[df.error_kind == "nodi"].iterrows():
        m = re.search(r"manager a (\d+) nodi > (\d+)", r.error)
        if m and int(m.group(1)) < 2 * int(m.group(2)):
            cand.append(f"{r.dataset} k={r.n_trees} B={r.budget} {r.selector} ({m.group(1)})")
    print("\n".join(cand) if cand else "nessuno (tutti morti ben oltre il tetto)")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
