# FASE 8 / Sez. 14: robustezza per campione e classe.

import argparse
import csv
from math import inf

from .. import config, forest_io
from ..bdd_backend import BACKEND, make_bdd
from ..compose import declare_vars
from ..discretize import cells_to_bits, discretize_forest, real_to_cells
from ..frontier import budgeted_forest
from ..robustness import (
    compile_frontier,
    escape_witness_compiled,
    robustness_witness_compiled,
)

FIELDS = [
    "dataset", "n_trees", "budget", "selector", "backend", "sample_idx",
    "true_class", "certified_class", "label", "targeted_radius",
    "is_untargeted_min", "flipped_bits", "flipped_features",
    "escape_radius", "escape_bits", "escape_features",
]


def run(name: str, k: int | None, budget: int, selector: str, out_path: str,
        seed: int = 0, max_samples: int | None = None) -> list[dict]:
    # calcola la robustezza per ogni campione e classe di un dataset
    rf = forest_io.load_forest(name)
    mapping = forest_io.label_encoding(name)  # controlla anche che la mappa delle etichette sia giusta
    dforest = discretize_forest(rf, n_trees=k)
    bdd = make_bdd()
    declare_vars(bdd, dforest)
    leaves = budgeted_forest(bdd, dforest, budget=budget, criterion=selector, rng=seed)
    compiled = compile_frontier(bdd, dforest, leaves)  # si compila una volta sola e ogni campione poi costa poco

    X, y = forest_io.load_split(name, "test")
    if max_samples is not None:
        X, y = X[:max_samples], y[:max_samples]  # taglio sul numero di campioni per i test set grandi
    rows = []
    for idx, (x, y_raw) in enumerate(zip(X, y)):
        bits = cells_to_bits(dforest, real_to_cells(dforest, x))
        prof = {
            l: robustness_witness_compiled(compiled, bits, l)
            for l in dforest.labels
        }
        certified = [l for l, (r, _) in prof.items() if r == 0]
        cert = certified[0] if certified else None
        rivals = {l: r for l, (r, _) in prof.items() if l != cert}
        untargeted = min(rivals.values(), default=inf)
        if cert is not None:  # la fuga ha senso solo per chi ha un certificato da perdere
            esc_r, esc_flips = escape_witness_compiled(compiled, bits, cert)
        else:
            esc_r, esc_flips = None, frozenset()
        for l, (radius, flips) in prof.items():
            rows.append({
                "dataset": name,
                "n_trees": dforest.n_trees,
                "budget": budget,
                "selector": selector,
                "backend": BACKEND,
                "sample_idx": idx,
                "true_class": mapping[y_raw],
                "certified_class": "" if cert is None else cert,
                "label": l,
                "targeted_radius": "inf" if radius == inf else radius,
                "is_untargeted_min": int(l != cert and radius == untargeted),
                "flipped_bits": "|".join(sorted(flips)),
                "flipped_features": "|".join(
                    sorted({b.split("_")[1] for b in flips})
                ),
                "escape_radius": "" if esc_r is None else ("inf" if esc_r == inf else esc_r),
                "escape_bits": "|".join(sorted(esc_flips)),
                "escape_features": "|".join(
                    sorted({b.split("_")[1] for b in esc_flips})
                ),
            })
    if out_path:
        config.RESULTS_DIR.mkdir(exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    return rows


def main(argv=None) -> None:
    # lancia la robustezza su un singolo dataset e stampa il riassunto
    p = argparse.ArgumentParser(description="Robustezza (MinSwitch) sul test set di un dataset")
    p.add_argument("--dataset", required=True)
    p.add_argument("--trees", type=int, default=None, help="prefisso di k alberi (default: forest piena)")
    p.add_argument("--budget", type=int, default=config.BUDGET_GRID[0])
    p.add_argument("--selector", default="infogain", choices=config.SELECTORS)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    out = args.out or str(config.RESULTS_DIR / f"robustness_{args.dataset}.csv")
    rows = run(args.dataset, args.trees, args.budget, args.selector, out)
    n_samples = len({r["sample_idx"] for r in rows})
    certified = len({r["sample_idx"] for r in rows if r["certified_class"] != ""})
    # solo campioni certificati e ognuno conta una volta anche se due rivali pareggiano
    unt = {
        r["sample_idx"]: float(r["targeted_radius"])
        for r in rows
        if r["certified_class"] != ""
        and r["is_untargeted_min"]
        and r["targeted_radius"] != "inf"
    }
    radii = sorted(unt.values())
    print(f"{args.dataset}: {n_samples} sample, {certified} certificati")
    if radii:
        from statistics import median

        print(
            f"raggio untargeted: min={radii[0]} mediana={median(radii)} "
            f"max={radii[-1]} (i piu' fragili sono i nonzero piu' piccoli)"
        )


if __name__ == "__main__":
    main()
    import os
    import sys

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)  # esce di colpo per evitare il rumore del distruttore di dd alla chiusura
