# FASE 8 / Sez. 14: riepilogo descrittivo dei 54 dataset.

import argparse
import csv

from .. import config, forest_io
from ..discretize import discretize_forest

FIELDS = [
    "dataset", "n_classes", "n_features", "n_train", "n_test",
    "n_trees", "max_depth", "total_cells", "max_cells", "n_input_bits",
    "test_acc",
]


def describe(name: str) -> dict:
    # misura un dataset contando classi feature celle bit alberi e accuratezza
    bundle = forest_io.load_bundle(name)
    rf = bundle["model"]
    dforest = discretize_forest(rf)  # si usa la forest piena per avere le celle vere del modello
    cells = [dforest.cells(f) for f in range(dforest.n_features)]
    bits = sum(dforest.bits(f) for f in range(dforest.n_features))
    X_tr, _ = forest_io.load_split(name, "train")
    X_te, _ = forest_io.load_split(name, "test")
    depths = [e.get_depth() for e in rf.estimators_]
    return {
        "dataset": name,
        "n_classes": len(rf.classes_),
        "n_features": dforest.n_features,
        "n_train": len(X_tr),
        "n_test": len(X_te),
        "n_trees": rf.n_estimators,
        "max_depth": max(depths),
        "total_cells": sum(cells),
        "max_cells": max(cells),
        "n_input_bits": bits,
        "test_acc": round(float(bundle["test_acc"]), 4),
    }


def main(argv=None) -> None:
    # scrive la tabella coi numeri chiave dei 54 dataset
    p = argparse.ArgumentParser(description="Tabella descrittiva dei 54 dataset: dimensioni, classi, celle, bit di input, accuratezza")
    p.add_argument("--out", default=str(config.RESULTS_DIR / "datasets_overview.csv"))
    args = p.parse_args(argv)
    config.RESULTS_DIR.mkdir(exist_ok=True)
    rows = [describe(n) for n in forest_io.iter_dataset_names()]
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    bits = sorted(r["n_input_bits"] for r in rows)
    print(f"{len(rows)} dataset -> {args.out}")
    print(f"bit di input: min {bits[0]}, mediana {bits[len(bits)//2]}, max {bits[-1]}")
    print(f"classi: {sorted({r['n_classes'] for r in rows})}")


if __name__ == "__main__":
    main()
