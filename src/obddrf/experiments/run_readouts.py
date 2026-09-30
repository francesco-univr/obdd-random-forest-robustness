# FASE 8 / Sez. 14: conteggi, volume e ranking per bit.

import argparse
import csv
import math
import multiprocessing as mp
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .. import config, forest_io
from ..bdd_backend import make_bdd
from ..compose import compose_forest, declare_vars, win_regions
from ..discretize import discretize_forest
from ..readouts import model_count, pvalues, shapley_values, weighted_count
from .run_robustness_batch import pick_configs

SUMMARY_FIELDS = [
    "dataset", "status", "n_trees", "n_input_bits", "n_classes",
    "certified_total", "certified_frac", "top_infogain", "top_pvalue",
    "top_shapley", "runtime_s",
]


def _entropy(q: float) -> float:
    # entropia binaria di una probabilita
    if q <= 0.0 or q >= 1.0:
        return 0.0
    return -q * math.log2(q) - (1 - q) * math.log2(1 - q)


def compute_readouts(name: str, cfg: dict, top: int, max_bits_shapley: int,
                     out_dir: Path | None = None) -> dict:
    # compone l'OBDD intero del modello e ne legge conteggi volumi e classifiche dei bit
    rf = forest_io.load_forest(name)
    dforest = discretize_forest(rf, n_trees=cfg["n_trees"])
    ordered = dforest.input_bit_names()
    n_bits = len(ordered)
    bdd = make_bdd()
    declare_vars(bdd, dforest)
    acc = compose_forest(bdd, dforest)
    wins = win_regions(bdd, dforest, acc)

    # con peso un mezzo il volume e' la frazione di spazio certificata come da Prop 5
    half = {v: (0.5, 0.5) for v in ordered}
    counts, vols = {}, {}
    union = bdd.false
    for l, w in wins.items():
        counts[l] = model_count(bdd, w, n_bits)
        vols[l] = weighted_count(bdd, w, ordered, half)
        union |= w
    total = sum(counts.values())

    # la classifica dei bit si calcola sull'unione delle regioni certificate
    N = model_count(bdd, union, n_bits)
    ig = {}
    for v in ordered:
        n1 = model_count(bdd, union & bdd.var(v), n_bits)
        ig[v] = _entropy(n1 / N) if N else 0.0
    pv = pvalues(bdd, union, ordered)
    top_ig = sorted(ordered, key=lambda v: ig[v], reverse=True)[:top]
    top_pv = sorted(ordered, key=lambda v: pv[v])[:top]  # vince il p-value piu' piccolo
    if n_bits <= max_bits_shapley:
        sh = shapley_values(bdd, union, ordered)
        top_sh = sorted(ordered, key=lambda v: sh[v])[:top]  # vince lo Shapley piu' negativo
    else:
        top_sh = [f"(saltato: {n_bits} bit > {max_bits_shapley})"]

    # csv di dettaglio del modello con le righe per classe e per bit
    detail = (out_dir or Path(config.RESULTS_DIR) / "readouts") / f"{name}.csv"
    detail.parent.mkdir(parents=True, exist_ok=True)
    with open(detail, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["kind", "key", "count_or_value"])
        for l in counts:
            w.writerow(["class_count", l, counts[l]])
            w.writerow(["class_volume_frac", l, vols[l]])
        for v in ordered:
            w.writerow(["bit_infogain", v, ig[v]])
            w.writerow(["bit_pvalue", v, pv[v]])

    return {
        "dataset": name, "status": "ok", "n_trees": cfg["n_trees"],
        "n_input_bits": n_bits, "n_classes": len(rf.classes_),
        "certified_total": total,
        "certified_frac": (total / 2**n_bits if n_bits else 0.0),
        "top_infogain": "|".join(top_ig),
        "top_pvalue": "|".join(top_pv),
        "top_shapley": "|".join(map(str, top_sh)),
        "runtime_s": "",
    }


def _worker(conn, name, cfg, top, max_bits_shapley) -> None:
    # calcola i readout in un sottoprocesso e li manda al padre
    import os

    conn.send(compute_readouts(name, cfg, top, max_bits_shapley))
    conn.close()
    os._exit(0)  # esce di colpo per evitare il rumore del distruttore di dd


def run_one(name, cfg, top, max_bits_shapley, timeout_s) -> dict:
    # lancia un modello in un sottoprocesso e lo uccide se sfora il tempo massimo
    parent, child = mp.Pipe(duplex=False)
    proc = mp.Process(target=_worker, args=(child, name, cfg, top, max_bits_shapley),
                      daemon=True)
    t0 = time.monotonic()
    proc.start()
    child.close()
    proc.join(timeout_s)
    base = {f: "" for f in SUMMARY_FIELDS}
    base.update(dataset=name, n_trees=cfg["n_trees"],
                runtime_s=round(time.monotonic() - t0, 2))
    if proc.is_alive():
        proc.terminate()
        proc.join()
        base["status"] = f"timeout {timeout_s:.0f}s"
        return base
    if parent.poll():
        row = parent.recv()
        row["runtime_s"] = round(time.monotonic() - t0, 2)
        return row
    base["status"] = f"crash (exit {proc.exitcode})"
    return base


def main(argv=None) -> None:
    # lancia i readout su tutti i modelli e scrive il riepilogo
    p = argparse.ArgumentParser(description="Readouts (model count, volume, ranking per-bit) sui 54 modelli ottenuti")
    p.add_argument("--grid", default=str(config.RESULTS_DIR / "grid_campagna.csv"))
    p.add_argument("--summary", default=str(config.RESULTS_DIR / "readouts_summary.csv"))
    p.add_argument("--top", type=int, default=5)
    p.add_argument("--max-bits-shapley", type=int, default=60)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args(argv)

    configs = pick_configs(args.grid)
    names = forest_io.iter_dataset_names()
    done: set[str] = set()
    mode = "w"
    if args.resume and Path(args.summary).exists():
        with open(args.summary, newline="", encoding="utf-8") as fh:
            done = {r["dataset"] for r in csv.DictReader(fh)}
        mode = "a"

    lock = threading.Lock()
    with open(args.summary, mode, newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
        if mode == "w":
            writer.writeheader()

        def handle(name) -> None:
            # elabora un modello e scrive la sua riga di riepilogo
            if name in done:
                return
            cfg = configs.get(name)
            if cfg is None:  # nessuna prova riuscita quindi il modello non si compila
                row = {f: "" for f in SUMMARY_FIELDS}
                row.update(dataset=name, status="non compilabile")
            else:
                row = run_one(name, cfg, args.top, args.max_bits_shapley, args.timeout)
            with lock:
                writer.writerow(row)
                fh.flush()
                print(f"{name}: {row['status']} | cert={row['certified_total']} "
                      f"frac={row['certified_frac']} IG[0]={row['top_infogain'][:18]} "
                      f"({row['runtime_s']}s)", file=sys.stderr, flush=True)

        todo = [n for n in names if n not in done]
        if args.jobs <= 1:
            for n in todo:
                handle(n)
        else:
            with ThreadPoolExecutor(max_workers=args.jobs) as pool:
                for _ in pool.map(handle, todo):
                    pass


if __name__ == "__main__":
    main()
    import os

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)  # esce di colpo per evitare il rumore del distruttore di dd alla chiusura
