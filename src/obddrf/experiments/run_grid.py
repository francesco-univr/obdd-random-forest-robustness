# FASE 8 / Sez. 14: griglia modello x budget x selettore.

import argparse
import csv
import multiprocessing as mp
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .. import config, forest_io
from ..bdd_backend import BACKEND, make_bdd
from ..compose import declare_vars
from ..discretize import discretize_forest
from ..frontier import budgeted_forest
from .resources import ResourceLimitExceeded, ResourceMonitor

FIELDS = [
    "dataset", "n_trees", "n_trees_full", "budget", "selector", "backend",
    "completed", "error", "n_input_bits", "peak_nodes", "n_leaves", "runtime_s",
]


def run_triple(rf, name: str, k: int, budget: int, selector: str,
               timeout_s: float, max_nodes: int, seed: int = 0) -> dict:
    # una singola prova su un modello con un budget e un selettore dentro un manager nuovo
    row = {
        "dataset": name, "n_trees": k, "n_trees_full": rf.n_estimators,
        "budget": budget, "selector": selector, "backend": BACKEND,
        "completed": False, "error": "", "n_input_bits": "",
        "peak_nodes": "", "n_leaves": "", "runtime_s": "",
    }
    t0 = time.monotonic()
    try:
        dforest = discretize_forest(rf, n_trees=k)
        row["n_input_bits"] = sum(dforest.bits(f) for f in range(dforest.n_features))
        bdd = make_bdd()
        declare_vars(bdd, dforest)
        monitor = ResourceMonitor(timeout_s, max_nodes)
        leaves = budgeted_forest(
            bdd, dforest, budget=budget, criterion=selector, rng=seed,
            on_step=lambda t, acc: monitor.check(bdd, acc),
        )
        row.update(completed=True, peak_nodes=monitor.peak, n_leaves=len(leaves))
    except ResourceLimitExceeded as e:
        row["error"] = str(e)
    except MemoryError:
        row["error"] = "MemoryError"
    row["runtime_s"] = round(time.monotonic() - t0, 3)
    return row


def _triple_worker(conn, name, k, budget, selector, timeout_s, max_nodes) -> None:
    # esegue una prova in un sottoprocesso e manda la riga al padre
    import os

    rf = forest_io.load_forest(name)
    conn.send(run_triple(rf, name, k, budget, selector, timeout_s, max_nodes))
    conn.close()
    os._exit(0)  # esce di colpo perche' la riga e' gia' nel pipe e il distruttore di dd farebbe solo rumore


def run_triple_isolated(name, k, budget, selector, timeout_s, max_nodes,
                        hard_timeout_s, n_trees_full) -> dict:
    # stessa prova ma dentro un sottoprocesso che si puo' uccidere se resta bloccato dentro una formula
    parent_conn, child_conn = mp.Pipe(duplex=False)
    proc = mp.Process(
        target=_triple_worker,
        args=(child_conn, name, k, budget, selector, timeout_s, max_nodes),
        daemon=True,
    )
    t0 = time.monotonic()
    proc.start()
    child_conn.close()
    proc.join(hard_timeout_s)
    base = {
        "dataset": name, "n_trees": k, "n_trees_full": n_trees_full,
        "budget": budget, "selector": selector, "backend": BACKEND,
        "completed": False, "error": "", "n_input_bits": "",
        "peak_nodes": "", "n_leaves": "",
        "runtime_s": round(time.monotonic() - t0, 3),
    }
    if proc.is_alive():
        proc.terminate()
        proc.join()
        base["error"] = f"hard kill dopo {hard_timeout_s:.0f}s (formula fuori dai check cooperativi)"
        return base
    if parent_conn.poll():
        return parent_conn.recv()
    base["error"] = f"processo terminato (exit {proc.exitcode}), probabile out-of-memory"
    return base


def main(argv=None) -> None:
    # lancia la campagna sulla griglia e scrive il csv
    p = argparse.ArgumentParser(description="Griglia modello x budget x selettore sui 54 modelli con test progressivo e ricerca del muro")
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--budgets", nargs="*", type=int, default=config.BUDGET_GRID)
    p.add_argument("--selectors", nargs="*", default=config.SELECTORS)
    p.add_argument("--trees", nargs="*", type=int, default=config.PROGRESSIVE_TREE_COUNTS)
    p.add_argument("--timeout", type=float, default=config.TIMEOUT_PER_MODEL_S)
    p.add_argument("--hard-timeout", type=float, default=None,
                   help="kill del sottoprocesso, default timeout+60s")
    p.add_argument("--max-nodes", type=int, default=config.MAX_MANAGER_NODES)
    p.add_argument("--out", default=str(config.RESULTS_DIR / "grid.csv"))
    p.add_argument("--resume", action="store_true",
                   help="riparte dal CSV esistente saltando le triple gia' registrate")
    p.add_argument("--locate-wall", action="store_true",
                   help="bisezione fra ultimo k riuscito e primo fallito per localizzare il muro")
    p.add_argument("--jobs", type=int, default=1,
                   help="coppie dataset/budget/selettore in parallelo, dimensionare sulla RAM")
    args = p.parse_args(argv)
    hard_timeout = args.hard_timeout or args.timeout + 60  # oltre il timeout normale la prova non puo' comunque piu' riuscire

    names = args.datasets or forest_io.iter_dataset_names()
    config.RESULTS_DIR.mkdir(exist_ok=True)

    done: dict[tuple, bool] = {}
    mode = "w"
    if args.resume and Path(args.out).exists():
        with open(args.out, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                key = (r["dataset"], int(r["n_trees"]), int(r["budget"]), r["selector"])
                done[key] = r["completed"] == "True"
        mode = "a"
        print(f"resume: {len(done)} triple gia' registrate", file=sys.stderr)

    lock = threading.Lock()  # protegge la scrittura del csv e la mappa delle prove gia' fatte

    with open(args.out, mode, newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if mode == "w":
            writer.writeheader()

        def measure(name, n_full, k, budget, selector) -> bool:
            # esegue la prova oppure la recupera dal csv se era gia' stata fatta
            key = (name, k, budget, selector)
            with lock:
                if key in done:
                    return done[key]
            row = run_triple_isolated(  # la prova e' lenta e gira fuori dal lucchetto
                name, k, budget, selector, args.timeout,
                args.max_nodes, hard_timeout, n_full,
            )
            with lock:
                done[key] = row["completed"]
                writer.writerow(row)
                fh.flush()
                print(
                    f"{name} k={k} B={budget} {selector}: "
                    f"{'ok' if row['completed'] else row['error']} "
                    f"({row['runtime_s']}s, peak={row['peak_nodes']}, "
                    f"leaves={row['n_leaves']})",
                    file=sys.stderr,
                    flush=True,
                )
            return row["completed"]

        def run_pair(name, n_full, ks, budget, selector) -> None:
            # fa salire il numero di alberi finche' riesce e poi cerca con la bisezione il punto esatto del muro
            k_ok, k_fail = 0, None
            for k in ks:
                if measure(name, n_full, k, budget, selector):
                    k_ok = k
                else:
                    k_fail = k
                    break
            if args.locate_wall and k_fail is not None:
                while k_fail - k_ok > 1:
                    mid = (k_ok + k_fail) // 2
                    if measure(name, n_full, mid, budget, selector):
                        k_ok = mid
                    else:
                        k_fail = mid

        pairs = []
        for name in names:
            n_full = forest_io.load_forest(name).n_estimators
            ks = sorted({min(k, n_full) for k in args.trees} | {n_full})
            for budget in args.budgets:
                for selector in args.selectors:
                    pairs.append((name, n_full, ks, budget, selector))

        if args.jobs <= 1:
            for pair in pairs:
                run_pair(*pair)
        else:
            # le coppie sono indipendenti quindi possono girare insieme come dice il Remark 12
            with ThreadPoolExecutor(max_workers=args.jobs) as pool:
                for _ in pool.map(lambda pr: run_pair(*pr), pairs):
                    pass


if __name__ == "__main__":
    main()
    import os

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)  # esce di colpo per evitare il rumore del distruttore di dd alla chiusura
