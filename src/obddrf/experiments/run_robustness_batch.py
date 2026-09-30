# FASE 8 / Sez. 14: batch di robustezza sui modelli trattabili.

import argparse
import csv
import multiprocessing as mp
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import median

from .. import config, forest_io
from . import run_robustness

SUMMARY_FIELDS = [
    "dataset", "n_trees", "n_trees_full", "budget", "selector", "status",
    "n_samples", "n_certified", "untargeted_min", "untargeted_median",
    "untargeted_max", "escape_min", "escape_median", "escape_max", "runtime_s",
]


def pick_configs(grid_csv: str) -> dict[str, dict]:
    # per ogni dataset prende la prova riuscita con piu' alberi e a parita' quella con meno fette
    best: dict[str, dict] = {}
    with open(grid_csv, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["completed"] != "True":
                continue
            name = r["dataset"]
            k, leaves = int(float(r["n_trees"])), int(float(r["n_leaves"]))  # il csv puo' avere numeri scritti come 19.0 quindi si passa da float
            cur = best.get(name)
            # vince chi ha piu' alberi e a parita' chi ha meno fette
            if cur is None or (k, -leaves) > (cur["n_trees"], -cur["n_leaves"]):
                best[name] = {
                    "n_trees": k, "n_leaves": leaves,
                    "n_trees_full": int(float(r["n_trees_full"])),
                    "budget": int(float(r["budget"])), "selector": r["selector"],
                }
    return best


def _summarize(name, cfg, rows, runtime) -> dict:
    # riassume le righe di un dataset in una sola riga e le statistiche usano solo i campioni certificati
    # ogni campione conta una volta sola anche quando due rivali pareggiano alla stessa distanza
    cert_rows = [r for r in rows if str(r["certified_class"]) != ""]
    unt_by_sample: dict = {}
    for r in cert_rows:
        if str(r["is_untargeted_min"]) == "1" and str(r["targeted_radius"]) != "inf":
            unt_by_sample[r["sample_idx"]] = float(r["targeted_radius"])
    untargeted = sorted(unt_by_sample.values())
    esc_by_sample: dict = {}
    for r in cert_rows:
        val = str(r.get("escape_radius", ""))
        if val not in ("", "inf"):
            esc_by_sample[r["sample_idx"]] = float(val)
    escapes = sorted(esc_by_sample.values())
    n_samples = len({r["sample_idx"] for r in rows})
    n_cert = len({r["sample_idx"] for r in cert_rows})
    return {
        "dataset": name, "n_trees": cfg["n_trees"],
        "n_trees_full": cfg["n_trees_full"], "budget": cfg["budget"],
        "selector": cfg["selector"], "status": "ok",
        "n_samples": n_samples, "n_certified": n_cert,
        "untargeted_min": untargeted[0] if untargeted else "",
        "untargeted_median": median(untargeted) if untargeted else "",
        "untargeted_max": untargeted[-1] if untargeted else "",
        "escape_min": escapes[0] if escapes else "",
        "escape_median": median(escapes) if escapes else "",
        "escape_max": escapes[-1] if escapes else "",
        "runtime_s": round(runtime, 2),
    }


def _worker(conn, name, cfg, out_path, max_samples) -> None:
    # manda al padre solo la riga di riepilogo perche' le righe intere riempivano la pipe e bloccavano il figlio
    import os

    t0 = time.monotonic()
    rows = run_robustness.run(
        name, cfg["n_trees"], cfg["budget"], cfg["selector"], out_path,
        max_samples=max_samples,
    )
    conn.send(_summarize(name, cfg, rows, time.monotonic() - t0))
    conn.close()
    os._exit(0)  # esce di colpo per evitare il rumore del distruttore di dd


def run_one(name, cfg, out_dir, max_samples, timeout_s) -> dict:
    # lancia un dataset in un sottoprocesso e lo uccide se sfora il tempo massimo
    out_path = str(Path(out_dir) / f"robustness_{name}.csv")
    parent, child = mp.Pipe(duplex=False)
    proc = mp.Process(target=_worker,
                      args=(child, name, cfg, out_path, max_samples), daemon=True)
    t0 = time.monotonic()
    proc.start()
    child.close()
    proc.join(timeout_s)
    base = {f: "" for f in SUMMARY_FIELDS}
    base.update(dataset=name, n_trees=cfg["n_trees"],
                n_trees_full=cfg["n_trees_full"], budget=cfg["budget"],
                selector=cfg["selector"], runtime_s=round(time.monotonic() - t0, 2))
    if proc.is_alive():
        proc.terminate()
        proc.join()
        base["status"] = f"timeout {timeout_s:.0f}s"
        return base
    if parent.poll():
        return parent.recv()
    base["status"] = f"crash (exit {proc.exitcode})"
    return base


def main(argv=None) -> None:
    # lancia la robustezza su tutti i modelli e scrive il riepilogo
    p = argparse.ArgumentParser(description="Robustezza sul test set per tutti i 54 modelli, config scelta dal CSV della griglia")
    p.add_argument("--grid", default=str(config.RESULTS_DIR / "grid_campagna.csv"))
    p.add_argument("--out-dir", default=str(config.RESULTS_DIR / "robustness"))
    p.add_argument("--summary", default=str(config.RESULTS_DIR / "robustness_summary.csv"))
    p.add_argument("--max-samples", type=int, default=config.ROBUSTNESS_MAX_SAMPLES,
                   help=f"cap sui sample per test set (default {config.ROBUSTNESS_MAX_SAMPLES}; 0 = tutti)")
    p.add_argument("--timeout", type=float, default=config.ROBUSTNESS_TIMEOUT_S)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args(argv)
    max_samples = None if args.max_samples == 0 else args.max_samples

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    configs = pick_configs(args.grid)
    all_names = forest_io.iter_dataset_names()

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
                row = run_one(name, cfg, args.out_dir, max_samples, args.timeout)
            with lock:
                writer.writerow(row)
                fh.flush()
                print(
                    f"{name}: {row['status']} | k={row['n_trees']} "
                    f"sample={row['n_samples']} cert={row['n_certified']} "
                    f"rob_untargeted min/med/max="
                    f"{row['untargeted_min']}/{row['untargeted_median']}/"
                    f"{row['untargeted_max']} ({row['runtime_s']}s)",
                    file=sys.stderr, flush=True,
                )

        todo = [n for n in all_names if n not in done]
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
