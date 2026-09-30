# Test dell'infrastruttura sperimentale

import csv
from math import inf

import pytest

from obddrf import forest_io
from obddrf.bdd_backend import make_bdd
from obddrf.compose import declare_vars
from obddrf.discretize import bits_to_cells, cells_to_bits, discretize_forest, real_to_cells
from obddrf.experiments import run_grid, run_robustness
from obddrf.experiments.resources import ResourceLimitExceeded, ResourceMonitor
from obddrf.frontier import budgeted_forest
from obddrf.robustness import robustness_profile, robustness_witness

from oracle_utils import certified_label


# Controlla il mapping delle classi
def test_label_encoding_validated_all_54():
    for name in forest_io.iter_dataset_names():
        mapping = forest_io.label_encoding(name)
        rf = forest_io.load_forest(name)
        assert sorted(mapping.values()) == list(range(len(rf.classes_))), name


# controlla che una tripla completi e riporti la riga
def test_run_triple_completes_and_reports():
    rf = forest_io.load_forest("iris")
    row = run_grid.run_triple(rf, "iris", k=5, budget=40, selector="modal_var",
                              timeout_s=120, max_nodes=10**7)
    assert row["completed"] is True
    assert row["n_leaves"] >= 1 and row["peak_nodes"] > 0
    assert row["n_input_bits"] == 11


# Controlla la segnalazione del timeout
def test_run_triple_timeout_reported():
    rf = forest_io.load_forest("iris")
    row = run_grid.run_triple(rf, "iris", k=rf.n_estimators, budget=10**6,
                              selector="modal_var", timeout_s=0.001, max_nodes=10**7)
    assert row["completed"] is False
    assert "timeout" in row["error"]


# controlla il tetto di nodi del monitor
def test_monitor_node_cap():
    monitor = ResourceMonitor(timeout_s=1e9, max_manager_nodes=1)
    bdd = make_bdd()
    bdd.declare("a", "b")
    f = bdd.add_expr(r"a /\ b")
    with pytest.raises(ResourceLimitExceeded):
        monitor.check(bdd, f)


# Controlla griglia progressiva e resume
def test_grid_csv_progressive_and_resume(tmp_path):
    out = tmp_path / "grid.csv"
    argv = [
        "--datasets", "iris", "--budgets", "60", "--selectors", "modal_var",
        "--trees", "3", "5", "--timeout", "120", "--out", str(out),
    ]
    run_grid.main(argv)
    rows = list(csv.DictReader(open(out, encoding="utf-8")))
    ks = [int(r["n_trees"]) for r in rows]
    assert ks == sorted(ks) and ks[0] == 3  # progressivo: 3, 5, ..., 26
    assert all(r["completed"] == "True" for r in rows)
    run_grid.main(argv + ["--resume"])
    assert list(csv.DictReader(open(out, encoding="utf-8"))) == rows


# Controlla l'esecuzione parallela
def test_grid_parallel_jobs(tmp_path):
    out = tmp_path / "grid.csv"
    run_grid.main([
        "--datasets", "iris", "banknote", "--budgets", "60", "--selectors",
        "modal_var", "--trees", "3", "--timeout", "15", "--jobs", "2",
        "--out", str(out),
    ])
    rows = list(csv.DictReader(open(out, encoding="utf-8")))
    by_ds = {r["dataset"] for r in rows}
    assert by_ds == {"iris", "banknote"}
    assert all(r["completed"] == "True" for r in rows if int(r["n_trees"]) == 3)


# Controlla kill duro e ricerca del muro
def test_grid_hard_kill_letter_and_wall_bisection(tmp_path):
    out = tmp_path / "grid.csv"
    run_grid.main([
        "--datasets", "letter", "--budgets", "1000", "--selectors", "modal_var",
        "--trees", "2", "--timeout", "1", "--hard-timeout", "8",
        "--locate-wall", "--out", str(out),
    ])
    rows = list(csv.DictReader(open(out, encoding="utf-8")))
    tried = sorted(int(r["n_trees"]) for r in rows)
    assert tried == [1, 2]  # progressione k=2, poi bisezione (0,2) -> k=1
    assert all(r["completed"] == "False" for r in rows)
    assert all(
        "kill" in r["error"] or "timeout" in r["error"] for r in rows
    )


# Controlla runner e witness di robustezza
def test_robustness_runner_and_witness(tmp_path):
    out = tmp_path / "rob.csv"
    rows = run_robustness.run("iris", k=5, budget=40, selector="infogain",
                              out_path=str(out))
    assert out.exists()
    X, _ = forest_io.load_split("iris", "test")
    assert len(rows) == len(X) * 3  # una riga per (sample, label)

    rf = forest_io.load_forest("iris")
    dforest = discretize_forest(rf, n_trees=5)
    by_sample: dict[int, list[dict]] = {}
    for r in rows:
        by_sample.setdefault(int(r["sample_idx"]), []).append(r)
    for idx, sample_rows in by_sample.items():
        bits = cells_to_bits(dforest, real_to_cells(dforest, X[idx]))
        for r in sample_rows:
            if r["targeted_radius"] == "inf":
                continue
            radius = float(r["targeted_radius"])
            flips = set(r["flipped_bits"].split("|")) if r["flipped_bits"] else set()
            assert len(flips) == radius  # il witness ha esattamente rob switch
            flipped = {v: (not b if v in flips else b) for v, b in bits.items()}
            got = certified_label(dforest, bits_to_cells(dforest, flipped))
            assert got == int(r["label"])  # ... e atterra nella regione Win_l
            if r["certified_class"] != "" and int(r["label"]) == int(r["certified_class"]):
                assert radius == 0


# Controlla il driver batch di robustezza
def test_robustness_batch_picks_config_and_runs(tmp_path):
    from obddrf.experiments import run_robustness_batch as rb

    grid = tmp_path / "grid.csv"
    # configurazioni complete e incomplete
    grid.write_text(
        "dataset,n_trees,n_trees_full,budget,selector,backend,completed,error,"
        "n_input_bits,peak_nodes,n_leaves,runtime_s\n"
        "iris,3,26,100000,modal_var,autoref,True,,7,40,1,0.1\n"
        "iris,5,26,1000,infogain,autoref,True,,11,80,40,0.5\n"
        # n_leaves puo' essere float nel CSV
        "iris,5,26,100000,modal_var,autoref,True,,11,90,3.0,0.2\n"
        "iris,10,26,1000,infogain,autoref,False,timeout,11,,,60\n",
        encoding="utf-8",
    )
    cfgs = rb.pick_configs(str(grid))
    # sceglie k massimo e meno foglie
    assert cfgs["iris"] == {
        "n_trees": 5, "n_leaves": 3, "n_trees_full": 26,
        "budget": 100000, "selector": "modal_var",
    }

    out_dir = tmp_path / "rob"
    summary = tmp_path / "summary.csv"
    rb.main([
        "--grid", str(grid), "--out-dir", str(out_dir),
        "--summary", str(summary), "--max-samples", "10",
        "--timeout", "120", "--jobs", "1",
    ])
    rows = {r["dataset"]: r for r in csv.DictReader(open(summary, encoding="utf-8"))}
    assert rows["iris"]["status"] == "ok"
    assert int(rows["iris"]["n_samples"]) == 10
    assert int(rows["iris"]["n_certified"]) >= 1
    # nessuna configurazione completata
    assert rows["letter"]["status"] == "non compilabile"


# Controlla pareggi fra rivali non certificati e mediana statistica
def test_robustness_summary_counts_each_sample_once():
    from obddrf.experiments import run_robustness_batch as rb

    cfg = {
        "n_trees": 3, "n_trees_full": 3,
        "budget": 100, "selector": "modal_var",
    }
    rows = [
        {"sample_idx": 0, "certified_class": 0, "label": 0,
         "targeted_radius": 0, "is_untargeted_min": 0, "escape_radius": 1},
        {"sample_idx": 0, "certified_class": 0, "label": 1,
         "targeted_radius": 2, "is_untargeted_min": 1, "escape_radius": 1},
        {"sample_idx": 0, "certified_class": 0, "label": 2,
         "targeted_radius": 2, "is_untargeted_min": 1, "escape_radius": 1},
        {"sample_idx": 1, "certified_class": 0, "label": 0,
         "targeted_radius": 0, "is_untargeted_min": 0, "escape_radius": 2},
        {"sample_idx": 1, "certified_class": 0, "label": 1,
         "targeted_radius": 4, "is_untargeted_min": 1, "escape_radius": 2},
        {"sample_idx": 1, "certified_class": 0, "label": 2,
         "targeted_radius": 5, "is_untargeted_min": 0, "escape_radius": 2},
        {"sample_idx": 2, "certified_class": "", "label": 0,
         "targeted_radius": 1, "is_untargeted_min": 1, "escape_radius": ""},
    ]

    got = rb._summarize("toy", cfg, rows, 0.1)
    assert got["n_samples"] == 3
    assert got["n_certified"] == 2
    assert (got["untargeted_min"], got["untargeted_median"],
            got["untargeted_max"]) == (2.0, 3.0, 4.0)
    assert (got["escape_min"], got["escape_median"],
            got["escape_max"]) == (1.0, 1.5, 2.0)


# Controlla i readout sperimentali
def test_readouts_deliverable4(tmp_path):
    from obddrf.experiments.run_readouts import compute_readouts

    cfg = {"n_trees": 4, "n_leaves": 1, "n_trees_full": 26,
           "budget": 100000, "selector": "modal_var"}
    r = compute_readouts("iris", cfg, top=5, max_bits_shapley=60, out_dir=tmp_path)
    assert (tmp_path / "iris.csv").exists()
    assert r["status"] == "ok"
    n = r["n_input_bits"]
    assert r["certified_total"] == pytest.approx(r["certified_frac"] * 2**n, rel=1e-6)
    assert len(r["top_infogain"].split("|")) == 5
    assert len(r["top_pvalue"].split("|")) == 5
    assert len(r["top_shapley"].split("|")) == 5


# Controlla il renderer OBDD
def test_draw_obdd_smoke(tmp_path):
    import numpy as np

    from obddrf.bdd_backend import make_bdd
    from obddrf.comparator import node_test
    from obddrf.discretize import DiscretizedForest
    from obddrf.experiments.draw_obdd import draw

    df = DiscretizedForest(1, 1, [0, 1], [np.arange(5, dtype=float)], [], [])
    bdd = make_bdd()
    bdd.declare(*df.input_bit_names())
    nt = node_test(bdd, df, 0, 2)
    out = tmp_path / "obdd.png"
    draw(bdd, nt, df.input_bit_names(), "test", out)
    assert out.exists() and out.stat().st_size > 0


# Controlla l'export OBDD interattivo
def test_obdd_interactive_export(tmp_path):
    from obddrf.experiments.obdd_interactive import export_json, write_html

    data = export_json("iris", k=3, label=1)
    assert data["nodes"] and data["edges"] and data["root"]
    ids = {n["id"] for n in data["nodes"]}
    for e in data["edges"]:  # ogni arco punta a un nodo o a un terminale
        assert e["src"] in ids
        assert e["dst"] in ids or e["dst"] in ("T", "F")
    for n in data["nodes"]:  # ogni nodo decodifica feature + bit
        assert 0 <= n["feature"] < len(data["feature_names"])
        assert n["bit"] >= 0
    out = tmp_path / "v.html"
    write_html(data, out)
    html = out.read_text(encoding="utf-8")
    assert "<svg" in html and data["root"] in html  # dati inline


# Controlla la scelta automatica dell'OBDD
def test_obdd_pick_best():
    from obddrf.experiments.obdd_interactive import pick_best

    k, label, n = pick_best("iris", max_nodes=50)
    assert 1 <= n <= 50
    assert pick_best("pima", max_nodes=50) is None  # 10818 nodi: scartato


# Controlla la tabella dei dataset
def test_datasets_overview():
    from obddrf.experiments.datasets_overview import describe

    d = describe("iris")
    assert d["n_classes"] == 3 and d["n_features"] == 4
    assert d["n_input_bits"] > 0 and d["n_trees"] == 26


# Confronta witness e profilo di robustezza
def test_witness_profile_consistency():
    rf = forest_io.load_forest("banknote")
    dforest = discretize_forest(rf, n_trees=5)
    bdd = make_bdd()
    declare_vars(bdd, dforest)
    leaves = budgeted_forest(bdd, dforest, budget=50, criterion="pvalue", rng=0)
    X, _ = forest_io.load_split("banknote", "test")
    for x in X[:25]:
        bits = cells_to_bits(dforest, real_to_cells(dforest, x))
        prof = robustness_profile(bdd, dforest, leaves, bits)
        for label in dforest.labels:
            r, flips = robustness_witness(bdd, dforest, leaves, bits, label)
            assert r == prof[label]
            if r not in (0, inf):
                assert 0 < len(flips) == r
