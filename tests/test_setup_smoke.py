# Test di base dell'ambiente

import warnings

import numpy as np
import pytest

from obddrf import bdd_backend, config, forest_io


# controlla le operazioni base del backend
def test_backend_basic_ops():
    bdd = bdd_backend.make_bdd()
    bdd.declare("a", "b")
    f = bdd.add_expr(r"a /\ ~b")
    assert f.count(nvars=2) == 1
    # let/cofactor (restrizione, serve per Prop. 6 e per la frontiera)
    assert bdd.let({"a": True}, f) == bdd.add_expr("~b")
    # quantificazione esistenziale (prodotto relazionale della Fase 4)
    assert bdd.exist({"b"}, f) == bdd.add_expr("a")
    assert bdd_backend.BACKEND in ("cudd", "autoref")


# controlla che i 54 dataset siano presenti
def test_all_54_datasets_present():
    names = forest_io.iter_dataset_names()
    assert len(names) == 54
    for name in names:
        d = config.DATASETS_DIR / name
        assert (d / f"{name}_rf.pkl").exists(), name
        assert (d / f"{name}_test.csv").exists(), name
        assert (d / f"{name}_train.csv").exists(), name


# Carica Iris senza warning di versione
def test_load_iris_bundle_clean():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        bundle = forest_io.load_bundle("iris")
    rf = bundle["model"]
    assert rf.n_estimators == bundle["params"]["n_estimators"] == 26
    assert list(rf.classes_) == [0, 1, 2]
    X, y = forest_io.load_split("iris", "test")
    assert X.shape == (30, 4)
    assert rf.predict(X[:3]).shape == (3,)


# Confronta TreeNodes con sklearn
def test_tree_nodes_walk_matches_sklearn():
    rf = forest_io.load_forest("iris")
    est = rf.estimators_[0]
    tn = forest_io.TreeNodes.from_estimator(est)
    X, _ = forest_io.load_split("iris", "test")
    ours = np.array([tn.predict_cells(x) for x in X])
    theirs = est.predict(X).astype(int)
    assert (ours == theirs).all()


@pytest.mark.parametrize("name", ["banknote", "letter", "shuttle"])
# Carica alcuni modelli aggiuntivi
def test_load_other_bundles(name):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        bundle = forest_io.load_bundle(name)
    rf = bundle["model"]
    assert rf.n_estimators <= 100 and 3 <= rf.max_depth <= 10
