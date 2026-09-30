# Test della discretizzazione e dell'encoding binario

import numpy as np
import pytest

from obddrf import forest_io
from obddrf.discretize import (
    bits_to_cells,
    cell_representative,
    cells_to_bits,
    discretize_forest,
    real_to_cells,
)

DATASETS = ["iris", "banknote", "glass", "diabetes"]


@pytest.fixture(scope="module", params=DATASETS)
# fixture che carica e discretizza ogni dataset di prova
def setup(request):
    name = request.param
    rf = forest_io.load_forest(name)
    dforest = discretize_forest(rf)
    X_tr, _ = forest_io.load_split(name, "train")
    X_te, _ = forest_io.load_split(name, "test")
    X = np.vstack([X_tr, X_te])
    return rf, dforest, X


# controlla le dimensioni di celle e bit
def test_structure(setup):
    rf, dforest, _ = setup
    assert dforest.n_trees == rf.n_estimators
    assert len(dforest.thresholds) == dforest.n_features == rf.n_features_in_
    for f in range(dforest.n_features):
        thr = dforest.thresholds[f]
        # soglie strettamente crescenti (Def. 1)
        assert (np.diff(thr) > 0).all()
        assert dforest.cells(f) == len(thr) + 1
        # b_f = ceil(log2 K): 2^b_f >= K, e il minimo possibile
        K, b = dforest.cells(f), dforest.bits(f)
        assert 2**b >= K and (b == 0 or 2 ** (b - 1) < K)
    # tau dei nodi interni coerente con le soglie globali
    for tn, tau in zip(dforest.trees, dforest.tree_tau):
        internal = tn.children_left != -1
        assert (tau[internal] >= 0).all()
        for i in np.flatnonzero(internal):
            assert dforest.thresholds[tn.feature[i]][tau[i]] == tn.threshold[i]
        assert (tau[~internal] == -1).all()


# controlla che reale a celle a bit e ritorno non perda nulla
def test_oracle_a_roundtrip_lossless(setup):
    _, dforest, X = setup
    for x in X:
        cells = real_to_cells(dforest, x)
        assert bits_to_cells(dforest, cells_to_bits(dforest, cells)) == cells


# controlla che valori della stessa cella diano lo stesso indice
def test_oracle_b_same_cell_same_index(setup):
    _, dforest, _ = setup
    for f in range(dforest.n_features):
        thr = dforest.thresholds[f]
        for cell in range(dforest.cells(f)):
            rep = cell_representative(dforest, f, cell)
            # due valori distinti dentro la stessa cella
            if cell == 0 and len(thr) > 0:
                v1, v2 = rep, float(thr[0])  # estremo incluso (right-closed)
            elif cell == len(thr) and len(thr) > 0:
                v1, v2 = rep, float(thr[-1]) + 2.0
            elif len(thr) > 0:
                lo, hi = float(thr[cell - 1]), float(thr[cell])
                v1, v2 = lo + (hi - lo) / 4, hi  # interno + estremo destro
            else:
                v1, v2 = -1.0, 1.0  # feature mai testata: 1 sola cella
            x1 = np.zeros(dforest.n_features)
            x2 = np.zeros(dforest.n_features)
            x1[f], x2[f] = v1, v2
            assert real_to_cells(dforest, x1)[f] == real_to_cells(dforest, x2)[f] == cell


# controlla il clamping quando le celle non sono potenza di due
def test_oracle_c_clamping_non_power_of_two(setup):
    _, dforest, _ = setup
    checked = 0
    for f in range(dforest.n_features):
        K, b = dforest.cells(f), dforest.bits(f)
        if K == 2**b:
            continue  # nessun pattern di overflow
        checked += 1
        for v in range(K, 2**b):  # tutti i pattern oltre l'ultima cella
            bits = {f"B_{f}_{j}": bool((v >> j) & 1) for j in range(b)}
            # le altre feature a 0
            for g in range(dforest.n_features):
                if g != f:
                    for j in range(dforest.bits(g)):
                        bits[f"B_{g}_{j}"] = False
            assert bits_to_cells(dforest, bits)[f] == K - 1  # Remark 1
    # almeno una feature non potenza di 2 nei dataset scelti
    assert checked > 0


# controlla che un valore sulla soglia vada a sinistra
def test_oracle_d_value_on_threshold_goes_left(setup):
    _, dforest, _ = setup
    for f in range(dforest.n_features):
        for i, ell in enumerate(dforest.thresholds[f]):
            x = np.zeros(dforest.n_features)
            x[f] = ell
            # v = l_i appartiene a (l_{i-1}, l_i] = cella i (test <=)
            assert real_to_cells(dforest, x)[f] == i


# controlla la Prop 1 la foresta classifica uguale sulle celle
def test_oracle_e_prop1_forest_classifies_identically(setup):
    rf, dforest, X = setup
    reps = np.array(
        [
            [
                cell_representative(dforest, f, c)
                for f, c in enumerate(real_to_cells(dforest, x))
            ]
            for x in X
        ]
    )
    assert (rf.predict(reps) == rf.predict(X)).all()


# Il prefisso usa solo le proprie soglie
def test_prefix_forest(setup):
    rf, dforest_full, _ = setup
    dforest5 = discretize_forest(rf, n_trees=5)
    assert dforest5.n_trees == 5 and len(dforest5.trees) == 5
    for f in range(dforest5.n_features):
        assert set(dforest5.thresholds[f]) <= set(dforest_full.thresholds[f])
