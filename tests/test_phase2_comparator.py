# Test del comparatore binario

import numpy as np
import pytest

from obddrf import forest_io
from obddrf.bdd_backend import make_bdd
from obddrf.comparator import node_test
from obddrf.discretize import discretize_forest, input_bit_name


# Confronta nt con X_f <= tau
def enumerate_check(bdd, dforest, f, tau):
    nt = node_test(bdd, dforest, f, tau)
    b = dforest.bits(f)
    for v in range(2**b):
        bits = {input_bit_name(f, j): bool((v >> j) & 1) for j in range(b)}
        val = bdd.let(bits, nt)
        assert val in (bdd.true, bdd.false)
        # l'overflow supera ogni tau valido
        assert (val == bdd.true) == (v <= tau), (f, tau, v)


# Controllo esaustivo su feature sintetiche
def test_exhaustive_synthetic():
    from obddrf.discretize import DiscretizedForest

    for K in range(2, 10):
        dforest = DiscretizedForest(
            n_features=1,
            n_trees=1,
            labels=[0, 1],
            thresholds=[np.arange(K - 1, dtype=float)],
            trees=[],
            tree_tau=[],
        )
        bdd = make_bdd()
        bdd.declare(*dforest.input_bit_names())
        for tau in range(K - 1):
            enumerate_check(bdd, dforest, 0, tau)


# Controllo sulle soglie di Iris
def test_exhaustive_real_forest():
    rf = forest_io.load_forest("iris")
    dforest = discretize_forest(rf)
    bdd = make_bdd()
    bdd.declare(*dforest.input_bit_names())
    pairs = {
        (int(tn.feature[i]), int(tau[i]))
        for tn, tau in zip(dforest.trees, dforest.tree_tau)
        for i in np.flatnonzero(tn.children_left != -1)
    }
    assert pairs
    for f, tau in sorted(pairs):
        enumerate_check(bdd, dforest, f, tau)


# Verifica linearita' e prefisso di zeri
def test_prop2_zero_prefix_support():
    from obddrf.discretize import DiscretizedForest

    K = 16  # b_f = 4
    dforest = DiscretizedForest(
        n_features=1,
        n_trees=1,
        labels=[0, 1],
        thresholds=[np.arange(K - 1, dtype=float)],
        trees=[],
        tree_tau=[],
    )
    bdd = make_bdd()
    bdd.declare(*dforest.input_bit_names())
    b = dforest.bits(0)
    for tau in range(K - 1):
        nt = node_test(bdd, dforest, 0, tau)
        assert bdd.support(nt) <= set(dforest.input_bit_names())
        # dimensione lineare in b_f
        assert len(nt) <= b + 1
    expected = bdd.true
    for j in range(b):
        expected &= ~bdd.var(input_bit_name(0, j))
    assert node_test(bdd, dforest, 0, 0) == expected


# controlla che un tau fuori dominio venga rifiutato
def test_tau_out_of_range_rejected():
    rf = forest_io.load_forest("iris")
    dforest = discretize_forest(rf)
    bdd = make_bdd()
    bdd.declare(*dforest.input_bit_names())
    with pytest.raises(ValueError):
        node_test(bdd, dforest, 0, dforest.cells(0) - 1)  # oltre l'ultima soglia
    with pytest.raises(ValueError):
        node_test(bdd, dforest, 0, -1)
