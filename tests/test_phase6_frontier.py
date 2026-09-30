# Test della frontiera con budget

import numpy as np
import pytest

from obddrf import forest_io
from obddrf.bdd_backend import make_bdd
from obddrf.compose import (
    compose_forest,
    declare_vars,
    state_var_names,
    win_regions,
)
from obddrf.counters import win_var
from obddrf.discretize import bits_to_cells, discretize_forest
from obddrf.frontier import budgeted_forest
from obddrf.readouts import model_count
from obddrf.selectors import select

from oracle_utils import certified_label

SELECTORS = ["infogain", "shapley", "pvalue", "random_node", "modal_var"]


@pytest.fixture(scope="module")
# fixture con un prefisso di cinque alberi di iris
def iris5():
    rf = forest_io.load_forest("iris")
    dforest = discretize_forest(rf, n_trees=5)
    bdd = make_bdd()
    declare_vars(bdd, dforest)
    acc = compose_forest(bdd, dforest)
    wins = win_regions(bdd, dforest, acc)
    n_bits = sum(dforest.bits(f) for f in range(dforest.n_features))
    totals = {l: model_count(bdd, w, n_bits) for l, w in wins.items()}
    return bdd, dforest, acc, totals, n_bits


# costruisce il bdd di un cubo da un assegnamento
def cube_bdd(bdd, rho):
    out = bdd.true
    for var, val in rho.items():
        v = bdd.var(var)
        out &= v if val else ~v
    return out


@pytest.mark.parametrize("criterion", SELECTORS)
# Controlla partizione e conteggi additivi
def test_partition_and_additive_counts(iris5, criterion):
    bdd, dforest, _, totals, n_bits = iris5
    leaves = budgeted_forest(bdd, dforest, budget=35, criterion=criterion, rng=0)
    assert len(leaves) > 1  # il budget ha effettivamente forzato il case-split
    assert all(leaf.t == dforest.n_trees for leaf in leaves)

    # partizione dei cubi
    union = bdd.false
    for leaf in leaves:
        c = cube_bdd(bdd, leaf.rho)
        assert c & union == bdd.false
        union |= c
    assert union == bdd.true

    # ricomposizione dei conteggi
    state = state_var_names(dforest)
    for l in dforest.labels:
        tot = 0
        for leaf in leaves:
            w_rho = bdd.exist(state, leaf.acc & bdd.var(win_var(l)))
            tot += model_count(bdd, w_rho, n_bits - len(leaf.rho))
        assert tot == totals[l], (criterion, l)


# Un budget ampio non produce split
def test_unlimited_budget_single_leaf(iris5):
    bdd, dforest, acc, _, _ = iris5
    leaves = budgeted_forest(bdd, dforest, budget=10_000, criterion="modal_var")
    assert len(leaves) == 1
    assert leaves[0].rho == {}
    assert leaves[0].acc == acc


# Controlla la foglia associata a ogni input
def test_leaf_membership_consistency(iris5):
    bdd, dforest, _, _, n_bits = iris5
    leaves = budgeted_forest(bdd, dforest, budget=35, criterion="infogain", rng=1)
    state = state_var_names(dforest)
    rng = np.random.default_rng(3)
    names = dforest.input_bit_names()
    for _ in range(100):
        bits = {v: bool(rng.integers(2)) for v in names}
        owners = [lf for lf in leaves if all(bits[k] == v for k, v in lf.rho.items())]
        assert len(owners) == 1  # partizione
        leaf = owners[0]
        free_bits = {k: v for k, v in bits.items() if k not in leaf.rho}
        got = [
            l
            for l in dforest.labels
            if bdd.let(free_bits, bdd.exist(state, leaf.acc & bdd.var(win_var(l))))
            == bdd.true
        ]
        expected = certified_label(dforest, bits_to_cells(dforest, bits))
        assert (got[0] if got else None) == expected


# Controlla i selettori della frontiera
def test_selector_semantics_directly():
    bdd = make_bdd()
    bdd.declare("x0", "x1", "x2")
    # x0 divide i modelli a meta'
    f = bdd.add_expr(r"(x0 /\ x1) \/ (~x0 /\ x2)")
    free = ["x0", "x1", "x2"]
    assert select(bdd, f, free, "infogain") == "x0"
    assert select(bdd, f, free, "modal_var") == "x0"  # 1 nodo ciascuna: tie
    assert select(bdd, f, free, "random_node", rng=0) in free
    # gli altri selettori devono restituire un bit libero
    assert select(bdd, f, free, "shapley") in free
    assert select(bdd, f, free, "pvalue") in free
