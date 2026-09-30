# Test dei readout OBDD

import math
from fractions import Fraction

import numpy as np
import pytest

from obddrf import forest_io
from obddrf.bdd_backend import make_bdd
from obddrf.compose import compose_forest, declare_vars, win_regions
from obddrf.discretize import discretize_forest, input_bit_name
from obddrf.readouts import (
    bottom_up_pass,
    conditional_count,
    hypergeom_sf,
    model_count,
    pvalues,
    shapley_values,
    weighted_count,
    zero_stratified_poly,
)

from oracle_utils import certified_label, make_dforest

RNG = np.random.default_rng(7)


@pytest.fixture(scope="module")
# Casi di prova per i readout
def formulas():
    bdd = make_bdd()
    n = 6
    names = [f"x{i}" for i in range(n)]
    bdd.declare(*names)
    exprs = [
        r"x0 /\ ~x3",
        r"(x0 /\ ~x3) \/ (x1 /\ x4 /\ ~x2) \/ (x5 /\ x2)",
        r"(x0 \/ x1) /\ (~x2 \/ x3) /\ (x4 \/ ~x5)",
        r"x1",          # dipende da 1 sola variabile (livelli saltati)
        r"~x5 \/ x5",   # tautologia
    ]
    out = []
    for e in exprs:
        f = bdd.add_expr(e)
        models = [
            tuple(bool((v >> i) & 1) for i in range(n))
            for v in range(2**n)
            if bdd.let({names[i]: bool((v >> i) & 1) for i in range(n)}, f)
            == bdd.true
        ]
        out.append((f, models))
    return bdd, names, out


# Confronta il pass generico con count e brute force
def test_generic_pass_replicates_library_count(formulas):
    bdd, names, cases = formulas
    n = len(names)
    for f, models in cases:
        brute = len(models)
        assert model_count(bdd, f, n) == brute
        generic = bottom_up_pass(
            bdd, f, names,
            weight=lambda i, b: 1,
            add=lambda a, b: a + b,
            mul=lambda a, b: a * b,
            zero=0, one=1,
        )
        assert generic == brute


# Controlla il conteggio condizionato
def test_conditional_count_prop6(formulas):
    bdd, names, cases = formulas
    n = len(names)
    for f, models in cases:
        rho = {"x0": True, "x3": False}
        n_yes = conditional_count(bdd, f, n, rho)
        brute = sum(1 for m in models if m[0] and not m[3])
        assert n_yes == brute
        n_no = model_count(bdd, f, n) - n_yes
        assert n_no == sum(1 for m in models if not (m[0] and not m[3]))


# Controlla il conteggio pesato
def test_weighted_count_prop5(formulas):
    bdd, names, cases = formulas
    for f, models in cases:
        w = {v: (float(RNG.uniform(0, 2)), float(RNG.uniform(0, 2))) for v in names}
        got = weighted_count(bdd, f, names, w)
        brute = sum(
            math.prod(w[v][1] if m[i] else w[v][0] for i, v in enumerate(names))
            for m in models
        )
        assert got == pytest.approx(brute, rel=1e-12)
        half = {v: (0.5, 0.5) for v in names}
        assert weighted_count(bdd, f, names, half) == pytest.approx(
            len(models) / 2 ** len(names), rel=1e-12
        )


# Controlla il polinomio stratificato
def test_zero_stratified_poly_prop8(formulas):
    bdd, names, cases = formulas
    for f, models in cases:
        P = zero_stratified_poly(bdd, f, names)
        for z, coeff in enumerate(P):
            assert coeff == sum(1 for m in models if sum(not b for b in m) == z)
        assert sum(P) == len(models)


# Controlla i valori di Shapley
def test_shapley_prop7(formulas):
    bdd, names, cases = formulas
    n = len(names)
    for f, models in cases:
        phi = shapley_values(bdd, f, names)
        for k, v in enumerate(names):
            brute = -sum(
                Fraction(1, n - sum(m)) for m in models if not m[k]
            )
            assert phi[v] == pytest.approx(float(brute), abs=1e-12)
        all_ones = 1 if tuple([True] * n) in {tuple(m) for m in models} else 0
        assert sum(phi.values()) == pytest.approx(all_ones - len(models), abs=1e-9)


# Controlla i p-value
def test_pvalues_def18(formulas):
    bdd, names, cases = formulas
    n = len(names)
    N, m = 2**n, 2 ** (n - 1)

    # coda ipergeometrica esatta con i binomiali interi
    def exact_sf(a, K):
        if a > min(K, m):
            return 0.0
        tot = sum(
            math.comb(m, i) * math.comb(N - m, K - i)
            for i in range(a, min(K, m) + 1)
        )
        return tot / math.comb(N, K)

    for f, models in cases:
        K = len(models)
        s = pvalues(bdd, f, names)
        for k, v in enumerate(names):
            a1 = sum(1 for mm in models if mm[k])
            expected = min(exact_sf(a1, K), exact_sf(K - a1, K))
            assert s[v] == pytest.approx(expected, rel=1e-9), v


# Controlla il tail per N grandi
def test_hypergeom_sf_large_n():
    p = hypergeom_sf(2**40, N=2**80, K=2**41, m=2**79)
    assert 0.0 <= p <= 1.0
    # caso fortemente arricchito
    tiny = hypergeom_sf(1000, N=2**30, K=1000, m=2**29)
    assert tiny < 1e-250


# conta i modelli di una classe per enumerazione
def count_brute_force(dforest, label):
    n_bits = sum(dforest.bits(f) for f in range(dforest.n_features))
    cnt = 0
    for raw in range(2**n_bits):
        bits, off = {}, 0
        for f in range(dforest.n_features):
            for j in range(dforest.bits(f)):
                bits[input_bit_name(f, j)] = bool((raw >> (off + j)) & 1)
            off += dforest.bits(f)
        from obddrf.discretize import bits_to_cells

        if certified_label(dforest, bits_to_cells(dforest, bits)) == label:
            cnt += 1
    return cnt


@pytest.mark.parametrize(
    "specs,labels",
    [
        ([("stump", 0, 0, 1), ("stump", 1, 1, 0), ("const", 0)], [0, 1]),
        ([("const", 0), ("const", 0), ("const", 1)], [0, 1, 2]),
    ],
)
# confronta il conteggio di Win con l'enumerazione
def test_win_count_synthetic(specs, labels):
    dforest = make_dforest(specs, labels)
    bdd = make_bdd()
    declare_vars(bdd, dforest)
    wins = win_regions(bdd, dforest, compose_forest(bdd, dforest))
    n_bits = sum(dforest.bits(f) for f in range(dforest.n_features))
    for l, w in wins.items():
        assert model_count(bdd, w, n_bits) == count_brute_force(dforest, l)


# Confronta #Win_l con l'enumerazione
def test_win_count_real_small_prefix():
    rf = forest_io.load_forest("iris")
    dforest = discretize_forest(rf, n_trees=2)
    n_bits = sum(dforest.bits(f) for f in range(dforest.n_features))
    assert n_bits <= 20, f"prefisso troppo ricco per il brute force: {n_bits} bit"
    bdd = make_bdd()
    declare_vars(bdd, dforest)
    wins = win_regions(bdd, dforest, compose_forest(bdd, dforest))
    for l, w in wins.items():
        assert model_count(bdd, w, n_bits) == count_brute_force(dforest, l)
