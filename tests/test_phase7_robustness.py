# Test della robustezza MinSwitch

from math import inf

import numpy as np
import pytest

from obddrf import forest_io
from obddrf.bdd_backend import make_bdd
from obddrf.compose import declare_vars
from obddrf.discretize import bits_to_cells, discretize_forest, input_bit_name
from obddrf.frontier import budgeted_forest
from obddrf.robustness import (
    compile_frontier,
    escape_witness_compiled,
    robustness,
    robustness_profile,
    robustness_witness,
    robustness_witness_compiled,
)

from oracle_utils import certified_label, make_dforest


# genera tutti i pattern di bit di input
def all_patterns(dforest):
    n_bits = sum(dforest.bits(f) for f in range(dforest.n_features))
    names = [
        input_bit_name(f, j)
        for f in range(dforest.n_features)
        for j in range(dforest.bits(f))
    ]
    for raw in range(2**n_bits):
        yield {v: bool((raw >> i) & 1) for i, v in enumerate(names)}


# Calcola la robustezza per enumerazione
def brute_rob(dforest, sample_bits, label):
    best = inf
    for bits in all_patterns(dforest):
        if certified_label(dforest, bits_to_cells(dforest, bits)) == label:
            d = sum(1 for v in bits if bits[v] != sample_bits[v])
            best = min(best, d)
    return best


# confronta la robustezza col brute force
def check_against_brute_force(dforest, budgets_criteria, n_samples=None):
    bdd = make_bdd()
    declare_vars(bdd, dforest)
    frontiers = [
        budgeted_forest(bdd, dforest, budget=b, criterion=c, rng=0)
        for b, c in budgets_criteria
    ]
    patterns = list(all_patterns(dforest))
    if n_samples is not None:
        rng = np.random.default_rng(11)
        patterns = [patterns[i] for i in rng.choice(len(patterns), n_samples, replace=False)]
    for sample in patterns:
        own = certified_label(dforest, bits_to_cells(dforest, sample))
        for label in dforest.labels:
            expected = brute_rob(dforest, sample, label)
            for leaves in frontiers:
                got = robustness(bdd, dforest, leaves, sample, label)
                assert got == expected, (sample, label, expected, got)
        if own is not None:
            assert robustness(bdd, dforest, frontiers[0], sample, own) == 0


@pytest.mark.parametrize(
    "specs,labels",
    [
        ([("stump", 0, 0, 1), ("stump", 1, 1, 0), ("const", 0)], [0, 1]),
        ([("const", 0), ("const", 1), ("const", 2)], [0, 1, 2]),  # Win vuoti: +inf
        ([("const", 0), ("const", 0), ("const", 1)], [0, 1, 2]),
    ],
)
# controlla la robustezza esaustiva su foreste sintetiche
def test_synthetic_exhaustive(specs, labels):
    dforest = make_dforest(specs, labels)
    # foglia unica (budget enorme) + frontiera spezzata (budget minimo)
    check_against_brute_force(
        dforest, [(10_000, "modal_var"), (1, "infogain"), (1, "random_node")]
    )


# Una regione vuota ha robustezza infinita
def test_empty_win_region_is_inf():
    dforest = make_dforest([("const", 0), ("const", 1), ("const", 2)], [0, 1, 2])
    bdd = make_bdd()
    declare_vars(bdd, dforest)
    leaves = budgeted_forest(bdd, dforest, budget=10_000, criterion="modal_var")
    sample = next(iter(all_patterns(dforest)))
    assert all(v == inf for v in robustness_profile(bdd, dforest, leaves, sample).values())


# Confronta MinSwitch con brute force su Iris
def test_real_iris_prefix_brute_force():
    rf = forest_io.load_forest("iris")
    dforest = discretize_forest(rf, n_trees=4)
    n_bits = sum(dforest.bits(f) for f in range(dforest.n_features))
    assert n_bits <= 12
    check_against_brute_force(
        dforest,
        [(10_000, "modal_var"), (30, "infogain"), (30, "pvalue")],
        n_samples=40,
    )


# Controlla raggio e witness su un modello reale
def test_witness_matches_minswitch_real_model():
    from obddrf.discretize import cells_to_bits, real_to_cells

    rf = forest_io.load_forest("banknote")  # binario, diagramma non banale
    dforest = discretize_forest(rf, n_trees=5)
    bdd = make_bdd()
    declare_vars(bdd, dforest)
    leaves = budgeted_forest(bdd, dforest, budget=50, criterion="pvalue", rng=0)
    X, _ = forest_io.load_split("banknote", "test")
    for x in X[:30]:
        bits = cells_to_bits(dforest, real_to_cells(dforest, x))
        prof = robustness_profile(bdd, dforest, leaves, bits)
        for l in dforest.labels:
            radius, flips = robustness_witness(bdd, dforest, leaves, bits, l)
            assert radius == prof[l]  # witness == min_switch
            if radius not in (0, inf):
                assert len(flips) == radius
                flipped = {v: (not b if v in flips else b) for v, b in bits.items()}
                assert certified_label(dforest, bits_to_cells(dforest, flipped)) == l


# Calcola per enumerazione la distanza minima da un pattern non certificato come own
def brute_escape(dforest, sample_bits, own):
    best = inf
    for bits in all_patterns(dforest):
        if certified_label(dforest, bits_to_cells(dforest, bits)) != own:
            d = sum(1 for v in bits if bits[v] != sample_bits[v])
            best = min(best, d)
    return best


# confronta la fuga col brute force
def check_escape_against_brute_force(dforest, budgets_criteria, n_samples=None):
    bdd = make_bdd()
    declare_vars(bdd, dforest)
    compiled_all = [
        compile_frontier(
            bdd, dforest,
            budgeted_forest(bdd, dforest, budget=b, criterion=c, rng=0),
        )
        for b, c in budgets_criteria
    ]
    patterns = list(all_patterns(dforest))
    if n_samples is not None:
        rng = np.random.default_rng(7)
        patterns = [patterns[i] for i in rng.choice(len(patterns), n_samples, replace=False)]
    for sample in patterns:
        own = certified_label(dforest, bits_to_cells(dforest, sample))
        if own is None:
            continue
        expected = brute_escape(dforest, sample, own)
        for compiled in compiled_all:
            got, flips = escape_witness_compiled(compiled, sample, own)
            assert got == expected, (sample, own, expected, got)
            assert got >= 1  # un certificato costa sempre almeno un flip da perdere
            rival_min = min(
                robustness_witness_compiled(compiled, sample, l)[0]
                for l in dforest.labels
                if l != own
            )
            assert got <= rival_min  # perdere il certificato non costa piu' che regalarlo a un rivale
            if got != inf:
                assert len(flips) == got
                flipped = {v: (not b if v in flips else b) for v, b in sample.items()}
                assert certified_label(dforest, bits_to_cells(dforest, flipped)) != own


# Confronta la fuga da Win con la ricerca esaustiva su foreste sintetiche
def test_escape_brute_force_synthetic():
    dforest = make_dforest(
        [("stump", 0, 0, 1), ("stump", 1, 1, 0), ("const", 0)], [0, 1]
    )
    check_escape_against_brute_force(
        dforest, [(10_000, "modal_var"), (1, "infogain")]
    )


# Confronta la fuga da Win con la ricerca esaustiva su un prefisso reale di iris
def test_escape_brute_force_iris_prefix():
    rf = forest_io.load_forest("iris")
    dforest = discretize_forest(rf, n_trees=4)
    check_escape_against_brute_force(
        dforest, [(10_000, "modal_var"), (30, "pvalue")], n_samples=25
    )


# Controlla il raggio dei sample reali
def test_real_samples_zero_radius_on_own_class():
    rf = forest_io.load_forest("iris")
    dforest = discretize_forest(rf, n_trees=5)
    bdd = make_bdd()
    declare_vars(bdd, dforest)
    leaves = budgeted_forest(bdd, dforest, budget=40, criterion="infogain", rng=0)
    from obddrf.discretize import cells_to_bits, real_to_cells

    X, _ = forest_io.load_split("iris", "test")
    for x in X:
        bits = cells_to_bits(dforest, real_to_cells(dforest, x))
        own = certified_label(dforest, bits_to_cells(dforest, bits))
        prof = robustness_profile(bdd, dforest, leaves, bits)
        if own is not None:
            assert prof[own] == 0
            assert all(prof[l] > 0 for l in dforest.labels if l != own)
        else:
            assert all(prof[l] > 0 for l in dforest.labels)
