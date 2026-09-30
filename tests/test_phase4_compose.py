# Test della composizione OBDD

import numpy as np
import pytest

from obddrf import forest_io
from obddrf.bdd_backend import make_bdd
from obddrf.compose import compose_forest, declare_vars, win_regions
from obddrf.discretize import (
    cells_to_bits,
    discretize_forest,
    input_bit_name,
    real_to_cells,
)

from oracle_utils import certified_label, majority_label, make_dforest

RNG = np.random.default_rng(42)


# compone la forest e restituisce bdd e accumulatore
def build(dforest):
    bdd = make_bdd()
    declare_vars(bdd, dforest)
    acc = compose_forest(bdd, dforest)
    wins = win_regions(bdd, dforest, acc)
    return bdd, acc, wins


# Confronta Win_l con l'automa di riferimento
def check_input(bdd, dforest, wins, bits):
    from obddrf.discretize import bits_to_cells

    cells = bits_to_cells(dforest, bits)
    expected = certified_label(dforest, cells)
    got = [l for l, w in wins.items() if bdd.let(bits, w) == bdd.true]
    assert len(got) <= 1
    assert (got[0] if got else None) == expected, (cells, expected, got)


@pytest.mark.parametrize(
    "specs,labels",
    [
        ([("const", 0)], [0, 1]),  # singleton (Remark 3)
        ([("stump", 0, 0, 1), ("const", 1)], [0, 1]),  # pareggio possibile
        ([("stump", 0, 0, 1), ("stump", 1, 1, 0), ("const", 0)], [0, 1]),
        ([("const", 0), ("const", 1), ("const", 2)], [0, 1, 2]),  # 1-1-1: nessuno
        ([("const", 0), ("const", 0), ("const", 1)], [0, 1, 2]),
        # maggioranza 3-2-2 senza clinch
        ([("const", l) for l in [0, 0, 0, 1, 2, 1, 2]], [0, 1, 2]),
        (
            [("stump", 0, 0, 1), ("stump", 1, 2, 0), ("stump", 0, 1, 2), ("const", 0), ("stump", 1, 0, 1)],
            [0, 1, 2],
        ),
    ],
)
# controlla Win_l esaustivo su foreste sintetiche
def test_synthetic_exhaustive(specs, labels):
    dforest = make_dforest(specs, labels)
    bdd, _, wins = build(dforest)
    n_bits = dforest.bits(0)
    for raw in range(2**n_bits):
        bits = {input_bit_name(0, j): bool((raw >> j) & 1) for j in range(n_bits)}
        check_input(bdd, dforest, wins, bits)


# Genera pattern di bit casuali
def random_bit_patterns(dforest, n):
    out = []
    for _ in range(n):
        bits = {}
        for f in range(dforest.n_features):
            for j in range(dforest.bits(f)):
                bits[input_bit_name(f, j)] = bool(RNG.integers(2))
        out.append(bits)
    return out


# restituisce i bit dei campioni reali di un dataset
def real_samples_bits(dforest, name, limit=None):
    X_tr, _ = forest_io.load_split(name, "train")
    X_te, _ = forest_io.load_split(name, "test")
    X = np.vstack([X_tr, X_te])[:limit]
    return [cells_to_bits(dforest, real_to_cells(dforest, x)) for x in X]


@pytest.mark.parametrize("k", [2, 3, 5])
# Controlla prefissi binari di Banknote
def test_real_binary_banknote_prefix(k):
    rf = forest_io.load_forest("banknote")
    dforest = discretize_forest(rf, n_trees=k)
    bdd, _, wins = build(dforest)
    from obddrf.discretize import bits_to_cells

    for bits in random_bit_patterns(dforest, 200) + real_samples_bits(dforest, "banknote", 200):
        check_input(bdd, dforest, wins, bits)
        # completezza binaria
        cells = bits_to_cells(dforest, bits)
        assert certified_label(dforest, cells) == majority_label(dforest, cells)


@pytest.mark.parametrize("k", [2, 3, 5])
# Controlla prefissi multiclasse di Iris
def test_real_multiclass_iris_prefix(k):
    rf = forest_io.load_forest("iris")
    dforest = discretize_forest(rf, n_trees=k)
    bdd, _, wins = build(dforest)
    for bits in random_bit_patterns(dforest, 200) + real_samples_bits(dforest, "iris"):
        check_input(bdd, dforest, wins, bits)


# Le regioni certificate sono disgiunte
def test_win_regions_pairwise_disjoint():
    rf = forest_io.load_forest("iris")
    dforest = discretize_forest(rf, n_trees=5)
    bdd, _, wins = build(dforest)
    labels = list(wins)
    for i, a in enumerate(labels):
        for b in labels[i + 1 :]:
            assert wins[a] & wins[b] == bdd.false
    # solo bit di input
    input_vars = set(dforest.input_bit_names())
    for w in wins.values():
        assert bdd.support(w) <= input_vars
