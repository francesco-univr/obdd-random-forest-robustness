# Test dei contatori e dell'automa del clinch

import pytest

from obddrf.bdd_backend import make_bdd
from obddrf.compose import declare_vars
from obddrf.counters import (
    _eq_const,
    _ge_plus_const,
    clinch,
    counter_msb,
    initial_state,
    lc_var,
    tree_relation,
    win_var,
    winning_threshold,
)
from obddrf.discretize import bits_to_cells, input_bit_name

from oracle_utils import make_dforest, ref_run


# calcola lo stato del contatore atteso dopo un albero
def expected_next(bdd, dforest, branch, counts, flags):
    M = counter_msb(dforest.n_trees)
    out = bdd.true
    if branch == "inc":  # contatori next fissati
        for l in dforest.labels:
            out &= _eq_const(
                bdd, [lc_var(l, j, nxt=True) for j in range(M + 1)], counts[l]
            )
    # win/hold lasciano liberi i contatori
    for l in dforest.labels:
        v = bdd.var(win_var(l, nxt=True))
        out &= v if flags[l] else ~v
    return out


# Confronta le relazioni con il simulatore
def check_forest(dforest):
    bdd = make_bdd()
    declare_vars(bdd, dforest)
    D = dforest.n_trees
    M = counter_msb(D)
    rels = [tree_relation(bdd, dforest, t) for t in range(D)]
    n_bits = sum(dforest.bits(f) for f in range(dforest.n_features))

    for raw in range(2**n_bits):
        bits = {input_bit_name(0, j): bool((raw >> j) & 1) for j in range(n_bits)}
        cells = bits_to_cells(dforest, bits)
        steps = ref_run(dforest, cells)

        cur = initial_state(bdd, dforest)
        counts = {l: 0 for l in dforest.labels}
        for t in range(D):
            branch, ref_counts, ref_flags = steps[t]
            sub = bdd.let({**bits, **cur}, rels[t])
            assert sub == expected_next(bdd, dforest, branch, ref_counts, ref_flags), (
                raw,
                t,
                branch,
            )
            # i contatori non servono dopo win/hold
            if branch == "inc":
                counts = ref_counts
            cur = {}
            for l in dforest.labels:
                for j in range(M + 1):
                    cur[lc_var(l, j)] = bool((counts[l] >> j) & 1)
                cur[win_var(l)] = ref_flags[l]
        # al piu' un vincitore
        assert sum(steps[-1][2].values()) <= 1


# controlla le costanti W e M della Def 5
def test_def5_constants():
    # il bit extra copre LC + R_t
    for D in range(1, 33):
        W, M = winning_threshold(D), counter_msb(D)
        assert W == D // 2 + 1
        assert 2 ** (M + 1) > W  # larghezza M+1 rappresenta 0..W
        assert 2 * (W - 1) < 2 ** (M + 1)  # niente overflow nel comparatore offset


# Controllo esaustivo di A >= B + R
def test_ge_plus_const_exhaustive():
    bdd = make_bdd()
    width = 3
    a_names = [f"a{j}" for j in range(width)]
    b_names = [f"b{j}" for j in range(width)]
    bdd.declare(*a_names, *b_names)
    a_bits = [bdd.var(n) for n in a_names]
    b_bits = [bdd.var(n) for n in b_names]
    for R in range(8):
        f = _ge_plus_const(bdd, a_bits, b_bits, R)
        for A in range(8):
            for B in range(8):
                if B + R >= 8:
                    continue  # fuori parola: mai usato (don't care)
                sub = {n: bool((A >> j) & 1) for j, n in enumerate(a_names)}
                sub |= {n: bool((B >> j) & 1) for j, n in enumerate(b_names)}
                assert (bdd.let(sub, f) == bdd.true) == (A >= B + R), (A, B, R)


# Clinch e' falso prima di meta' forest
def test_clinch_short_circuit():
    dforest = make_dforest([("const", 0)] * 7, labels=[0, 1, 2])
    bdd = make_bdd()
    declare_vars(bdd, dforest)
    for t in range(7):
        c = clinch(bdd, dforest, 0, t)
        if t < 7 // 2:
            assert c == bdd.false, t
    assert clinch(bdd, dforest, 0, 7 // 2) != bdd.false


# Controlla Clinch nel caso multiclasse
def test_clinch_raw_semantics_multiclass():
    dforest = make_dforest([("const", 0)] * 5, labels=[0, 1, 2])
    D, W, M = 5, winning_threshold(5), counter_msb(5)
    bdd = make_bdd()
    declare_vars(bdd, dforest)
    for t in range(D // 2, D):
        R = D - 1 - t
        c = clinch(bdd, dforest, 0, t)
        for v0 in range(W):
            for v1 in range(W):
                for v2 in range(W):
                    sub = {}
                    for l, v in zip([0, 1, 2], [v0, v1, v2]):
                        for j in range(M + 1):
                            sub[lc_var(l, j)] = bool((v >> j) & 1)
                    expected = v0 >= v1 + R and v0 >= v2 + R
                    assert (bdd.let(sub, c) == bdd.true) == expected, (t, v0, v1, v2)


@pytest.mark.parametrize("D", [2, 3, 4, 5, 6])
# Controlla l'equivalenza binaria di Clinch
def test_clinch_binary_equivalence(D):
    dforest = make_dforest([("const", 0)] * D, labels=[0, 1])
    W, M = winning_threshold(D), counter_msb(D)
    bdd = make_bdd()
    declare_vars(bdd, dforest)
    for t in range(D):
        c = clinch(bdd, dforest, 0, t)
        for v_s in range(min(t, W - 1) + 1):
            v_l = t - v_s
            if v_l > W - 1:
                continue
            sub = {}
            for l, v in zip([0, 1], [v_s, v_l]):
                for j in range(M + 1):
                    sub[lc_var(l, j)] = bool((v >> j) & 1)
            assert (bdd.let(sub, c) == bdd.true) == (v_s + 1 >= W), (t, v_s, v_l)


# Una forest singleton certifica subito
def test_phi0_singleton_d1():
    check_forest(make_dforest([("const", 0)], labels=[0, 1]))
    check_forest(make_dforest([("stump", 0, 1, 0)], labels=[0, 1]))


# Controlla l'inizializzazione del primo albero
def test_phi0_initialization_first_tree():
    check_forest(make_dforest([("const", 1), ("const", 1)], labels=[0, 1]))
    check_forest(make_dforest([("stump", 0, 0, 1), ("const", 1)], labels=[0, 1]))


# Controlla sequenze di voto binarie
def test_vote_evolution_binary():
    # 2 voti uguali -> vince al secondo albero
    check_forest(make_dforest([("const", 1), ("const", 1)], labels=[0, 1]))
    # pareggio 1-1 -> nessun flag
    check_forest(make_dforest([("const", 0), ("const", 1)], labels=[0, 1]))
    # input-dependent: maggioranza 2-1 variabile
    check_forest(
        make_dforest(
            [("stump", 0, 0, 1), ("stump", 1, 1, 0), ("const", 0)], labels=[0, 1]
        )
    )
    check_forest(
        make_dforest(
            [("stump", 0, 1, 0), ("stump", 0, 1, 0), ("stump", 1, 0, 1)],
            labels=[0, 1],
        )
    )


# Controlla sequenze di voto multiclasse
def test_vote_evolution_multiclass():
    check_forest(make_dforest([("const", 0), ("const", 0), ("const", 1)], labels=[0, 1, 2]))
    # split 1-1-1: argmax esiste (tie-break) ma nessun clinch
    dforest = make_dforest([("const", 0), ("const", 1), ("const", 2)], labels=[0, 1, 2])
    check_forest(dforest)
    steps = ref_run(dforest, (0,))
    assert not any(steps[-1][2].values())  # nessuna etichetta certificata
    # D=5, voti A,B,A,C,A: A clincha all'ultimo voto (2 >= 1+0 su entrambi)
    check_forest(
        make_dforest(
            [("const", 0), ("const", 1), ("const", 0), ("const", 2), ("const", 0)],
            labels=[0, 1, 2],
        )
    )
    # maggioranza 3-2-2 senza clinch
    dforest = make_dforest(
        [("const", l) for l in [0, 0, 0, 1, 2, 1, 2]], labels=[0, 1, 2]
    )
    check_forest(dforest)
    assert not any(ref_run(dforest, (0,))[-1][2].values())


# Controlla contatori piu' larghi
def test_vote_evolution_d4_wider_counters():
    check_forest(
        make_dforest(
            [("const", 1), ("const", 1), ("const", 1), ("const", 1)], labels=[0, 1]
        )
    )
    check_forest(
        make_dforest(
            [("stump", 0, 0, 1), ("const", 1), ("stump", 1, 1, 0), ("const", 0)],
            labels=[0, 1],
        )
    )
