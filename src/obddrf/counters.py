# FASE 3 delle Sez 6 e 7 conta i voti degli alberi con contatori in bit e accende il flag di chi ha vinto di sicuro
# i nomi delle variabili seguono INTERFACES e la stessa regola vale per ogni albero come da Remark 4

import numpy as np

from .comparator import node_test
from .discretize import DiscretizedForest


# Def 5


def winning_threshold(n_trees: int) -> int:
    # W e' il numero minimo di voti che basta per vincere di sicuro come da Def 5
    return n_trees // 2 + 1


def counter_msb(n_trees: int) -> int:
    # M e' il bit piu' alto dei contatori e il bit in piu' evita che la somma con R trabocchi come da Sez 7
    if n_trees % 2 == 0:
        return (n_trees // 2 - 1).bit_length() + 1
    return (n_trees - 1).bit_length()


def lc_var(label: int, j: int, nxt: bool = False) -> str:
    # nome del bit j del contatore della classe scelta
    return f"lc_{label}_{j}" + ("_n" if nxt else "")


def win_var(label: int, nxt: bool = False) -> str:
    # nome del flag di vittoria della classe scelta
    return f"w_{label}" + ("_n" if nxt else "")


def counter_bits(bdd, label: int, M: int, nxt: bool = False) -> list:
    # lista delle variabili del contatore dal bit piccolo al bit grande
    return [bdd.var(lc_var(label, j, nxt)) for j in range(M + 1)]


# connettivi di comodo


def _iff(a, b):
    # vero quando i due bit sono uguali
    return (a & b) | (~a & ~b)


def _xor(a, b):
    # vero quando i due bit sono diversi
    return (a & ~b) | (~a & b)


# comparatori sui bit dei contatori


def _le_const(bdd, bits: list, c: int):
    # formula che dice se i bit valgono al massimo c usando la stessa ricorsione della Def 4
    if c >= 2 ** len(bits) - 1:
        return bdd.true
    psi = bdd.true
    for j, bit in enumerate(bits):  # dal bit piccolo al bit grande
        psi = (~bit & psi) if (c >> j) & 1 == 0 else (~bit | psi)
    return psi


def _ge_const(bdd, bits: list, c: int):
    # formula che dice se i bit valgono almeno c cioe' il contrario di valere al massimo c meno uno
    if c <= 0:
        return bdd.true
    if c > 2 ** len(bits) - 1:
        return bdd.false
    return ~_le_const(bdd, bits, c - 1)


def _eq_const(bdd, names: list[str], v: int):
    # formula che fissa ogni bit al valore giusto per dire che la variabile vale v come da Def 5
    out = bdd.true
    for j, name in enumerate(names):
        var = bdd.var(name)
        out &= var if (v >> j) & 1 else ~var
    return out


def _ge_plus_const(bdd, a_bits: list, b_bits: list, R: int):
    # formula che dice se A vale almeno B piu' R costruita senza catena di riporti come da Sez 7
    chi = bdd.true
    for j in range(len(a_bits)):
        if j == 0:
            kappa = bdd.false
        else:
            m = R % (1 << j)
            kappa = bdd.false if m == 0 else _ge_const(bdd, b_bits[:j], (1 << j) - m)
        g = _xor(b_bits[j], kappa)
        if (R >> j) & 1:
            g = ~g
        chi = (a_bits[j] & ~g) | (_iff(a_bits[j], g) & chi)
    return chi


# Def 9


def clinch(bdd, dforest: DiscretizedForest, s: int, t: int):
    # dice se la classe s ha gia' vinto di sicuro anche se tutti i voti rimasti andassero a un rivale come da Def 9
    D = dforest.n_trees
    R = D - 1 - t
    W = winning_threshold(D)
    if R > W - 1:  # prima di meta' alberi nessuno puo' avere gia' vinto quindi la formula e' falsa
        return bdd.false
    M = counter_msb(D)
    a = counter_bits(bdd, s, M)
    out = bdd.true
    for l in dforest.labels:
        if l != s:
            out &= _ge_plus_const(bdd, a, counter_bits(bdd, l, M), R)
    return out


# Def 10


def hold(bdd, dforest: DiscretizedForest):
    # se qualcuno ha gia' vinto i flag restano fermi e i contatori vengono lasciati liberi come da Remark 5
    out = bdd.true
    for l in dforest.labels:
        out &= _iff(bdd.var(win_var(l, nxt=True)), bdd.var(win_var(l)))
    return out


def _win_out(bdd, dforest: DiscretizedForest, s: int):
    # accende il flag della classe s e spegne quelli di tutti gli altri come da Def 10
    out = bdd.var(win_var(s, nxt=True))
    for l in dforest.labels:
        if l != s:
            out &= ~bdd.var(win_var(l, nxt=True))
    return out


def _increment_rel(bdd, label: int, M: int, W: int):
    # aggiunge uno al contatore della classe votata e il tetto W non viene mai davvero toccato come da Remark 4
    cur = counter_bits(bdd, label, M)
    at_cap = _ge_const(bdd, cur, W)
    plus_one = bdd.true
    carry = bdd.true
    for j in range(M + 1):
        nxt_bit = bdd.var(lc_var(label, j, nxt=True))
        plus_one &= _iff(nxt_bit, _xor(cur[j], carry))
        carry = cur[j] & carry
    eq_w = _eq_const(bdd, [lc_var(label, j, nxt=True) for j in range(M + 1)], W)
    return (at_cap & eq_w) | (~at_cap & plus_one)


def _copy_rel(bdd, label: int, M: int):
    # ricopia il contatore senza cambiarlo per le classi che non hanno preso il voto
    out = bdd.true
    for j in range(M + 1):
        out &= _iff(bdd.var(lc_var(label, j, nxt=True)), bdd.var(lc_var(label, j)))
    return out


def _inc_out(bdd, dforest: DiscretizedForest, s: int):
    # il voto aumenta il contatore di s ricopia gli altri e tiene tutti i flag spenti come da Def 10
    D = dforest.n_trees
    M, W = counter_msb(D), winning_threshold(D)
    out = _increment_rel(bdd, s, M, W)
    for l in dforest.labels:
        if l != s:
            out &= _copy_rel(bdd, l, M)
    for l in dforest.labels:
        out &= ~bdd.var(win_var(l, nxt=True))
    return out


def leaf_paths(tn, tau: np.ndarray) -> list[tuple[int, int, list[tuple[int, int, int]]]]:
    # elenca le foglie di un albero con il cammino di test che porta a ciascuna come da Def 6
    out = []

    def rec(i: int, lits: list):
        # scende nell'albero e raccoglie i cammini fino alle foglie
        if tn.children_left[i] == -1:
            out.append((i, int(tn.leaf_label[i]), list(lits)))
            return
        f, th = int(tn.feature[i]), int(tau[i])
        lits.append((f, th, 1))
        rec(int(tn.children_left[i]), lits)
        lits.pop()
        lits.append((f, th, 0))
        rec(int(tn.children_right[i]), lits)
        lits.pop()

    rec(0, [])
    return out


def path_formula(bdd, dforest: DiscretizedForest, lits: list[tuple[int, int, int]]):
    # formula del cammino che unisce i test presi dritti o negati come da Def 6
    out = bdd.true
    for f, tau, d in lits:
        nt = node_test(bdd, dforest, f, tau)
        out &= nt if d == 1 else ~nt
    return out


def step(bdd, dforest: DiscretizedForest, t: int):
    # per ogni foglia il voto o blinda la vittoria o aumenta il contatore come da Def 10
    clinch_cache: dict[int, object] = {}
    win_cache: dict[int, object] = {}
    out = bdd.false
    for _, s, lits in leaf_paths(dforest.trees[t], dforest.tree_tau[t]):
        if s not in clinch_cache:
            clinch_cache[s] = clinch(bdd, dforest, s, t)
            win_cache[s] = (clinch_cache[s] & _win_out(bdd, dforest, s)) | (
                ~clinch_cache[s] & _inc_out(bdd, dforest, s)
            )
        out |= path_formula(bdd, dforest, lits) & win_cache[s]
    return out


def tree_relation(bdd, dforest: DiscretizedForest, t: int):
    # regola completa di un albero che congela tutto se qualcuno ha vinto altrimenti fa votare come da Def 10
    won = bdd.false
    for l in dforest.labels:
        won |= bdd.var(win_var(l))
    return (won & hold(bdd, dforest)) | (~won & step(bdd, dforest, t))


def initial_state(bdd, dforest: DiscretizedForest) -> dict[str, bool]:
    # stato di partenza con tutti i contatori a zero e tutti i flag spenti come da Def 7
    M = counter_msb(dforest.n_trees)
    sub: dict[str, bool] = {}
    for l in dforest.labels:
        for j in range(M + 1):
            sub[lc_var(l, j)] = False
        sub[win_var(l)] = False
    return sub
