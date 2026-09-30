# FASE 4 della Sez 8 incolla gli alberi uno dopo l'altro in un unico OBDD tenendo vive solo due copie di stato
# alla fine Win_l contiene tutti gli input che la forest certifica per la classe l

from .counters import (
    counter_msb,
    initial_state,
    lc_var,
    tree_relation,
    win_var,
)


def declare_vars(bdd, dforest) -> None:
    # dichiara le variabili con i bit di input in cima e ogni bit di stato accanto al suo gemello next come da vincolo 5
    names = list(dforest.input_bit_names())
    M = counter_msb(dforest.n_trees)
    for l in dforest.labels:
        for j in range(M + 1):
            names.append(lc_var(l, j))
            names.append(lc_var(l, j, nxt=True))
    for l in dforest.labels:
        names.append(win_var(l))
        names.append(win_var(l, nxt=True))
    bdd.declare(*names)


def state_var_names(dforest, nxt: bool = False) -> list[str]:
    # elenca le variabili di una copia di stato cioe' contatori e flag come da Def 12
    M = counter_msb(dforest.n_trees)
    return [
        lc_var(l, j, nxt) for l in dforest.labels for j in range(M + 1)
    ] + [win_var(l, nxt) for l in dforest.labels]


def rename_next_to_current(dforest) -> dict[str, str]:
    # dizionario che rinomina ogni variabile next nella sua versione corrente come da Def 13
    return dict(zip(state_var_names(dforest, nxt=True), state_var_names(dforest)))


def _and_exists(bdd, u, v, qvars):
    # fa la congiunzione e toglie le variabili di stato in un colpo solo quando il motore lo sa fare
    if hasattr(bdd, "and_exists"):
        return bdd.and_exists(u, v, qvars)
    return bdd.exist(qvars, u & v)


def compose_forest(bdd, dforest, on_step=None):
    # costruisce l'accumulatore finale aggiungendo un albero alla volta come da Def 14
    cur = state_var_names(dforest)
    ren = rename_next_to_current(dforest)

    phi0 = bdd.let(initial_state(bdd, dforest), tree_relation(bdd, dforest, 0))  # primo passo dallo stato tutto a zero come da Remark 4
    acc = bdd.let(ren, phi0)
    if on_step is not None:
        on_step(0, acc)

    for t in range(1, dforest.n_trees):
        phi_t = tree_relation(bdd, dforest, t)
        acc = bdd.let(ren, _and_exists(bdd, acc, phi_t, cur))
        if on_step is not None:
            on_step(t, acc)
    return acc


def win_regions(bdd, dforest, acc) -> dict[int, object]:
    # estrae per ogni classe la regione degli input certificati come da Def 14
    cur = state_var_names(dforest)
    return {
        l: _and_exists(bdd, acc, bdd.var(win_var(l)), cur) for l in dforest.labels
    }
