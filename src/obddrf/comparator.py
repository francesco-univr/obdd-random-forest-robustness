# FASE 2 della Sez 5 costruisce la formula che dice se l'indice di cella sta sotto o uguale alla soglia
# la ricorsione psi della Def 4 guarda un bit alla volta e la formula resta piccola come dice la Prop 2

from .discretize import DiscretizedForest, input_bit_name


def node_test(bdd, dforest: DiscretizedForest, f: int, tau: int):
    # formula nt della Def 4 sui bit della feature f dove tau e' l'indice della soglia da confrontare
    if not 0 <= tau <= dforest.cells(f) - 2:
        raise ValueError(f"tau={tau} fuori da [0, cells({f})-2={dforest.cells(f) - 2}]")
    psi = bdd.true  # caso base della ricorsione
    for j in range(dforest.bits(f)):  # dal bit meno importante al piu' importante
        not_bit = ~bdd.var(input_bit_name(f, j))
        if (tau >> j) & 1 == 0:
            psi = not_bit & psi
        else:
            psi = not_bit | psi
    return psi
