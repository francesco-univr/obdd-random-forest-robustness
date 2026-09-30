# sceglie il motore OBDD della libreria dd e prova prima cudd veloce in C poi ripiega su autoref in puro Python

try:
    from dd import cudd as _impl  # type: ignore

    BACKEND = "cudd"
except ImportError:
    from dd import autoref as _impl

    BACKEND = "autoref"


def make_bdd():
    # crea un manager BDD nuovo e vuoto dove poi si dichiarano le variabili nell'ordine giusto
    return _impl.BDD()
