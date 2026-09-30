# FASE 5 delle Sez 9 10 e 11 legge dall'OBDD il conteggio il volume pesato lo Shapley e il p-value
# un solo passaggio dal basso verso l'alto serve per tutte le letture cambiando solo le operazioni usate

from math import lgamma

from .bdd_backend import BACKEND  # noqa: F401 tiene visibile quale motore e' in uso


# passaggio generico


def bottom_up_pass(bdd, u, ordered_vars: list[str], weight, add, mul, zero, one):
    # passata dal basso con memoria che combina i due rami di ogni nodo con le operazioni scelte come da Teorema 1
    n = len(ordered_vars)
    pos = {v: i for i, v in enumerate(ordered_vars)}

    def level(f) -> int:
        # posizione della variabile di un nodo nell'ordine
        if f == bdd.true or f == bdd.false:
            return n  # i terminali stanno sotto tutte le variabili come da Def 16
        return pos[f.var]

    def skip(i: int, j: int):
        # peso delle variabili saltate tra due livelli come da Def 16
        acc = one
        for k in range(i, j):
            acc = mul(acc, add(weight(k, False), weight(k, True)))
        return acc

    memo: dict = {}

    def rec(f):
        # combina i due figli di ogni nodo salendo dal basso
        if f == bdd.false:
            return zero
        if f == bdd.true:
            return one
        if f in memo:
            return memo[f]
        i = level(f)
        var = ordered_vars[i]
        lo = bdd.let({var: False}, f)
        hi = bdd.let({var: True}, f)
        res = add(
            mul(mul(weight(i, False), skip(i + 1, level(lo))), rec(lo)),
            mul(mul(weight(i, True), skip(i + 1, level(hi))), rec(hi)),
        )
        memo[f] = res
        return res

    return mul(skip(0, level(u)), rec(u))


# Teorema 1


def model_count(bdd, u, n_vars: int) -> int:
    # conta gli input che rendono vera la formula usando la primitiva della libreria come da Remark 7
    return int(bdd.count(u, nvars=n_vars))


def conditional_count(bdd, u, n_vars: int, rho: dict[str, bool]) -> int:
    # conta gli input della formula che rispettano anche l'assegnamento parziale come da Prop 6
    cube = bdd.true
    for var, val in rho.items():
        v = bdd.var(var)
        cube &= v if val else ~v
    return model_count(bdd, u & cube, n_vars)


# Prop 5


def weighted_count(bdd, u, ordered_vars: list[str], weights: dict[str, tuple[float, float]]) -> float:
    # somma pesata sugli input veri dove ogni bit porta il suo peso come da Prop 5
    return bottom_up_pass(
        bdd,
        u,
        ordered_vars,
        weight=lambda i, b: weights[ordered_vars[i]][1 if b else 0],
        add=lambda a, b: a + b,
        mul=lambda a, b: a * b,
        zero=0.0,
        one=1.0,
    )


# Prop 7 e 8


def _poly_add(a: list[int], b: list[int]) -> list[int]:
    # somma due polinomi rappresentati come liste di coefficienti
    if len(a) < len(b):
        a, b = b, a
    out = list(a)
    for i, c in enumerate(b):
        out[i] += c
    return out


def _poly_mul(a: list[int], b: list[int]) -> list[int]:
    # moltiplica due polinomi rappresentati come liste di coefficienti
    out = [0] * (len(a) + len(b) - 1)
    for i, ca in enumerate(a):
        if ca:
            for j, cb in enumerate(b):
                out[i + j] += ca * cb
    return out


def zero_stratified_poly(bdd, u, ordered_vars: list[str]) -> list[int]:
    # polinomio che conta i modelli in base a quanti zeri hanno come da Prop 8
    return bottom_up_pass(
        bdd,
        u,
        ordered_vars,
        weight=lambda i, b: [0, 1] if not b else [1],  # il ramo a zero porta un fattore t e il ramo a uno porta un fattore uno
        add=_poly_add,
        mul=_poly_mul,
        zero=[0],
        one=[1],
    )


def shapley_values(bdd, u, ordered_vars: list[str]) -> dict[str, float]:
    # valore di Shapley di ogni bit calcolato con una passata sui polinomi per bit come da Prop 7 e 8
    out = {}
    for var in ordered_vars:
        g = u & ~bdd.var(var)
        P = zero_stratified_poly(bdd, g, ordered_vars)
        out[var] = -sum(c / z for z, c in enumerate(P) if z >= 1 and c)
    return out


# Def 18 e Prop 9


def _log_comb(n: int, k: int) -> float:
    # logaritmo del coefficiente binomiale n su k
    return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)


def hypergeom_sf(a: int, N: int, K: int, m: int) -> float:
    # coda della ipergeometrica in log gamma e con l'approssimazione normale quando i numeri sono enormi come da Prop 9
    import math

    hi = min(K, m)
    if a <= 0:
        return 1.0
    if a > hi:
        return 0.0

    p_in = K / N
    sigma = math.sqrt(m * p_in * (1.0 - p_in) * ((N - m) / (N - 1)))
    if sigma > 2000.0:  # sopra questa soglia il conto esatto e' lento quindi si usa la normale
        z = ((2 * a * N - 2 * K * m) / (2 * N) - 0.5) / sigma  # la differenza va fatta tra interi per non traboccare
        if z > 8.0:  # coda profonda con il rapporto di Mills che conserva l'ordine
            return math.exp(-z * z / 2.0) / (z * math.sqrt(2.0 * math.pi))
        return min(1.0, 0.5 * math.erfc(z / math.sqrt(2.0)))

    log_t = _log_comb(m, a) + _log_comb(N - m, K - a) - _log_comb(N, K)
    total = 1.0
    term = 1.0
    for i in range(a, hi):
        term *= ((m - i) / (i + 1)) * ((K - i) / (N - m - K + i + 1))
        total += term
        if term < 1e-17 * total:
            break
    return min(1.0, math.exp(log_t) * total)


def pvalues(bdd, u, ordered_vars: list[str]) -> dict[str, float]:
    # punteggio statistico di ogni bit come da Def 18 e Prop 9
    n = len(ordered_vars)
    N, m = 2**n, 2 ** (n - 1)
    K = model_count(bdd, u, n)
    out = {}
    for var in ordered_vars:
        a1 = model_count(bdd, u & bdd.var(var), n)
        a0 = K - a1
        a_hi = max(a1, a0)  # il lato con piu' soluzioni da' sempre il p-value piu' piccolo
        out[var] = hypergeom_sf(a_hi, N, K, m)
    return out
