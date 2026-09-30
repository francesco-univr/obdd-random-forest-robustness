# FASE 6 sceglie il bit su cui tagliare con i cinque criteri della Def 21
# i tre criteri semantici guardano le soluzioni mentre i due strutturali guardano solo la forma del diagramma
# se non c'e' nulla da misurare si prende il primo bit libero e il ramo collassa comunque

import math

import numpy as np

from .readouts import model_count, pvalues, shapley_values


def _entropy(q: float) -> float:
    # entropia binaria di una probabilita
    if q <= 0.0 or q >= 1.0:
        return 0.0
    return -q * math.log2(q) - (1 - q) * math.log2(1 - q)


def _node_var_counts(bdd, u, free: set[str]) -> dict[str, int]:
    # conta quanti nodi del diagramma testano ogni bit libero tenendo conto degli archi negati
    counts: dict[str, int] = {}
    seen = set()
    stack = [u]
    while stack:
        f = stack.pop()
        if f == bdd.true or f == bdd.false:
            continue
        reg = ~f if getattr(f, "negated", False) else f
        if reg in seen:
            continue
        seen.add(reg)
        if reg.var in free:
            counts[reg.var] = counts.get(reg.var, 0) + 1
        stack.append(reg.low)
        stack.append(reg.high)
    return counts


def select(bdd, acc, free_input_vars: list[str], criterion: str, rng=None, state_vars=()) -> str:
    # restituisce il bit libero su cui fare il taglio secondo il criterio scelto come da Def 19 e 21
    if not free_input_vars:
        raise ValueError("nessun bit di input libero")
    rng = np.random.default_rng(rng)

    if criterion in ("infogain", "shapley", "pvalue"):
        proj = bdd.exist(list(state_vars), acc) if state_vars else acc
        n = len(free_input_vars)
        N = model_count(bdd, proj, n)
        if N == 0:
            return free_input_vars[0]
        if criterion == "infogain":
            best, best_ig = free_input_vars[0], -1.0
            for v in free_input_vars:
                n1 = model_count(bdd, proj & bdd.var(v), n)
                ig = _entropy(n1 / N)
                if ig > best_ig:
                    best, best_ig = v, ig
            return best
        if criterion == "shapley":
            phi = shapley_values(bdd, proj, free_input_vars)
            return min(free_input_vars, key=lambda v: phi[v])  # vince il piu' negativo
        s = pvalues(bdd, proj, free_input_vars)
        return min(free_input_vars, key=lambda v: s[v])

    if criterion in ("random_node", "modal_var"):
        counts = _node_var_counts(bdd, acc, set(free_input_vars))
        if not counts:
            return free_input_vars[0]
        if criterion == "modal_var":
            return max(free_input_vars, key=lambda v: counts.get(v, 0))
        vars_, weights = zip(*counts.items())
        return str(rng.choice(vars_, p=np.array(weights) / sum(weights)))

    raise ValueError(f"criterio sconosciuto: {criterion}")
