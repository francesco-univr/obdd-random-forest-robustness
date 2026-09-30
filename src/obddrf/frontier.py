# FASE 6 della Sez 12 fa crescere l'OBDD entro un tetto di nodi e quando sfora taglia lo spazio in due su un bit
# gli alberi che restano vanno sempre ristretti alla fetta corrente altrimenti il conteggio sbaglia come da Prop 10
# alla fine le fette coprono tutto lo spazio senza sovrapporsi e le letture si sommano sulle fette

from dataclasses import dataclass

import numpy as np

from .compose import _and_exists, rename_next_to_current, state_var_names
from .counters import initial_state, tree_relation
from .discretize import DiscretizedForest
from .selectors import select


@dataclass
class FrontierLeaf:
    # una fetta della frontiera con il suo assegnamento il suo OBDD e il prossimo albero da aggiungere come da Def 19

    rho: dict[str, bool]
    acc: object
    t: int


def budgeted_forest(
    bdd,
    dforest: DiscretizedForest,
    budget: int,
    criterion: str,
    rng=None,
    on_step=None,
) -> list[FrontierLeaf]:
    # algoritmo BudgetedForest della Sez 12 con un callback opzionale per registrare il picco di nodi
    rng = np.random.default_rng(rng)
    D = dforest.n_trees
    cur = state_var_names(dforest)
    ren = rename_next_to_current(dforest)
    input_vars = dforest.input_bit_names()

    phis: dict[int, object] = {}

    def phi(t: int):
        # restituisce la relazione dell'albero t calcolandola una volta sola
        if t not in phis:
            phis[t] = tree_relation(bdd, dforest, t)
        return phis[t]

    acc0 = bdd.let(ren, bdd.let(initial_state(bdd, dforest), phi(0)))  # primo passo dallo stato tutto a zero come da Remark 4
    if on_step is not None:
        on_step(0, acc0)

    frontier: list[FrontierLeaf] = [FrontierLeaf(rho={}, acc=acc0, t=1)]
    final: list[FrontierLeaf] = []
    while frontier:
        leaf = frontier.pop()
        rho, acc, t = leaf.rho, leaf.acc, leaf.t
        while t < D and len(acc) <= budget:  # fa crescere la fetta finche' sta nel tetto
            phi_rho = bdd.let(rho, phi(t)) if rho else phi(t)  # albero ristretto alla fetta
            acc = bdd.let(ren, _and_exists(bdd, acc, phi_rho, cur))
            t += 1
            if on_step is not None:
                on_step(t - 1, acc)
        if t == D:
            final.append(FrontierLeaf(rho=dict(rho), acc=acc, t=t))
            continue
        free = [v for v in input_vars if v not in rho]  # il tetto e' sforato e serve un nuovo taglio come da Def 19
        if not free:
            # tutti i bit sono gia' fissati quindi si finisce comunque perche' l'accumulatore collassa come da Prop 10
            while t < D:
                acc = bdd.let(ren, _and_exists(bdd, acc, bdd.let(rho, phi(t)), cur))
                t += 1
                if on_step is not None:
                    on_step(t - 1, acc)
            final.append(FrontierLeaf(rho=dict(rho), acc=acc, t=t))
            continue
        x = select(bdd, acc, free, criterion, rng=rng, state_vars=cur)
        for val in (True, False):
            frontier.append(
                FrontierLeaf(
                    rho={**rho, x: val}, acc=bdd.let({x: val}, acc), t=t
                )
            )
    return final
