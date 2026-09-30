# Oracoli indipendenti per contatori e composizione

import numpy as np

from obddrf.discretize import DiscretizedForest
from obddrf.forest_io import TreeNodes


# Albero a foglia singola
def const_tree(label: int):
    tn = TreeNodes(
        children_left=np.array([-1]),
        children_right=np.array([-1]),
        feature=np.array([-2]),
        threshold=np.array([-2.0]),
        leaf_label=np.array([label]),
    )
    return tn, np.array([-1])


THRESHOLDS = (0.5, 1.5)  # 3 celle, b_0 = 2 (il pattern 3 clampa sulla cella 2)


# Stump sulla feature 0
def stump(tau_idx: int, thr: float, left_label: int, right_label: int):
    tn = TreeNodes(
        children_left=np.array([1, -1, -1]),
        children_right=np.array([2, -1, -1]),
        feature=np.array([0, -2, -2]),
        threshold=np.array([thr, -2.0, -2.0]),
        leaf_label=np.array([-1, left_label, right_label]),
    )
    return tn, np.array([tau_idx, -1, -1])


# Costruisce una forest sintetica
def make_dforest(specs, labels):
    trees, taus = [], []
    for spec in specs:
        if spec[0] == "const":
            tn, tau = const_tree(spec[1])
        else:
            tn, tau = stump(spec[1], THRESHOLDS[spec[1]], spec[2], spec[3])
        trees.append(tn)
        taus.append(tau)
    return DiscretizedForest(
        n_features=1,
        n_trees=len(trees),
        labels=list(labels),
        thresholds=[np.array(THRESHOLDS)],
        trees=trees,
        tree_tau=taus,
    )


# Restituisce l'etichetta votata
def walk(dforest: DiscretizedForest, t: int, cells) -> int:
    tn, tau = dforest.trees[t], dforest.tree_tau[t]
    i = 0
    while tn.children_left[i] != -1:
        i = int(
            tn.children_left[i]
            if cells[tn.feature[i]] <= tau[i]
            else tn.children_right[i]
        )
    return int(tn.leaf_label[i])


# Esegue l'automa di riferimento
def ref_run(dforest: DiscretizedForest, cells):
    D = dforest.n_trees
    counts = {l: 0 for l in dforest.labels}
    flags = {l: False for l in dforest.labels}
    out = []
    for t in range(D):
        if any(flags.values()):
            out.append(("hold", None, dict(flags)))
            continue
        s = walk(dforest, t, cells)
        R = D - 1 - t
        if all(counts[s] >= counts[l] + R for l in dforest.labels if l != s):
            flags = {l: l == s for l in dforest.labels}
            out.append(("win", None, dict(flags)))
        else:
            counts[s] += 1
            out.append(("inc", dict(counts), dict(flags)))
    return out


# Restituisce l'etichetta certificata
def certified_label(dforest: DiscretizedForest, cells) -> int | None:
    flags = ref_run(dforest, cells)[-1][2]
    winners = [l for l, v in flags.items() if v]
    assert len(winners) <= 1
    return winners[0] if winners else None


# Restituisce la maggioranza stretta
def majority_label(dforest: DiscretizedForest, cells) -> int | None:
    votes = {l: 0 for l in dforest.labels}
    for t in range(dforest.n_trees):
        votes[walk(dforest, t, cells)] += 1
    best = max(votes, key=lambda l: votes[l])
    if votes[best] >= dforest.n_trees // 2 + 1:
        return best
    return None
