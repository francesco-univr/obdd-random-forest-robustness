# FASE 1 delle Sez 3 e 4 taglia i valori reali in celle usando le soglie degli alberi e scrive ogni cella in bit

import math
from dataclasses import dataclass

import numpy as np

from .forest_io import TreeNodes


def input_bit_name(f: int, j: int) -> str:
    # nome del bit numero j della feature f dove j uguale a zero e' il bit meno importante
    return f"B_{f}_{j}"


@dataclass
class DiscretizedForest:
    # tiene le soglie di ogni feature e gli alberi riletti

    n_features: int
    n_trees: int
    labels: list[int]
    thresholds: list[np.ndarray]  # soglie in ordine crescente per ogni feature
    trees: list[TreeNodes]
    tree_tau: list[np.ndarray]  # indice della soglia per ogni nodo interno e meno uno sulle foglie

    def cells(self, f: int) -> int:
        # numero di celle della feature cioe' soglie piu' uno come da Def 2
        return len(self.thresholds[f]) + 1

    def bits(self, f: int) -> int:
        # quanti bit servono per scrivere l'indice di cella come da Def 3
        return math.ceil(math.log2(self.cells(f)))

    def input_bit_names(self) -> list[str]:
        # tutti i bit di input nell'ordine del vincolo 5 con il bit piu' grande per primo dentro ogni feature
        return [
            input_bit_name(f, j)
            for f in range(self.n_features)
            for j in reversed(range(self.bits(f)))
        ]


def discretize_forest(rf, n_trees: int | None = None) -> DiscretizedForest:
    # raccoglie le soglie dai primi n alberi le ordina e costruisce la discretizzazione come da Def 1 e 2
    D = rf.n_estimators if n_trees is None else n_trees
    if not 1 <= D <= rf.n_estimators:
        raise ValueError(f"n_trees={n_trees} fuori da [1, {rf.n_estimators}]")
    trees = [TreeNodes.from_estimator(est) for est in rf.estimators_[:D]]
    F = rf.n_features_in_

    per_feature: list[set[float]] = [set() for _ in range(F)]
    for tn in trees:
        internal = tn.children_left != -1
        for f, thr in zip(tn.feature[internal], tn.threshold[internal]):
            per_feature[f].add(float(thr))
    thresholds = [np.array(sorted(s), dtype=float) for s in per_feature]

    tree_tau = []
    for tn in trees:
        tau = np.full(len(tn.feature), -1, dtype=int)
        internal = np.flatnonzero(tn.children_left != -1)
        for i in internal:
            f = tn.feature[i]
            idx = int(np.searchsorted(thresholds[f], tn.threshold[i]))  # ritrova la posizione della soglia nell'array ordinato
            assert thresholds[f][idx] == tn.threshold[i]
            tau[i] = idx
        tree_tau.append(tau)

    return DiscretizedForest(
        n_features=F,
        n_trees=D,
        labels=list(range(len(rf.classes_))),
        thresholds=thresholds,
        trees=trees,
        tree_tau=tree_tau,
    )


def real_to_cells(dforest: DiscretizedForest, x: np.ndarray) -> tuple[int, ...]:
    # trasforma un campione reale negli indici di cella contando le soglie sotto ogni valore come da Def 1
    return tuple(
        int(np.searchsorted(dforest.thresholds[f], x[f], side="left"))
        for f in range(dforest.n_features)
    )


def cells_to_bits(dforest: DiscretizedForest, cells: tuple[int, ...]) -> dict[str, bool]:
    # scrive ogni indice di cella nei suoi bit come da Def 3
    bits: dict[str, bool] = {}
    for f, c in enumerate(cells):
        for j in range(dforest.bits(f)):
            bits[input_bit_name(f, j)] = bool((c >> j) & 1)
    return bits


def bits_to_cells(dforest: DiscretizedForest, bits: dict[str, bool]) -> tuple[int, ...]:
    # rilegge i bit come indici di cella e schiaccia i valori troppo grandi sull'ultima cella come da Remark 1
    cells = []
    for f in range(dforest.n_features):
        v = sum(
            (1 << j) for j in range(dforest.bits(f)) if bits[input_bit_name(f, j)]
        )
        cells.append(min(v, dforest.cells(f) - 1))
    return tuple(cells)


def cell_representative(dforest: DiscretizedForest, f: int, cell: int) -> float:
    # sceglie un valore reale che sta dentro la cella e serve nei test della Prop 1
    thr = dforest.thresholds[f]
    if len(thr) == 0:
        return 0.0
    if cell == 0:
        return float(thr[0]) - 1.0
    if cell == len(thr):
        return float(thr[-1]) + 1.0
    return float(thr[cell - 1] + thr[cell]) / 2.0
