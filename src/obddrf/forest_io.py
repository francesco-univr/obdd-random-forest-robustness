# carica i 54 modelli e i loro dataset e i file pkl vanno aperti solo con joblib mai con pickle

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import DATASETS_DIR


def iter_dataset_names(datasets_dir: Path = DATASETS_DIR) -> list[str]:
    # restituisce i nomi dei 54 dataset in ordine alfabetico
    return sorted(p.name for p in datasets_dir.iterdir() if p.is_dir())


def load_bundle(name: str, datasets_dir: Path = DATASETS_DIR) -> dict:
    # apre il file pkl del dataset che contiene il modello i parametri e l'accuratezza salvata
    return joblib.load(datasets_dir / name / f"{name}_rf.pkl")


def load_forest(name: str, datasets_dir: Path = DATASETS_DIR):
    # restituisce solo la random forest presa dal bundle
    return load_bundle(name, datasets_dir)["model"]


def load_split(name: str, split: str, datasets_dir: Path = DATASETS_DIR):
    # legge il csv di train o test e separa i valori dalla colonna delle etichette
    df = pd.read_csv(datasets_dir / name / f"{name}_{split}.csv")
    X = df.iloc[:, :-1].to_numpy(dtype=float)
    y = df.iloc[:, -1].to_numpy()
    return X, y


def label_encoding(name: str, datasets_dir: Path = DATASETS_DIR) -> dict:
    # trova come tradurre le etichette del csv nei numeri del modello e tiene la prima mappa che riproduce l'accuratezza salvata
    bundle = load_bundle(name, datasets_dir)
    rf = bundle["model"]
    X_tr, y_tr = load_split(name, "train", datasets_dir)
    X_te, y_te = load_split(name, "test", datasets_dir)
    classes = sorted(set(y_tr) | set(y_te))
    n = len(rf.classes_)
    if len(classes) != n:
        raise ValueError(f"{name}: {len(classes)} etichette CSV vs {n} classi")

    candidates = [{c: i for i, c in enumerate(classes)}]  # etichette in ordine alfabetico
    all_y = list(y_tr) + list(y_te)
    appearance = list(dict.fromkeys(all_y))
    candidates.append({c: i for i, c in enumerate(appearance)})  # etichette in ordine di prima apparizione
    freq = {c: all_y.count(c) for c in classes}
    by_freq = sorted(classes, key=lambda c: (-freq[c], appearance.index(c)))
    candidates.append({c: i for i, c in enumerate(by_freq)})  # etichette dalla piu' frequente alla piu' rara
    candidates.append({c: i for i, c in enumerate(reversed(classes))})  # etichette in ordine alfabetico rovesciato
    pred_tr = rf.predict(X_tr)
    majority = {
        c: int(np.bincount(pred_tr[y_tr == c], minlength=n).argmax())
        for c in classes
    }
    if sorted(majority.values()) == list(range(n)):
        candidates.append(majority)  # etichette indovinate dai voti del modello sul train

    pred_te = rf.predict(X_te)
    for mapping in candidates:
        acc = float(np.mean(pred_te == np.array([mapping[v] for v in y_te])))
        if abs(acc - bundle["test_acc"]) <= 5e-3:
            return mapping
    raise ValueError(f"{name}: nessun mapping etichette riproduce test_acc={bundle['test_acc']}")


@dataclass
class TreeNodes:
    # vista di un albero sklearn come array affiancati dove il figlio sinistro e' il ramo del test vero

    children_left: np.ndarray
    children_right: np.ndarray
    feature: np.ndarray
    threshold: np.ndarray
    leaf_label: np.ndarray  # vale meno uno sui nodi interni

    @classmethod
    def from_estimator(cls, estimator) -> "TreeNodes":
        # costruisce i nodi piani da un albero di sklearn
        t = estimator.tree_
        is_leaf = t.children_left == -1
        labels = np.argmax(t.value[:, 0, :], axis=1)  # ogni riga di value dice quanti esempi di ogni classe sono finiti nel nodo
        leaf_label = np.where(is_leaf, labels, -1)
        return cls(
            children_left=t.children_left.copy(),
            children_right=t.children_right.copy(),
            feature=t.feature.copy(),
            threshold=t.threshold.copy(),
            leaf_label=leaf_label,
        )

    def predict_cells(self, real_sample: np.ndarray) -> int:
        # cammina l'albero a mano con i test di minore o uguale e restituisce il voto della foglia raggiunta
        i = 0
        while self.children_left[i] != -1:
            if real_sample[self.feature[i]] <= self.threshold[i]:
                i = self.children_left[i]
            else:
                i = self.children_right[i]
        return int(self.leaf_label[i])
