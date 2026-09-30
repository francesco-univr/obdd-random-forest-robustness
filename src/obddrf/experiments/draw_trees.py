# FASE 8 / Sez. 14: rendering degli alberi sklearn.

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import plot_tree

from .. import config, forest_io


def draw_trees(name: str, n: int, out_dir: Path) -> list[str]:
    # salva come png i primi n alberi del dataset
    rf = forest_io.load_forest(name)
    df = pd.read_csv(config.DATASETS_DIR / name / f"{name}_train.csv")
    feat_names = list(df.columns[:-1])
    class_names = [str(c) for c in rf.classes_]
    written = []
    for i in range(min(n, rf.n_estimators)):
        est = rf.estimators_[i]
        # larghezza proporzionale alla profondita' per restare leggibile
        fig, ax = plt.subplots(figsize=(max(10, 2 ** min(est.get_depth(), 4)), 8))
        plot_tree(est, ax=ax, feature_names=feat_names, class_names=class_names,
                  filled=True, rounded=True, fontsize=8, impurity=False)
        ax.set_title(f"{name} — albero {i} (profondita' {est.get_depth()}, "
                     f"test x[f] <= soglia)")
        out = out_dir / f"tree_{name}_{i}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=120)
        plt.close(fig)
        written.append(out.name)
    return written


def main(argv=None) -> None:
    # disegna i primi alberi di un dataset come immagini
    p = argparse.ArgumentParser(description="Disegna i primi n alberi di un dataset (sklearn plot_tree)")
    p.add_argument("--dataset", default="iris")
    p.add_argument("--n", type=int, default=1, help="quanti alberi disegnare")
    p.add_argument("--out", default=str(config.RESULTS_DIR / "figures"))
    args = p.parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = draw_trees(args.dataset, args.n, out_dir)
    print(f"alberi di {args.dataset} scritti in {out_dir}:")
    for w in written:
        print(" -", w)


if __name__ == "__main__":
    main()
