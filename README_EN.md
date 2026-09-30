# Robustness of a Random-Forest Classifier via OBDD

BDSS 2025/26 project, University of Verona. The full report is in `report.pdf`.
An Italian version of this README is in `README.md`.

## Layout

- `src/obddrf/` holds the pipeline with one module per step of the assignment
- `src/obddrf/experiments/` holds the campaign, the analysis and the figures
- `tests/` holds 110 tests with one oracle file per phase
- `results/` holds the study CSVs and figures
- `report.pdf` is the full report
- `INTERFACES.md` documents the data structures shared between modules

The commands below read the 54 datasets from `datasets/datasets/<name>/`,
already included in the archive next to `src/`. They are the same ones handed
out in class, with train and test csv and the pkl model for each dataset.

## Environment

The project uses Python 3.12 and the dependencies listed in `requirements.txt`.
Scikit-learn stays pinned to 1.4.2 with numpy 1.x to keep compatibility with
the stored models.

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\pip install -e .
```

The second command registers `obddrf` in the venv. The backend uses the `dd`
library. The code tries `dd.cudd` and falls back automatically to `dd.autoref`
when CUDD is not available.

From here on every command runs from the project root with the venv Python. On
Windows the prefix is `.venv\Scripts\python`, on Linux or Mac it is
`.venv/bin/python`.

## Quick start

To see the project working on a single model, without rerunning the whole
campaign, these commands are enough.

Robustness on one dataset, printing radius and certified counts

```
.venv\Scripts\python -m obddrf.experiments.run_robustness --dataset iris
```

The same on another model, choosing tree prefix, budget and selector

```
.venv\Scripts\python -m obddrf.experiments.run_robustness --dataset banknote --trees 20 --budget 10000 --selector modal_var
```

Summary table of the 54 datasets with input bits, classes and trees

```
.venv\Scripts\python -m obddrf.experiments.datasets_overview
```

Draw one tree of the forest as an image

```
.venv\Scripts\python -m obddrf.experiments.draw_trees --dataset iris
```

Interactive OBDD in HTML for the small models, with zoom and click on features

```
.venv\Scripts\python -m obddrf.experiments.obdd_interactive --dataset iris
```

## Reproduce the full study

First the test suite with the oracle of every phase. If this passes, the
pipeline is correct end to end.

```
.venv\Scripts\python -m pytest
```

Run the six commands below in this order. The grid goes first because
robustness, readouts and figures read its CSV. The dataset table goes before
the figures. Grid, robustness and readouts have --resume, so they can be
interrupted by closing the window and resumed by rerunning the same command.
Rerunning from scratch overwrites the existing CSVs.

Dataset table with input bits, classes and trees

```
.venv\Scripts\python -m obddrf.experiments.datasets_overview
```

Grid campaign over 54 models, 3 budgets, 5 selectors and growing prefixes with
the adjacent pair verified at the wall. This is the long part, a few hours

```
.venv\Scripts\python -m obddrf.experiments.run_grid --selectors infogain shapley pvalue random_node modal_var --locate-wall --jobs 4 --resume --out results\grid_campagna.csv
```

Grid analysis with walls, budgets and selectors

```
.venv\Scripts\python -m obddrf.experiments.analyze_grid results\grid_campagna.csv
```

Robustness over the test sets with up to 50 samples per model

```
.venv\Scripts\python -m obddrf.experiments.run_robustness_batch --max-samples 50 --timeout 600 --jobs 4
```

Readouts with counts, volume and bit rankings

```
.venv\Scripts\python -m obddrf.experiments.run_readouts --jobs 4 --resume
```

Study figures

```
.venv\Scripts\python -m obddrf.experiments.make_figures
```

With little RAM or few cores lower --jobs to 2. The grid and robustness
parameters are in `src/obddrf/config.py` with the rationale in comments.

## Final results

Robustness covers 50 of the 54 models. `factors` and `karhunen` reach the
600-second limit. `letter` and `texture` have no compilable prefix. The 50
covered models give 2304 samples. Of these 2229 are certified and 75 stay out
of the distribution. Every certified sample counts once even when several
rivals sit at the same distance.
