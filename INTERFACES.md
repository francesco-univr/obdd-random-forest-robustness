# INTERFACES — strutture dati condivise

> Documento vivo: aggiornarlo a OGNI modifica di una struttura condivisa
> (disciplina del prompt). I riferimenti a Sezioni/Definizioni sono a
> `bdss-project-assignment-2526.pdf`.

## 1. Forest discretizzata (Sezioni 3–4) — `obddrf.discretize.DiscretizedForest`

```python
@dataclass
class DiscretizedForest:
    n_features: int                 # F
    n_trees: int                    # D
    labels: list[int]               # L = rf.classes_ (interi 0..|L|-1)
    thresholds: list[np.ndarray]    # per feature f: soglie ordinate strettamente
                                    # crescenti l_0 < ... < l_{k-1} raccolte da TUTTI gli alberi
    # derivati:
    #   cells(f)  = len(thresholds[f]) + 1          (Def. 2; K celle, indici 0..K-1)
    #   b_f       = ceil(log2(cells(f))), minimo 1   (Def. 3)
    trees: list[TreeNodes]          # struttura sklearn riletta (vedi §5)
```

- Celle (Def. 1): `(-inf, l_0], (l_0, l_1], ..., (l_{k-1}, +inf)` — aperte a sinistra,
  chiuse a destra, TRANNE l'ultima. Tutti i test sono in forma `<=` (vincolo 2).
- Nei nodi sklearn il test `x[f] <= threshold` viene tradotto in un indice di soglia
  `tau in {0..cells(f)-2}`: la posizione della soglia nell'array `thresholds[f]`.
- Clamping (Remark 1): se l'intero codificato dai bit supera `cells(f)-1`
  (possibile solo se `cells(f)` non è potenza di 2) si interpreta come ULTIMA cella.
  Nessun vincolo exactly-one, mai (vincolo 1).

## 2. Naming variabili BDD (Def. 3, 5, 8 + vincolo 5)

| ruolo                        | nome           | note                                   |
|------------------------------|----------------|----------------------------------------|
| bit di input (cella)         | `B_{f}_{j}`    | j = 0 è il bit MENO significativo       |
| bit contatore etichetta      | `lc_{l}_{j}`   | larghezza M+1, M = ceil(log2(D/2)) + 1 (Def. 5; il bit extra tiene LC+R_t nella parola, Sez. 7) |
| flag vittoria                | `w_{l}`        | il `label_win^l` del documento          |
| copie *next* (barrate)       | suffisso `_n`  | `lc_{l}_{j}_n`, `w_{l}_n`               |

`b_f = ceil(log2 cells(f))` esatto: vale **0** se `cells(f) = 1` (feature mai
testata: nessuna variabile, X_f è costante 0). Helper: `counters.lc_var`,
`counters.win_var`, `discretize.input_bit_name`, `compose.state_var_names`,
`compose.rename_next_to_current`.

Ordine variabili nell'OBDD (vincolo 5 — leva di performance n.1):

```
B_0_{b0-1} ... B_0_0, B_1_{b1-1} ... , B_{F-1}_0,        # tutti i bit di input in cima
lc_0_0, lc_0_0_n, lc_0_1, lc_0_1_n, ...,                 # ogni current INTERLACCIATO
w_0, w_0_n, w_1, w_1_n, ...                              # col proprio next
```

(L'ordine interno dei bit di input — MSB-first per feature — è una scelta di
default; può essere rivisto, ma va cambiato SOLO qui e in `compose.declare_vars`.)

## 3. Formato di un sample

```
reale:  np.ndarray shape (F,) float            — riga del CSV
celle:  tuple[int, ...] lunghezza F            — indice di cella per feature (Def. 1, test <=)
bit:    dict[str, bool]                        — {f"B_{f}_{j}": bit_j(cell_f)} per ogni f, j
```

Conversioni in `obddrf.discretize`: `real_to_cells`, `cells_to_bits`, `bits_to_cells`.
Round-trip `reale -> celle -> bit -> celle` lossless (oracolo Fase 1).

## 4. Foglia della frontiera (Def. 19, 21) — `obddrf.frontier.FrontierLeaf`

Tripla `(rho, A, t)`:
- `rho`: assignment parziale dei bit di input, `dict[str, bool]` (il cubo chi_rho)
- `A`: accumulatore OBDD cofattorizzato su rho (dopo gli alberi T_0..T_{t-1})
- `t`: indice del prossimo albero da comporre; foglia FINALE quando t == D

Invariante (Prop. 10): i cubi chi_rho partizionano {0,1}^{n_x};
ogni Phi_t residuo va proiettato su rho (`Phi_t|rho`) — obbligatorio, Prop. 10.3.

## 5. Albero sklearn riletto — `obddrf.forest_io.TreeNodes`

Array paralleli estratti da `estimator.tree_` (children_left/right, feature,
threshold, value); foglia quando `children_left[i] == -1`; etichetta di foglia
`lambda_E(b) = argmax(value[b])` (in caso di parità sklearn prende il primo —
si replica `np.argmax`). Convenzione sklearn: ramo sinistro = test `<=` vero
(d_i = 1 nella Def. 6), ramo destro = falso.

## 6. Backend OBDD — `obddrf.bdd_backend`

`make_bdd()` restituisce un manager `dd` con API comune (`declare`, `add_expr`,
`let`, `exist`, `count`, `apply`). Preferisce `dd.cudd` (vincolo 3); su questa
macchina (Windows, niente toolchain CUDD) ripiega su `dd.autoref` — stessa API.
`BACKEND` vale `"cudd"` o `"autoref"`. I passaggi bottom-up custom (conteggio
pesato, Shapley, MinSwitch) sono implementati genericamente in `readouts.py` e
funzionano su entrambi.

## 7. Caricamento modelli — `obddrf.forest_io`

`datasets/datasets/<name>/<name>_rf.pkl` è un **dump joblib** (NON pickle puro)
di un dict: chiavi `model` (RandomForestClassifier, sklearn 1.4.2), `dataset`,
`params`, `cv_acc`, `test_acc`, `n_features`, `n_classes`, `n_rows`,
`train_rows`, `test_rows`, `seed`. Caricare SOLO con `joblib.load`.

CSV: `<name>_train.csv` / `<name>_test.csv`, ultima colonna = etichetta
(stringa o numero); `rf.classes_` sono interi. Mappa: `forest_io.label_encoding`
(convenzione LabelEncoder = etichette uniche ordinate), **validata** contro
`bundle["test_acc"]` ad ogni chiamata.

## 8. Readouts e robustezza — `obddrf.readouts` / `obddrf.robustness`

- `bottom_up_pass(bdd, u, ordered_vars, weight, add, mul, zero, one)`:
  l'unico schema bottom-up (Teorema 1); `ordered_vars` nell'ordine del
  manager, supporto di `u` incluso. Cofattori via `let` (robusto rispetto
  agli archi complementati, identico su cudd/autoref).
- Specializzazioni: `model_count` (delega a `BDD.count`), `weighted_count`
  (Prop. 5, pesi `(w0, w1)` per variabile), `zero_stratified_poly` +
  `shapley_values` (Prop. 8), `pvalues` (Def. 18, log-gamma).
- `robustness.min_switch[_witness]` (Teorema 2 / Prop. 11): min-plus sugli
  OBDD di foglia con contatori e flag ancora vivi; il witness ritorna i bit
  di cella flippati. `robustness(...)` / `robustness_witness(...)` /
  `robustness_profile(...)` aggregano sulla frontiera (entry flips di rho +
  MinSwitch interno).
- Percorso compilato (usato dal batch): `CompiledLeaf` estrae la fetta in
  array post-ordine (cofattori complement-safe); `compile_frontier(bdd,
  dforest, leaves) -> list[CompiledLeaf]`; `robustness_witness_compiled(
  compiled, sample_bits, target_label) -> (raggio, frozenset bit)`.
- **Fuga da Win** (distanza dal complemento della regione certificata
  propria): `CompiledLeaf.escape_witness(sample_bits, own_label)` e
  `escape_witness_compiled(compiled, sample_bits, own_label) -> (raggio,
  frozenset bit)`. Pesi: flag della propria classe alto = inf, altri flag
  liberi, contatori liberi. Invarianti testate: fuga >= 1 per un campione
  certificato e fuga <= min raggio rivale; coincide col rivale sui binari
  con alberi dispari.

## 9. Esperimenti — `obddrf.experiments`

- `run_grid.run_triple(rf, name, k, budget, selector, timeout_s, max_nodes)`
  -> riga dict con schema `run_grid.FIELDS`. CLI `python -m
  obddrf.experiments.run_grid`. La progressione k si ferma al primo fallimento
  per ogni coppia di budget e selettore. La bisezione verifica la coppia
  adiacente al muro di scalabilita'.
- `run_robustness.run(name, k, budget, selector, out_path)` -> righe per
  (sample, label) con raggio targeted, flag `is_untargeted_min`, bit/feature
  flippati e colonne `escape_radius` / `escape_bits` / `escape_features`
  ripetute su ogni riga del sample e vuote per i non certificati.
  CLI `python -m obddrf.experiments.run_robustness`.
- `run_robustness_batch` usa un sottoprocesso per dataset. Il figlio scrive il
  CSV per-dataset e manda al padre SOLO la riga di riepilogo (mandare le
  righe intere riempiva la pipe e produceva falsi timeout). `_summarize`
  usa solo i campioni certificati, deduplica per sample (i pareggi fra
  rivali non contano doppio) e usa la mediana statistica. I campi summary sono
  `untargeted_min/median/max` + `escape_min/median/max`.
- I default finali sono `config.ROBUSTNESS_MAX_SAMPLES = 50` e
  `config.ROBUSTNESS_TIMEOUT_S = 600.0`.
- `resources.ResourceMonitor(timeout_s, max_manager_nodes)`: limiti
  cooperativi via callback `on_step`; memoria approssimata dai nodi totali
  del manager.
