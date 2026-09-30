# Robustness of a Random-Forest Classifier via OBDD

Progetto BDSS 2025/26 dell'Universita' di Verona. La relazione completa si
trova in `report.pdf`. Una versione in inglese di questo README e' in
`README_EN.md`.

## Struttura

- `src/obddrf/` contiene la pipeline con un modulo per fase della consegna
- `src/obddrf/experiments/` contiene campagna, analisi e generazione delle figure
- `tests/` contiene 110 test con un file di oracoli per fase
- `results/` contiene CSV e figure dello studio
- `report.pdf` e' la relazione completa
- `INTERFACES.md` documenta le strutture dati condivise fra i moduli

I comandi qui sotto leggono i 54 dataset da `datasets/datasets/<nome>/`, gia'
inclusi nell'archivio accanto a `src/`. Sono gli stessi forniti a lezione con
csv di train e test e il modello pkl per ogni dataset.

## Ambiente

Il progetto usa Python 3.12 e le dipendenze elencate in `requirements.txt`.
Scikit-learn resta fissato alla versione 1.4.2 con numpy 1.x per mantenere la
compatibilita' con i modelli salvati.

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\pip install -e .
```

Il secondo comando registra `obddrf` nel venv. Il backend usa la libreria `dd`.
Il codice prova `dd.cudd` e ripiega automaticamente su `dd.autoref` quando CUDD
non e' disponibile.

Da qui in poi ogni comando va lanciato dalla root del progetto con il Python
del venv. Su Windows il prefisso e' `.venv\Scripts\python`, su Linux o Mac e'
`.venv/bin/python`.

## Avvio rapido

Per vedere il progetto in funzione su un singolo modello, senza rilanciare
tutta la campagna, bastano questi comandi.

Robustezza su un dataset con stampa a schermo di raggio e certificati

```
.venv\Scripts\python -m obddrf.experiments.run_robustness --dataset iris
```

Lo stesso su un altro modello scegliendo prefisso di alberi budget e selettore

```
.venv\Scripts\python -m obddrf.experiments.run_robustness --dataset banknote --trees 20 --budget 10000 --selector modal_var
```

Tabella riassuntiva dei 54 dataset con bit di input classi e alberi

```
.venv\Scripts\python -m obddrf.experiments.datasets_overview
```

Disegno di un albero della foresta come immagine

```
.venv\Scripts\python -m obddrf.experiments.draw_trees --dataset iris
```

OBDD interattivo in HTML per i modelli piccoli con zoom e click sulle feature

```
.venv\Scripts\python -m obddrf.experiments.obdd_interactive --dataset iris
```

## Riprodurre lo studio completo

Prima di tutto la suite di test con gli oracoli di ogni fase. Se questa passa
la pipeline e' corretta end to end.

```
.venv\Scripts\python -m pytest
```

I sei comandi qui sotto vanno lanciati in questo ordine. La griglia va per
prima perche' robustezza readouts e figure leggono il suo CSV. La tabella dei
dataset va prima delle figure. Griglia robustezza e readouts hanno --resume
quindi si possono interrompere chiudendo la finestra e riprendere rilanciando
lo stesso comando. Rilanciare da capo sovrascrive i CSV esistenti.

Tabella dei dataset con bit di input classi e alberi

```
.venv\Scripts\python -m obddrf.experiments.datasets_overview
```

Campagna sulla griglia con 54 modelli, 3 budget, 5 selettori e prefissi
progressivi con verifica della coppia adiacente al muro. E' la parte lunga da
alcune ore

```
.venv\Scripts\python -m obddrf.experiments.run_grid --selectors infogain shapley pvalue random_node modal_var --locate-wall --jobs 4 --resume --out results\grid_campagna.csv
```

Analisi della griglia con muri budget e selettori

```
.venv\Scripts\python -m obddrf.experiments.analyze_grid results\grid_campagna.csv
```

Robustezza sui test set con fino a 50 campioni per modello

```
.venv\Scripts\python -m obddrf.experiments.run_robustness_batch --max-samples 50 --timeout 600 --jobs 4
```

Readouts con conteggi volume e ranking dei bit

```
.venv\Scripts\python -m obddrf.experiments.run_readouts --jobs 4 --resume
```

Figure dello studio

```
.venv\Scripts\python -m obddrf.experiments.make_figures
```

Con poca RAM o pochi core abbassare --jobs a 2. I parametri della griglia e
della robustezza si trovano in `src/obddrf/config.py` con la motivazione in
commento.

## Risultati finali

La robustezza copre 50 modelli su 54. `factors` e `karhunen` raggiungono il
limite di 600 secondi. `letter` e `texture` non hanno un prefisso compilabile.
I 50 modelli coperti forniscono 2304 campioni. Di questi 2229 sono certificati
e 75 restano fuori dalla distribuzione. Ogni campione certificato conta una
volta anche quando piu' rivali sono alla stessa distanza.
