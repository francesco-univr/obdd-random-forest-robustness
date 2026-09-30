# raccoglie tutti i parametri regolabili dello studio e ogni altro file li importa da qui

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = PROJECT_ROOT / "datasets" / "datasets"
RESULTS_DIR = PROJECT_ROOT / "results"

# tre tetti di nodi per la frontiera della Def 19 dove il piccolo forza molti tagli e il grande quasi nessuno
BUDGET_GRID: list[int] = [1_000, 10_000, 100_000]

# tempo massimo in secondi per una singola prova prima di dichiararla fallita
TIMEOUT_PER_MODEL_S: float = 300.0

# protocollo finale del batch di robustezza sui test set
ROBUSTNESS_TIMEOUT_S: float = 600.0
ROBUSTNESS_MAX_SAMPLES: int = 50

# tetto di nodi del manager che usiamo come misura portabile della memoria
MAX_MANAGER_NODES: int = 20_000_000
MEMORY_LIMIT_MB: int = 6144  # valore solo informativo della RAM attesa per una prova

# test progressivo della Sezione 1

# quanti alberi provare prima della forest piena e poi la bisezione verifica la coppia adiacente al muro
PROGRESSIVE_TREE_COUNTS: list[int] = [5, 10, 20, 30, 50]

# i cinque criteri di scelta del bit su cui tagliare della Def 21
SELECTORS: list[str] = ["infogain", "shapley", "pvalue", "random_node", "modal_var"]
