# FASE 8 / Sez. 14: limiti cooperativi di tempo e nodi.

import time


class ResourceLimitExceeded(RuntimeError):
    pass  # errore lanciato quando una prova sfora i limiti


class ResourceMonitor:
    # controllo cooperativo chiamato tra una composizione e l'altra che misura la memoria contando i nodi

    def __init__(self, timeout_s: float, max_manager_nodes: int):
        # salva i limiti di tempo e nodi e l'istante di partenza
        self.timeout_s = timeout_s
        self.max_manager_nodes = max_manager_nodes
        self.t0 = time.monotonic()
        self.peak = 0

    def elapsed(self) -> float:
        # secondi trascorsi dall'inizio della prova
        return time.monotonic() - self.t0

    def check(self, bdd, acc) -> None:
        # ferma la prova se supera il tempo o il tetto di nodi
        self.peak = max(self.peak, len(acc))
        if self.elapsed() > self.timeout_s:
            raise ResourceLimitExceeded(f"timeout dopo {self.elapsed():.1f}s")
        n = len(bdd)
        if n > self.max_manager_nodes:
            collect = getattr(bdd, "collect_garbage", None)  # il conteggio include anche i nodi morti quindi si pulisce e si riconta
            if collect is not None:
                collect()
                n = len(bdd)
            if n > self.max_manager_nodes:
                raise ResourceLimitExceeded(
                    f"manager a {n} nodi > {self.max_manager_nodes}"
                )
