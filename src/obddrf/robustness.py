# FASE 7 della Sez 13 misura quanti bit vanno girati al minimo per far certificare un'altra classe
# e' lo stesso passaggio dal basso della FASE 5 ma con il minimo al posto della somma come da Teorema 2
# girare un bit di cella costa uno mentre far vincere un rivale e' vietato come da Prop 11
# sulla frontiera si paga anche l'ingresso nella fetta e si prende il minimo su tutte le fette
# la fuga invece misura i bit per perdere il proprio certificato cioe' la distanza dal complemento di Win

from math import inf

from .compose import state_var_names
from .counters import win_var
from .discretize import DiscretizedForest
from .frontier import FrontierLeaf
from .readouts import bottom_up_pass


def min_switch(
    bdd,
    dforest: DiscretizedForest,
    acc,
    rho: dict[str, bool],
    sample_bits: dict[str, bool],
    target_label: int,
) -> float:
    # distanza minima dentro una fetta calcolata con la passata min piu' come da Teorema 2
    free_inputs = [v for v in dforest.input_bit_names() if v not in rho]
    ordered = free_inputs + state_var_names(dforest)
    input_set = set(free_inputs)
    flag_of = {win_var(l): l for l in dforest.labels}

    def weight(i: int, b: bool) -> float:
        # costo di un bit nella passata min piu' come da Prop 11
        var = ordered[i]
        if var in input_set:  # girare un bit di cella costa uno se e' diverso dal campione
            return 0.0 if b == sample_bits[var] else 1.0
        if var in flag_of:  # il flag del target deve accendersi e quelli dei rivali no
            if flag_of[var] == target_label:
                return 0.0 if b else inf
            return inf if b else 0.0
        return 0.0  # i bit dei contatori sono liberi e non costano nulla

    return bottom_up_pass(
        bdd,
        acc,
        ordered,
        weight,
        add=min,
        mul=lambda a, b: a + b,
        zero=inf,
        one=0.0,
    )


def robustness(
    bdd,
    dforest: DiscretizedForest,
    leaves: list[FrontierLeaf],
    sample_bits: dict[str, bool],
    target_label: int,
) -> float:
    # raggio verso la classe scelta prendendo il minimo su tutte le fette come da Prop 11
    best = inf
    for leaf in leaves:
        entry = sum(1 for var, val in leaf.rho.items() if sample_bits[var] != val)
        if entry >= best:
            continue  # saltiamo la fetta se solo entrarci costa gia' piu' del minimo trovato
        d = min_switch(bdd, dforest, leaf.acc, leaf.rho, sample_bits, target_label)
        best = min(best, entry + d)
    return best


_TRUE, _FALSE = -1, -2  # codici dei terminali negli array compilati


class CompiledLeaf:
    # copia il diagramma di una fetta dentro semplici array cosi' la distanza si calcola veloce per ogni campione
    # i figli negati vengono raddrizzati durante la copia come verificato nei test

    __slots__ = ("rho", "input_name", "flag_lab", "lo", "hi", "root")

    def __init__(self, bdd, dforest, leaf: FrontierLeaf):
        # copia la fetta in array piani raddrizzando gli archi negati
        input_set = set(dforest.input_bit_names())
        flag_of = {win_var(l): l for l in dforest.labels}
        self.rho = leaf.rho
        self.input_name: list = []   # nome del bit di input oppure niente
        self.flag_lab: list[int] = []  # etichetta del flag di vittoria oppure meno uno
        self.lo: list[int] = []
        self.hi: list[int] = []
        idx: dict = {}

        import sys

        sys.setrecursionlimit(max(sys.getrecursionlimit(), 200_000))

        def build(f) -> int:
            # copia un nodo e i suoi figli e assegna a ognuno un indice
            if f == bdd.true:
                return _TRUE
            if f == bdd.false:
                return _FALSE
            if f in idx:
                return idx[f]
            colo, cohi = (~f.low, ~f.high) if f.negated else (f.low, f.high)
            li, hii = build(colo), build(cohi)  # prima i figli poi il padre
            i = len(self.lo)
            v = f.var
            self.input_name.append(v if v in input_set else None)
            self.flag_lab.append(flag_of.get(v, -1))
            self.lo.append(li)
            self.hi.append(hii)
            idx[f] = i
            return i

        self.root = build(leaf.acc)

    def _walk(self, sample_bits, flag_w) -> tuple[float, frozenset]:
        # passata min piu' sugli array come da Teorema 2 con i pesi dei flag decisi da chi chiama
        inp, fl, lo, hi = self.input_name, self.flag_lab, self.lo, self.hi
        n = len(lo)
        dist = [0.0] * n
        choice = [False] * n
        for i in range(n):
            name = inp[i]
            if name is not None:  # girare il bit costa uno se e' diverso dal campione
                sb = sample_bits[name]
                w0 = 0.0 if not sb else 1.0
                w1 = 0.0 if sb else 1.0
            else:
                lab = fl[i]
                if lab < 0:  # contatore libero che non costa nulla
                    w0 = w1 = 0.0
                else:
                    w0, w1 = flag_w[lab]
            li, hii = lo[i], hi[i]
            d0 = 0.0 if li == _TRUE else (inf if li == _FALSE else dist[li])
            d1 = 0.0 if hii == _TRUE else (inf if hii == _FALSE else dist[hii])
            c0, c1 = w0 + d0, w1 + d1
            if c0 <= c1:
                dist[i], choice[i] = c0, False
            else:
                dist[i], choice[i] = c1, True
        r = self.root
        total = 0.0 if r == _TRUE else (inf if r == _FALSE else dist[r])
        flips: set = set()
        i = r
        while i >= 0:
            name = inp[i]
            if name is not None and choice[i] != sample_bits[name]:
                flips.add(name)
            i = hi[i] if choice[i] else lo[i]
        return total, frozenset(flips)

    def radius_witness(self, sample_bits, target_label) -> tuple[float, frozenset]:
        # il flag del target deve accendersi e quelli dei rivali no come da Prop 11
        flag_w = {
            lab: (inf, 0.0) if lab == target_label else (0.0, inf)
            for lab in self.flag_lab
            if lab >= 0
        }
        return self._walk(sample_bits, flag_w)

    def escape_witness(self, sample_bits, own_label) -> tuple[float, frozenset]:
        # per perdere il certificato basta spegnere il flag della propria classe e gli altri restano liberi
        flag_w = {
            lab: (0.0, inf) if lab == own_label else (0.0, 0.0)
            for lab in self.flag_lab
            if lab >= 0
        }
        return self._walk(sample_bits, flag_w)


def compile_frontier(bdd, dforest, leaves: list[FrontierLeaf]) -> list[CompiledLeaf]:
    # compila ogni fetta una volta sola e poi la si riusa per tutti i campioni
    return [CompiledLeaf(bdd, dforest, leaf) for leaf in leaves]


def robustness_witness_compiled(
    compiled: list[CompiledLeaf],
    sample_bits: dict[str, bool],
    target_label: int,
) -> tuple[float, frozenset]:
    # raggio e bit da girare usando le fette gia' compilate ed e' la funzione usata davvero nel batch
    best: tuple[float, frozenset] = (inf, frozenset())
    for cl in compiled:
        entry = frozenset(v for v, val in cl.rho.items() if sample_bits[v] != val)
        if len(entry) >= best[0]:
            continue
        d, flips = cl.radius_witness(sample_bits, target_label)
        if len(entry) + d < best[0]:
            best = (len(entry) + d, entry | flips)
    return best


def escape_witness_compiled(
    compiled: list[CompiledLeaf],
    sample_bits: dict[str, bool],
    own_label: int,
) -> tuple[float, frozenset]:
    # distanza e bit per uscire dalla regione certificata della propria classe prendendo il minimo sulle fette
    best: tuple[float, frozenset] = (inf, frozenset())
    for cl in compiled:
        entry = frozenset(v for v, val in cl.rho.items() if sample_bits[v] != val)
        if len(entry) >= best[0]:
            continue
        d, flips = cl.escape_witness(sample_bits, own_label)
        if len(entry) + d < best[0]:
            best = (len(entry) + d, entry | flips)
    return best


def robustness_witness(
    bdd,
    dforest: DiscretizedForest,
    leaves: list[FrontierLeaf],
    sample_bits: dict[str, bool],
    target_label: int,
) -> tuple[float, frozenset]:
    # comodo per i test perche' compila al volo mentre il batch compila una volta sola
    return robustness_witness_compiled(
        compile_frontier(bdd, dforest, leaves), sample_bits, target_label
    )


def robustness_profile(
    bdd,
    dforest: DiscretizedForest,
    leaves: list[FrontierLeaf],
    sample_bits: dict[str, bool],
) -> dict[int, float]:
    # raggio verso ogni classe e quello della classe certificata vale zero
    return {
        l: robustness(bdd, dforest, leaves, sample_bits, l) for l in dforest.labels
    }
