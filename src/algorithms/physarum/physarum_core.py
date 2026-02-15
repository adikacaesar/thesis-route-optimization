from dataclasses import dataclass
from typing import Dict, Tuple, List


@dataclass
class PhysarumConfig:
    tau_init: float = 1.0       # nilai awal conductance
    evap_rate: float = 0.05     # evaporasi per iterasi (0..1)
    deposit_q: float = 1.0      # kekuatan deposit
    eps: float = 1e-6           # stabilizer


class PhysarumModel:
    """
    Model conductance tau(u,v) untuk edge POI graph.
    Kita treat edge sebagai directed (u->v).
    """

    def __init__(self, edges: List[Tuple[str, str]], cfg: PhysarumConfig):
        self.cfg = cfg
        self.tau: Dict[Tuple[str, str], float] = {(u, v): cfg.tau_init for (u, v) in edges}

    def evaporate(self):
        r = self.cfg.evap_rate
        for k in list(self.tau.keys()):
            self.tau[k] = max(self.cfg.eps, (1.0 - r) * self.tau[k])

    def deposit_from_route(self, route: List[str], base_cost: float):
        """
        Deposit conductance pada edge yang dipakai route.
        Deposit besar kalau base_cost kecil (rute bagus).
        """
        if base_cost <= 0:
            return
        delta = self.cfg.deposit_q / base_cost

        for i in range(len(route) - 1):
            u = route[i]
            v = route[i + 1]
            key = (u, v)
            if key in self.tau:
                self.tau[key] += delta

    def effective_weight(self, u: str, v: str, base_w: float) -> float:
        """
        Hitung bobot efektif dari base_w dan tau.
        """
        tau_uv = self.tau.get((u, v), self.cfg.tau_init)
        return float(base_w) / (self.cfg.eps + tau_uv)
