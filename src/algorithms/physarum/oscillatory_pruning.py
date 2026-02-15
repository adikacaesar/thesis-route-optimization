from dataclasses import dataclass
from typing import Dict, Tuple
import math


@dataclass
class PruneConfig:
    # konservatif:
    threshold: float = 0.20      # ambang rendah -> prune lebih jarang
    amplitude: float = 0.05      # osilasi kecil
    omega: float = 0.8           # frekuensi sedang
    patience: int = 5            # harus "jelek" 5x berturut-turut baru prune
    min_edges_keep: int = 60     # jangan prune sampai graf terlalu tipis


class OscillatoryPruner:
    """
    Edge dipruning kalau skor (tau + A*sin(w*t)) terus di bawah threshold
    selama 'patience' iter berturut-turut.
    """
    def __init__(self, cfg: PruneConfig):
        self.cfg = cfg
        self.bad_streak: Dict[Tuple[str, str], int] = {}  # edge -> streak count

    def step_and_prune(self, tau: Dict[Tuple[str, str], float], t_iter: int) -> int:
        if len(tau) <= self.cfg.min_edges_keep:
            return 0

        osc = self.cfg.amplitude * math.sin(self.cfg.omega * t_iter)

        # update streaks
        for e, val in list(tau.items()):
            score = val + osc
            if score < self.cfg.threshold:
                self.bad_streak[e] = self.bad_streak.get(e, 0) + 1
            else:
                self.bad_streak[e] = 0

        # prune candidates
        candidates = [e for e, s in self.bad_streak.items() if s >= self.cfg.patience and e in tau]

        # cap pruning so we keep at least min_edges_keep
        max_prune = max(0, len(tau) - self.cfg.min_edges_keep)
        candidates = candidates[:max_prune]

        for e in candidates:
            tau.pop(e, None)
            self.bad_streak.pop(e, None)

        return len(candidates)
