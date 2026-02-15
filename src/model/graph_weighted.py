from src.model.graph import Graph
from src.algorithms.physarum.physarum_core import PhysarumModel


class WeightedGraph:
    """
    Wrapper Graph: travel_time() mengembalikan bobot efektif berdasarkan Physarum.
    Kalau edge sudah dipruning (tidak ada di physarum.tau), bobot dibuat sangat besar
    agar GA otomatis menghindari edge tersebut.
    """
    def __init__(self, base_graph: Graph, physarum: PhysarumModel, pruned_penalty: float = 1e6):
        self.base = base_graph
        self.physarum = physarum
        self.pois = base_graph.pois
        self.pruned_penalty = pruned_penalty

    def travel_time(self, u: str, v: str) -> float:
        base_w = self.base.travel_time(u, v)
        if (u, v) not in self.physarum.tau:
            return float(self.pruned_penalty)
        return self.physarum.effective_weight(u, v, base_w)
