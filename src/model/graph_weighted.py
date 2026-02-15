from typing import Optional
from src.model.graph import Graph
from src.algorithms.physarum.physarum_core import PhysarumModel


class WeightedGraph:
    """
    Wrapper Graph: travel_time() mengembalikan bobot efektif berdasarkan Physarum.
    """
    def __init__(self, base_graph: Graph, physarum: PhysarumModel):
        self.base = base_graph
        self.physarum = physarum
        self.pois = base_graph.pois  # supaya kompatibel dengan evaluator

    def travel_time(self, u: str, v: str) -> float:
        base_w = self.base.travel_time(u, v)
        return self.physarum.effective_weight(u, v, base_w)
