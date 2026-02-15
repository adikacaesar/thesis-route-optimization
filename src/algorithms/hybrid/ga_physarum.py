from dataclasses import dataclass
from typing import List, Tuple, Optional

from src.model.graph import Graph
from src.model.objective import evaluate_route
from src.model.graph_weighted import WeightedGraph
from src.algorithms.ga.ga_core import run_ga, GAConfig
from src.algorithms.physarum.physarum_core import PhysarumModel, PhysarumConfig
from src.algorithms.physarum.oscillatory_pruning import OscillatoryPruner, PruneConfig


@dataclass
class HybridConfig:
    outer_iters: int = 10
    late_penalty: float = 10.0
    start_time_min: int = 480


def run_hybrid_ga_physarum(
    base_g: Graph,
    start_id: str,
    end_id: str,
    visit_ids: List[str],
    ga_cfg: GAConfig,
    phy_cfg: PhysarumConfig,
    hy_cfg: HybridConfig,
    pr_cfg: Optional[PruneConfig] = None,
) -> Tuple[List[str], float]:
    """
    Hybrid loop:
      (1) Jalankan GA pada WeightedGraph (bobot efektif dari Physarum)
      (2) Evaluasi rute terbaik di base graph (real cost)
      (3) Update Physarum: evaporate + deposit
      (4) Oscillatory pruning (konservatif) untuk memangkas edge lemah
    Return: (best_route_on_base, best_base_cost)
    """
    # Init Physarum on all base edges (directed)
    edges = list(base_g.travel_min.keys())
    phys = PhysarumModel(edges, phy_cfg)

    # Init pruner
    if pr_cfg is None:
        pr_cfg = PruneConfig()
    pruner = OscillatoryPruner(pr_cfg)

    best_route: Optional[List[str]] = None
    best_base_cost = float("inf")

    for it in range(1, hy_cfg.outer_iters + 1):
        # Weighted graph for GA (pruned edges become very expensive inside WeightedGraph)
        wg = WeightedGraph(base_g, phys)

        # Run GA using weighted travel_time
        route_eff, cost_eff = run_ga(
            g=wg,
            start_id=start_id,
            end_id=end_id,
            visit_ids=visit_ids,
            start_time_min=hy_cfg.start_time_min,
            late_penalty=hy_cfg.late_penalty,
            cfg=ga_cfg,
        )

        # Evaluate the same route on BASE graph (real cost)
        res_base = evaluate_route(
            base_g,
            route_eff,
            start_time_min=hy_cfg.start_time_min,
            late_penalty=hy_cfg.late_penalty,
        )
        base_cost = res_base.total_cost

        print(f"[HY] iter {it:02d} | eff_cost {cost_eff:8.2f} | base_cost {base_cost:8.2f}")

        # Track global best (base cost)
        if base_cost < best_base_cost:
            best_base_cost = base_cost
            best_route = route_eff[:]

        # Update Physarum using base_cost (lebih stabil daripada eff_cost)
        phys.evaporate()
        phys.deposit_from_route(route_eff, base_cost)

        # Oscillatory pruning (in-place modifies phys.tau)
        pruned = pruner.step_and_prune(phys.tau, it)

        # Debug ringkas pruning
        if it == 1 or it % 2 == 0 or it == hy_cfg.outer_iters:
            print(f"[PR] iter {it:02d} | pruned {pruned:3d} | edges_left {len(phys.tau)}")

    if best_route is None:
        # should never happen, but keep safe
        best_route = [start_id] + visit_ids + [end_id]
        best_base_cost = evaluate_route(
            base_g,
            best_route,
            start_time_min=hy_cfg.start_time_min,
            late_penalty=hy_cfg.late_penalty,
        ).total_cost

    return best_route, best_base_cost
