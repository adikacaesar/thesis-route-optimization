from dataclasses import dataclass
from typing import List, Tuple

from src.model.graph import Graph
from src.model.objective import evaluate_route
from src.model.graph_weighted import WeightedGraph
from src.algorithms.ga.ga_core import run_ga, GAConfig
from src.algorithms.physarum.physarum_core import PhysarumModel, PhysarumConfig


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
) -> Tuple[List[str], float]:
    """
    Return best route (base-cost evaluated) dan best base-cost.
    """
    # init Physarum on all edges from base graph
    edges = list(base_g.travel_min.keys())
    phys = PhysarumModel(edges, phy_cfg)

    best_route = None
    best_base_cost = float("inf")

    for it in range(1, hy_cfg.outer_iters + 1):
        # weighted graph for GA
        wg = WeightedGraph(base_g, phys)

        # run GA using weighted travel_time
        route_eff, cost_eff = run_ga(
            g=wg,
            start_id=start_id,
            end_id=end_id,
            visit_ids=visit_ids,
            start_time_min=hy_cfg.start_time_min,
            late_penalty=hy_cfg.late_penalty,
            cfg=ga_cfg,
        )

        # evaluate best route on BASE graph (real cost)
        res_base = evaluate_route(
            base_g,
            route_eff,
            start_time_min=hy_cfg.start_time_min,
            late_penalty=hy_cfg.late_penalty,
        )
        base_cost = res_base.total_cost

        print(f"[HY] iter {it:02d} | eff_cost {cost_eff:8.2f} | base_cost {base_cost:8.2f}")

        # track global best (base cost)
        if base_cost < best_base_cost:
            best_base_cost = base_cost
            best_route = route_eff[:]

        # update Physarum using base_cost (lebih stabil)
        phys.evaporate()
        phys.deposit_from_route(route_eff, base_cost)

    top = sorted(phys.tau.items(), key=lambda kv: kv[1], reverse=True)[:5]
    print("[HY] top tau:", ", ".join([f"{u}->{v}:{val:.3f}" for ((u,v), val) in top]))

    return best_route, best_base_cost
