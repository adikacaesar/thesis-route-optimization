from src.model.graph import load_pois, load_time_matrix, Graph
from src.model.validate import validate_graph, reachable_from
from src.model.objective import evaluate_route, print_schedule
from src.algorithms.greedy import greedy_nearest_feasible, greedy_timewindow_aware
from src.algorithms.ga.ga_core import run_ga, GAConfig
from src.algorithms.hybrid.ga_physarum import run_hybrid_ga_physarum, HybridConfig
from src.algorithms.physarum.physarum_core import PhysarumConfig
from src.algorithms.physarum.oscillatory_pruning import PruneConfig
from src.eval.logger import save_run_log


def main():
    # =========================
    # Load data
    # =========================
    pois = load_pois("data/processed/poi.csv")
    times = load_time_matrix("data/processed/time_matrix.csv")
    g = Graph(pois, times)

    # =========================
    # M1: Hello Graph
    # =========================
    print("=== M1: Hello Graph ===")
    print(f"POI count: {len(g.pois)}")
    print(f"Edge count: {len(g.travel_min)}")
    for (u, v) in [("A", "B"), ("B", "C"), ("C", "E"), ("I", "J")]:
        print(f"travel {u}->{v} = {g.travel_time(u, v)} min")

    # =========================
    # M1.5: Validate Graph
    # =========================
    print("\n=== M1.5: Validate Graph ===")
    errors = validate_graph(g)
    if not errors:
        print("OK: graph valid (no missing ids/edges detected)")
    else:
        print("Found issues:")
        for e in errors:
            print("-", e)

    # =========================
    # M1.6: Reachability Check
    # =========================
    print("\n=== M1.6: Reachability Check ===")
    start_id = "A"
    reach = reachable_from(g, start_id)
    print(f"Reachable from {start_id}: {len(reach)}/{len(g.pois)}")
    missing = sorted(set(g.pois.keys()) - reach)
    if missing:
        print("Not reachable:", missing)
    else:
        print(f"OK: all POIs reachable from {start_id}")

    # =========================
    # M2.5: Compare Greedy Baselines
    # =========================
    print("\n=== M2.5: Compare Greedy Baselines ===")

    start_id = "A"
    end_id = "J"
    visit_ids = ["B", "C", "D", "E", "F"]
    start_time = 480  # 08:00

    route1 = greedy_nearest_feasible(g, start_id, end_id, visit_ids, start_time)
    res1 = evaluate_route(g, route1, start_time_min=start_time, late_penalty=10.0)

    route2 = greedy_timewindow_aware(g, start_id, end_id, visit_ids, start_time, late_penalty=10.0)
    res2 = evaluate_route(g, route2, start_time_min=start_time, late_penalty=10.0)

    print("\n[Greedy travel-only]")
    print("Route:", " -> ".join(route1))
    print(f"Travel {res1.total_travel} | Wait {res1.total_wait} | Late {res1.total_late} | Cost {res1.total_cost:.2f}")

    print("\n[Greedy time-window aware]")
    print("Route:", " -> ".join(route2))
    print(f"Travel {res2.total_travel} | Wait {res2.total_wait} | Late {res2.total_late} | Cost {res2.total_cost:.2f}")

    # =========================
    # M3: Genetic Algorithm
    # =========================
    print("\n=== M3: Genetic Algorithm (GA) ===")

    ga_cfg = GAConfig(
        population_size=60,
        generations=150,
        crossover_rate=0.9,
        mutation_rate=0.2,
        tournament_k=3,
        seed=123,
    )

    best_route, best_cost = run_ga(
        g=g,
        start_id=start_id,
        end_id=end_id,
        visit_ids=visit_ids,
        start_time_min=start_time,
        late_penalty=10.0,
        cfg=ga_cfg,
    )

    print("\n[GA Result]")
    print("Best route:", " -> ".join(best_route))
    print(f"Best cost: {best_cost:.2f}")

    ga_res = evaluate_route(g, best_route, start_time_min=start_time, late_penalty=10.0)
    print_schedule(g, ga_res, title="GA best schedule")

    # =========================
    # M4 + M5: Hybrid GA + Physarum + Oscillatory Pruning
    # =========================
    print("\n=== M4+M5: Hybrid GA + Physarum + Oscillatory Pruning (Conservative) ===")

    ga_cfg_fast = GAConfig(
        population_size=50,
        generations=80,
        crossover_rate=0.9,
        mutation_rate=0.2,
        tournament_k=3,
        seed=123,
    )

    phy_cfg = PhysarumConfig(
        tau_init=1.0,
        evap_rate=0.05,
        deposit_q=2.0,
        eps=1e-6,
    )

    hy_cfg = HybridConfig(
        outer_iters=8,
        late_penalty=10.0,
        start_time_min=start_time,
    )

    pr_cfg = PruneConfig(
        threshold=0.20,
        amplitude=0.05,
        omega=0.8,
        patience=5,
        min_edges_keep=60,
    )

    hy_route, hy_cost = run_hybrid_ga_physarum(
        base_g=g,
        start_id=start_id,
        end_id=end_id,
        visit_ids=visit_ids,
        ga_cfg=ga_cfg_fast,
        phy_cfg=phy_cfg,
        hy_cfg=hy_cfg,
        pr_cfg=pr_cfg,
    )

    print("\n[Hybrid+Pruning Result]")
    print("Best route:", " -> ".join(hy_route))
    print(f"Best base cost: {hy_cost:.2f}")

    hy_res = evaluate_route(g, hy_route, start_time_min=start_time, late_penalty=10.0)
    print_schedule(g, hy_res, title="Hybrid GA+Physarum+Pruning best schedule")

    # =========================
    # Save run log
    # =========================
    log_text = []
    log_text.append("=== GA + HYBRID(+PRUNING) RUN SUMMARY ===")
    log_text.append(f"visit_ids={visit_ids}")
    log_text.append(f"start_time={start_time}")
    log_text.append(f"ga_cfg={ga_cfg}")
    log_text.append(f"ga_cfg_fast={ga_cfg_fast}")
    log_text.append(f"phy_cfg={phy_cfg}")
    log_text.append(f"hy_cfg={hy_cfg}")
    log_text.append(f"pr_cfg={pr_cfg}")
    log_text.append(f"best_ga_route={' -> '.join(best_route)}")
    log_text.append(f"best_ga_cost={best_cost:.2f}")
    log_text.append(f"best_hybrid_route={' -> '.join(hy_route)}")
    log_text.append(f"best_hybrid_cost={hy_cost:.2f}")

    path = save_run_log("hybrid_pruning_run", "\n".join(log_text))
    print(f"\nSaved run log: {path}")


if __name__ == "__main__":
    main()
