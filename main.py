from src.model.graph import load_pois, load_time_matrix, Graph
from src.model.validate import validate_graph, reachable_from
from src.model.objective import evaluate_route, print_schedule
from src.algorithms.greedy import greedy_nearest_feasible, greedy_timewindow_aware
from src.algorithms.ga.ga_core import run_ga, GAConfig
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

    # Greedy travel-only
    route1 = greedy_nearest_feasible(g, start_id, end_id, visit_ids, start_time)
    res1 = evaluate_route(g, route1, start_time_min=start_time, late_penalty=10.0)

    # Greedy time-window aware
    route2 = greedy_timewindow_aware(g, start_id, end_id, visit_ids, start_time, late_penalty=10.0)
    res2 = evaluate_route(g, route2, start_time_min=start_time, late_penalty=10.0)

    print("\n[Greedy travel-only]")
    print("Route:", " -> ".join(route1))
    print(f"Travel {res1.total_travel} | Wait {res1.total_wait} | Late {res1.total_late} | Cost {res1.total_cost:.2f}")

    print("\n[Greedy time-window aware]")
    print("Route:", " -> ".join(route2))
    print(f"Travel {res2.total_travel} | Wait {res2.total_wait} | Late {res2.total_late} | Cost {res2.total_cost:.2f}")

    # =========================
    # M3: Genetic Algorithm (GA)
    # =========================
    print("\n=== M3: Genetic Algorithm (GA) ===")

    cfg = GAConfig(
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
        cfg=cfg,
    )

    print("\n[GA Result]")
    print("Best route:", " -> ".join(best_route))
    print(f"Best cost: {best_cost:.2f}")

    # Schedule detail
    ga_res = evaluate_route(g, best_route, start_time_min=start_time, late_penalty=10.0)
    print_schedule(g, ga_res, title="GA best schedule")

    # =========================
    # Save run log
    # =========================
    log_text = []
    log_text.append("=== GA RUN SUMMARY ===")
    log_text.append(f"visit_ids={visit_ids}")
    log_text.append(f"start_time={start_time}")
    log_text.append(f"late_penalty=10.0")
    log_text.append(f"cfg={cfg}")
    log_text.append(f"best_route={' -> '.join(best_route)}")
    log_text.append(f"best_cost={best_cost:.2f}")
    log_text.append(
        f"travel={ga_res.total_travel} "
        f"wait={ga_res.total_wait} "
        f"late={ga_res.total_late} "
        f"service={ga_res.total_service}"
    )

    path = save_run_log("ga_run", "\n".join(log_text))
    print(f"\nSaved run log: {path}")


if __name__ == "__main__":
    main()
