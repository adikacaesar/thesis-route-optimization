from typing import List, Set, Tuple
from src.model.graph import Graph

def _simulate_move_cost(
    g: Graph,
    current_id: str,
    next_id: str,
    current_time: int,
    late_penalty: float,
) -> Tuple[float, int, int, int]:
    """
    Simulasikan pindah dari current -> next pada current_time.
    Return:
      (incremental_cost, new_time_after_service, wait, late)
    """
    travel = int(round(g.travel_time(current_id, next_id)))
    t_arrive = current_time + travel

    poi = g.pois[next_id]

    wait = 0
    if t_arrive < poi.open_min:
        wait = poi.open_min - t_arrive

    t_start = t_arrive + wait

    late = 0
    if t_start > poi.close_min:
        late = t_start - poi.close_min

    t_depart = t_start + poi.service_min

    inc_cost = travel + wait + late_penalty * late
    return float(inc_cost), t_depart, wait, late


def greedy_timewindow_aware(
    g: Graph,
    start_id: str,
    end_id: str,
    visit_ids: List[str],
    start_time_min: int,
    late_penalty: float = 10.0,
) -> List[str]:
    """
    Greedy time-window aware:
    - tiap langkah pilih next yang menghasilkan incremental cost terendah
      (travel + wait + late_penalty*late)
    """
    remaining: Set[str] = set(visit_ids)
    route: List[str] = [start_id]
    current = start_id
    t = start_time_min

    while remaining:
        best = None  # (inc_cost, next_id, new_time)
        for cand in remaining:
            inc_cost, t_new, wait, late = _simulate_move_cost(g, current, cand, t, late_penalty)
            key = (inc_cost, late, wait, cand)  # tie-break: anti-late, anti-wait
            if best is None or key < best[0]:
                best = (key, cand, t_new)

        _, next_id, t = best
        route.append(next_id)
        remaining.remove(next_id)
        current = next_id

    # terakhir ke end_id (anggap end_id boleh tanpa service, tapi file POI kita sudah service=0)
    route.append(end_id)
    return route


def greedy_nearest_feasible(
    g: Graph,
    start_id: str,
    end_id: str,
    visit_ids: List[str],
    start_time_min: int,
) -> List[str]:
    """
    Greedy sederhana (yang sebelumnya):
    pilih next berdasar travel_time saja.
    """
    remaining: Set[str] = set(visit_ids)
    route: List[str] = [start_id]
    current = start_id

    while remaining:
        next_id = min(remaining, key=lambda x: g.travel_time(current, x))
        route.append(next_id)
        remaining.remove(next_id)
        current = next_id

    route.append(end_id)
    return route
