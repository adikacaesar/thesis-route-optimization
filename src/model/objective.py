from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from .graph import Graph

@dataclass
class StopSchedule:
    poi_id: str
    arrive: int
    wait: int
    start_service: int
    depart: int
    late: int  # menit terlambat melewati close time

@dataclass
class EvalResult:
    total_travel: int
    total_wait: int
    total_late: int
    total_service: int
    total_cost: float
    schedule: List[StopSchedule]

def evaluate_route(
    g: Graph,
    route: List[str],
    start_time_min: int,
    late_penalty: float = 10.0,
) -> EvalResult:
    """
    route: urutan POI yang akan dikunjungi (mis. ["A","C","B","J"])
    start_time_min: waktu mulai dalam menit (mis. 480 = 08:00)
    late_penalty: bobot penalti keterlambatan (semakin besar -> makin anti telat)
    """
    if len(route) < 1:
        raise ValueError("Route must contain at least 1 node")

    # basic sanity
    for pid in route:
        if pid not in g.pois:
            raise KeyError(f"POI not found in route: {pid}")

    t = start_time_min
    total_travel = 0
    total_wait = 0
    total_late = 0
    total_service = 0
    schedule: List[StopSchedule] = []

    for i, pid in enumerate(route):
        poi = g.pois[pid]

        # travel from previous
        if i > 0:
            prev = route[i - 1]
            travel = int(round(g.travel_time(prev, pid)))
            t += travel
            total_travel += travel

        arrive = t

        # wait if arrive before open
        wait = 0
        if arrive < poi.open_min:
            wait = poi.open_min - arrive
            t += wait
            total_wait += wait

        start_service = t

        # late if start_service after close
        late = 0
        if start_service > poi.close_min:
            late = start_service - poi.close_min
            total_late += late

        # service
        t += poi.service_min
        total_service += poi.service_min

        depart = t

        schedule.append(
            StopSchedule(
                poi_id=pid,
                arrive=arrive,
                wait=wait,
                start_service=start_service,
                depart=depart,
                late=late,
            )
        )

    # cost: travel + wait + penalty*late
    total_cost = float(total_travel + total_wait + late_penalty * total_late)

    return EvalResult(
        total_travel=total_travel,
        total_wait=total_wait,
        total_late=total_late,
        total_service=total_service,
        total_cost=total_cost,
        schedule=schedule,
    )

def fmt_time(m: int) -> str:
    hh = m // 60
    mm = m % 60
    return f"{hh:02d}:{mm:02d}"

def print_schedule(g, result, title: str = "Schedule"):
    print(f"\n--- {title} ---")
    header = f"{'POI':<3} {'Name':<10} {'Arr':>5} {'Wait':>4} {'Start':>5} {'Dep':>5} {'Late':>4} {'TW':>13}"
    print(header)
    print("-" * len(header))

    for s in result.schedule:
        poi = g.pois[s.poi_id]
        tw = f"{fmt_time(poi.open_min)}-{fmt_time(poi.close_min)}"
        print(
            f"{s.poi_id:<3} {poi.name[:10]:<10} "
            f"{fmt_time(s.arrive):>5} {s.wait:>4} "
            f"{fmt_time(s.start_service):>5} {fmt_time(s.depart):>5} "
            f"{s.late:>4} {tw:>13}"
        )

    print("-" * len(header))
    print(
        f"Travel={result.total_travel}  Wait={result.total_wait}  "
        f"Late={result.total_late}  Service={result.total_service}  Cost={result.total_cost:.2f}"
    )
