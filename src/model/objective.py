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
