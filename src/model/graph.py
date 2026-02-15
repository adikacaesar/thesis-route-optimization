from dataclasses import dataclass
from typing import Dict, Tuple
import csv

@dataclass(frozen=True)
class POI:
    poi_id: str
    name: str
    lat: float
    lon: float
    open_min: int
    close_min: int
    service_min: int

class Graph:
    def __init__(self, pois: Dict[str, POI], travel_min: Dict[Tuple[str, str], float]):
        self.pois = pois
        self.travel_min = travel_min

    def travel_time(self, u: str, v: str) -> float:
        if u == v:
            return 0.0
        key = (u, v)
        if key not in self.travel_min:
            raise KeyError(f"Missing travel time for edge {u}->{v}")
        return float(self.travel_min[key])

def load_pois(path: str) -> Dict[str, POI]:
    pois: Dict[str, POI] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            poi = POI(
                poi_id=row["poi_id"].strip(),
                name=row["name"].strip(),
                lat=float(row["lat"]),
                lon=float(row["lon"]),
                open_min=int(row["open_min"]),
                close_min=int(row["close_min"]),
                service_min=int(row["service_min"]),
            )
            if poi.poi_id in pois:
                raise ValueError(f"Duplicate poi_id: {poi.poi_id}")
            pois[poi.poi_id] = poi
    return pois

def load_time_matrix(path: str) -> Dict[Tuple[str, str], float]:
    t: Dict[Tuple[str, str], float] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            u = row["from_id"].strip()
            v = row["to_id"].strip()
            w = float(row["travel_min"])
            t[(u, v)] = w
    return t
