from typing import List, Tuple, Set
from .graph import Graph
from collections import deque

def validate_graph(g: Graph) -> List[str]:
    errors: List[str] = []
    poi_ids = set(g.pois.keys())

    # 1) Cek: semua edge mengarah ke POI yang ada
    for (u, v) in g.travel_min.keys():
        if u not in poi_ids:
            errors.append(f"Edge from_id not found in POI: {u}")
        if v not in poi_ids:
            errors.append(f"Edge to_id not found in POI: {v}")

    # 2) Cek: missing edges (untuk kasus fully-connected seperti dummy M1)
    missing: List[Tuple[str, str]] = []
    for u in poi_ids:
        for v in poi_ids:
            if u == v:
                continue
            if (u, v) not in g.travel_min:
                missing.append((u, v))

    if missing:
        # tampilkan sebagian agar tidak spam
        sample = ", ".join([f"{u}->{v}" for (u, v) in missing[:10]])
        errors.append(f"Missing edges: {len(missing)} (sample: {sample})")

    return errors

def reachable_from(g: Graph, start_id: str) -> Set[str]:
    if start_id not in g.pois:
        raise KeyError(f"Start POI not found: {start_id}")

    # build adjacency from travel_min keys
    adj = {pid: [] for pid in g.pois.keys()}
    for (u, v) in g.travel_min.keys():
        if u in adj:
            adj[u].append(v)

    seen = set([start_id])
    q = deque([start_id])

    while q:
        u = q.popleft()
        for v in adj.get(u, []):
            if v not in seen:
                seen.add(v)
                q.append(v)
    return seen