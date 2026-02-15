import random
from dataclasses import dataclass
from typing import List, Tuple

from src.model.graph import Graph
from src.model.objective import evaluate_route


@dataclass
class GAConfig:
    population_size: int = 60
    generations: int = 150
    crossover_rate: float = 0.9
    mutation_rate: float = 0.2
    tournament_k: int = 3
    seed: int = 123


def _make_individual(rng: random.Random, visit_ids: List[str]) -> List[str]:
    ind = visit_ids[:]
    rng.shuffle(ind)
    return ind


def _tournament_select(rng: random.Random, pop: List[List[str]], fitness: List[float], k: int) -> List[str]:
    best_i = None
    for _ in range(k):
        i = rng.randrange(len(pop))
        if best_i is None or fitness[i] < fitness[best_i]:
            best_i = i
    return pop[best_i][:]  # copy


def _order_crossover(rng: random.Random, p1: List[str], p2: List[str]) -> Tuple[List[str], List[str]]:
    """
    OX (Order Crossover) untuk permutation.
    """
    n = len(p1)
    if n < 2:
        return p1[:], p2[:]

    a = rng.randrange(n)
    b = rng.randrange(n)
    if a > b:
        a, b = b, a

    def ox(parent_a, parent_b):
        child = [None] * n
        # copy slice
        child[a:b+1] = parent_a[a:b+1]
        used = set(child[a:b+1])

        # fill the rest in order from parent_b
        idx = (b + 1) % n
        for gene in parent_b:
            if gene in used:
                continue
            while child[idx] is not None:
                idx = (idx + 1) % n
            child[idx] = gene
            used.add(gene)

        return child

    c1 = ox(p1, p2)
    c2 = ox(p2, p1)
    return c1, c2


def _swap_mutation(rng: random.Random, ind: List[str]) -> None:
    n = len(ind)
    if n < 2:
        return
    i = rng.randrange(n)
    j = rng.randrange(n)
    ind[i], ind[j] = ind[j], ind[i]


def run_ga(
    g: Graph,
    start_id: str,
    end_id: str,
    visit_ids: List[str],
    start_time_min: int,
    late_penalty: float,
    cfg: GAConfig,
) -> Tuple[List[str], float]:
    """
    Return: (best_route_full, best_cost)
    best_route_full = [start] + perm(visit_ids) + [end]
    """
    rng = random.Random(cfg.seed)

    # init population (permutation only)
    pop = [_make_individual(rng, visit_ids) for _ in range(cfg.population_size)]

    def eval_perm(perm: List[str]) -> float:
        route = [start_id] + perm + [end_id]
        res = evaluate_route(g, route, start_time_min=start_time_min, late_penalty=late_penalty)
        return res.total_cost

    fitness = [eval_perm(ind) for ind in pop]

    best_idx = min(range(len(pop)), key=lambda i: fitness[i])
    best_perm = pop[best_idx][:]
    best_cost = fitness[best_idx]

    for gen in range(1, cfg.generations + 1):
        new_pop: List[List[str]] = []

        # elitism: keep best
        new_pop.append(best_perm[:])

        while len(new_pop) < cfg.population_size:
            p1 = _tournament_select(rng, pop, fitness, cfg.tournament_k)
            p2 = _tournament_select(rng, pop, fitness, cfg.tournament_k)

            if rng.random() < cfg.crossover_rate:
                c1, c2 = _order_crossover(rng, p1, p2)
            else:
                c1, c2 = p1[:], p2[:]

            if rng.random() < cfg.mutation_rate:
                _swap_mutation(rng, c1)
            if rng.random() < cfg.mutation_rate:
                _swap_mutation(rng, c2)

            new_pop.append(c1)
            if len(new_pop) < cfg.population_size:
                new_pop.append(c2)

        pop = new_pop
        fitness = [eval_perm(ind) for ind in pop]

        # update best
        gen_best_idx = min(range(len(pop)), key=lambda i: fitness[i])
        gen_best_cost = fitness[gen_best_idx]
        if gen_best_cost < best_cost:
            best_cost = gen_best_cost
            best_perm = pop[gen_best_idx][:]

        # log ringkas tiap beberapa gen (biar tidak spam)
        if gen == 1 or gen % 10 == 0 or gen == cfg.generations:
            avg_cost = sum(fitness) / len(fitness)
            print(f"[GA] gen {gen:3d} | best {best_cost:8.2f} | avg {avg_cost:8.2f}")

    best_route = [start_id] + best_perm + [end_id]
    return best_route, best_cost
