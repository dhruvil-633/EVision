"""Genetic Algorithm scheduler (transparent custom implementation).

Chromosome
----------
A real-valued matrix G of shape (n_evs, n_slots) with every gene in [0, 1].
Gene G[i, t] is the requested fraction of the charger rating of EV i in slot
t, so the decoded desired power is

    desired[i, t] = G[i, t] * charger_rating_i

Feasibility
-----------
A raw chromosome can be physically impossible, so it is NEVER evaluated
directly: simulation.constraints.decode_schedule turns it into a feasible
schedule (availability, charger rating, shared station limit, remaining
battery demand). Fitness is therefore always measured on the SAME schedule
the dashboard displays - there is no surrogate objective.

Operators
---------
selection  tournament (size cfg.ga_tournament)
crossover  uniform gene-wise, probability cfg.ga_crossover_prob
mutation   Gaussian noise on randomly chosen genes, then clipping to [0, 1]
elitism    cfg.ga_elite best individuals are copied unchanged
seeding    one all-ones individual (charge as soon as possible) plus random
           individuals, plus the rule-based baseline schedule; in hybrid
           mode the population is seeded around the fuzzy priority matrix

DEAP note: DEAP is not used because the chromosome is a plain numpy matrix
and the whole population is evaluated with vectorised numpy; the operators
above are implemented explicitly so every step is inspectable from the
dashboard.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np

from algorithms.rule_based import run as run_rule_based
from algorithms.uncontrolled import run as run_uncontrolled
from evaluation.metrics import Normalizers, objective, objective_batch
from simulation.constraints import decode_population, decode_schedule
from simulation.energy_accounting import simulate

NAME = "Genetic Algorithm"


def default_normalizers(scenario) -> Normalizers:
    """Reference values taken from the uncontrolled baseline of the scenario."""
    return Normalizers.from_result(run_uncontrolled(scenario), scenario)


def evaluate_genes(genes, scenario, norms, priority=None):
    """Decode -> simulate -> objective. Returns (J, result, breakdown)."""
    desired = genes * scenario.fleet.charger_kw[:, None]
    power = decode_schedule(desired, scenario, priority)
    result = simulate(power, scenario, name=NAME)
    terms = objective(result, scenario, norms)
    return terms["J"], result, terms


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------
def _tournament(pop, scores, rng, k):
    k = int(np.clip(k, 2, len(pop)))
    idx = rng.choice(len(pop), size=k, replace=False)
    return pop[idx[np.argmin(scores[idx])]]


def _crossover(p1, p2, rng, prob):
    if rng.random() > prob:
        return p1.copy(), p2.copy()
    mask = rng.random(p1.shape) < 0.5                 # uniform gene-wise
    return np.where(mask, p1, p2), np.where(mask, p2, p1)


def _mutate(child, rng, cfg, priority=None):
    mask = rng.random(child.shape) < cfg.ga_mutation_prob
    if mask.any():
        child = child.copy()
        child[mask] += rng.normal(0.0, cfg.ga_mutation_sigma, int(mask.sum()))
        if priority is not None:
            # fuzzy-guided mutation: some mutated genes are pulled towards the
            # fuzzy priority of that (EV, slot) pair
            guide = mask & (rng.random(child.shape) < 0.3)
            child[guide] = 0.5 * child[guide] + 0.5 * priority[guide]
    return np.clip(child, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def run(scenario, priority: Optional[np.ndarray] = None, name: str = NAME,
        norms: Optional[Normalizers] = None, seed: Optional[int] = None,
        progress: Optional[Callable[[int, int, float], None]] = None):
    """Optimise a charging schedule with a GA.

    priority (optional) is the fuzzy priority matrix; passing it switches the
    optimiser into hybrid mode (seeded population, priority-weighted
    contention handling and fuzzy-guided mutation).
    """
    t0 = time.perf_counter()
    cfg, fleet = scenario.cfg, scenario.fleet
    n, T = fleet.n_evs, cfg.n_slots
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    norms = norms or default_normalizers(scenario)

    if n == 0:
        return simulate(np.zeros((0, T)), scenario, name=name,
                        runtime_s=time.perf_counter() - t0,
                        extras={"history": {"best": [], "mean": [], "worst": []},
                                "params": {}, "fuzzy_guided": priority is not None})

    pop_size = max(4, int(cfg.ga_population))
    generations = max(1, int(cfg.ga_generations))
    elite = int(np.clip(cfg.ga_elite, 1, pop_size - 1))

    if priority is None:
        pop = rng.random((pop_size, n, T))
    else:
        base = np.clip(priority, 0.0, 1.0)
        pop = np.clip(base[None, :, :] + rng.normal(0.0, 0.25, (pop_size, n, T)), 0.0, 1.0)
    # Heuristic seeding: the population always contains the two baseline
    # schedules, so the GA result can never be worse (in J) than the better
    # of them, and the search starts from a feasible, energy-complete point.
    pop[0] = 1.0                                   # charge as soon as possible
    pop[1] = np.clip(run_rule_based(scenario).power / fleet.charger_kw[:, None], 0.0, 1.0)

    def fitness(population):
        # Population-level decode + objective: identical numbers to the
        # per-individual path (asserted in tests), roughly 40x faster.
        return objective_batch(decode_population(population, scenario, priority),
                               scenario, norms)

    scores = fitness(pop)
    history = {"best": [], "mean": [], "worst": []}

    for gen in range(generations):
        order = np.argsort(scores)
        pop, scores = pop[order], scores[order]
        history["best"].append(float(scores[0]))
        history["mean"].append(float(scores.mean()))
        history["worst"].append(float(scores[-1]))
        if progress is not None:
            progress(gen + 1, generations, float(scores[0]))

        children = [pop[i].copy() for i in range(elite)]          # elitism
        while len(children) < pop_size:
            p1 = _tournament(pop, scores, rng, cfg.ga_tournament)
            p2 = _tournament(pop, scores, rng, cfg.ga_tournament)
            c1, c2 = _crossover(p1, p2, rng, cfg.ga_crossover_prob)
            for child in (c1, c2):
                if len(children) < pop_size:
                    children.append(_mutate(child, rng, cfg, priority))
        pop = np.array(children)
        scores = fitness(pop)

    order = np.argsort(scores)
    pop, scores = pop[order], scores[order]
    history["best"].append(float(scores[0]))
    history["mean"].append(float(scores.mean()))
    history["worst"].append(float(scores[-1]))

    best_J, result, terms = evaluate_genes(pop[0], scenario, norms, priority)
    result.name = name
    result.runtime_s = time.perf_counter() - t0
    result.extras = {
        "history": history,
        "best_J": best_J,
        "breakdown": {k: v for k, v in terms.items() if k != "metrics"},
        "best_genes": pop[0],
        "fuzzy_guided": priority is not None,
        "params": {"population": pop_size, "generations": generations,
                   "crossover_prob": cfg.ga_crossover_prob,
                   "mutation_prob": cfg.ga_mutation_prob,
                   "mutation_sigma": cfg.ga_mutation_sigma,
                   "elite": elite, "tournament": cfg.ga_tournament,
                   "seed": cfg.seed if seed is None else seed},
    }
    return result
