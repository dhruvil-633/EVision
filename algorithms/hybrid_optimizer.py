"""Proposed method - Hybrid Fuzzy Logic + Genetic Algorithm optimiser.

How the two soft computing techniques interact
----------------------------------------------
The Mamdani controller produces a priority matrix ``pi[i, t]`` in [0, 1] for
every (EV, time slot) pair from four linguistic inputs (urgency, price, solar,
headroom). That matrix enters the GA in three concrete places:

1. POPULATION SEEDING - the initial chromosomes are Gaussian perturbations of
   the fuzzy priority matrix instead of uniform noise, so the search starts in
   a region that already respects deadline pressure, tariffs and solar.
2. CONSTRAINT RESOLUTION - when the decoded demand of a slot exceeds the
   shared station limit, capacity is distributed by weighted water filling
   with the fuzzy priorities as weights (instead of fair proportional
   curtailment), so urgent and solar-aligned vehicles keep their power.
3. GUIDED MUTATION - a share of mutated genes is pulled half-way towards the
   fuzzy priority of their (EV, slot) pair, which keeps the search anchored to
   domain knowledge instead of drifting randomly.

The fitness function itself is unchanged, so hybrid and plain GA are compared
on exactly the same objective, scenario and constraints. No claim of global
optimality is made: both are stochastic heuristics.
"""

from __future__ import annotations

from typing import Callable, Optional

from algorithms import genetic_algorithm as ga
from algorithms.fuzzy_controller import FuzzyController, priority_matrix

NAME = "Hybrid Fuzzy + GA"


def run(scenario, controller: Optional[FuzzyController] = None,
        norms=None, seed: Optional[int] = None,
        progress: Optional[Callable[[int, int, float], None]] = None):
    """Run the fuzzy-guided GA and attach the fuzzy priority matrix."""
    priority = priority_matrix(scenario, controller or FuzzyController())
    result = ga.run(scenario, priority=priority, name=NAME, norms=norms,
                    seed=seed, progress=progress)
    result.extras["priority"] = priority
    return result
