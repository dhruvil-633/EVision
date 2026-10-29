"""Baseline 2 - rule-based smart charging (crisp heuristic, no fuzzy logic).

A transparent greedy heuristic evaluated slot by slot. For every plugged-in
EV the controller computes a crisp laxity ratio

    required_power_i(t) = remaining_energy_i / (time_left_i * dt * efficiency)
    laxity_i(t)         = required_power_i / charger_rating_i      (0 .. 1+)

laxity >= 1 means the EV must charge at full rate from now on to make its
deadline. The crisp rules are:

  R1  charge now if laxity >= URGENT (deadline pressure), else
  R2  charge now if the slot is cheap (tariff <= 40th percentile), else
  R3  charge now if on-site solar is still unused in this slot, else
  R4  stay idle.

Contention for the shared station limit is resolved in proportion to laxity.
This baseline is a heuristic, NOT a globally optimal schedule.
"""

from __future__ import annotations

import time

import numpy as np

from simulation.constraints import _allocate_under_cap, decode_schedule
from simulation.energy_accounting import simulate

NAME = "Rule-based"
URGENT = 0.55


def run(scenario, urgent_threshold: float = URGENT):
    t0 = time.perf_counter()
    cfg, fleet = scenario.cfg, scenario.fleet
    n, T, dt, eff = fleet.n_evs, cfg.n_slots, cfg.dt, cfg.charging_efficiency
    power = np.zeros((n, T))
    if n == 0:
        return simulate(power, scenario, name=NAME, runtime_s=time.perf_counter() - t0)

    cheap = scenario.tariff <= np.percentile(scenario.tariff, 40)
    avail = fleet.availability()
    remaining = fleet.requested_kwh.copy()

    for t in range(T):
        here = avail[:, t] & (remaining > 1e-9)
        if not here.any():
            continue
        slots_left = np.maximum(fleet.departure_slot - t, 1)
        required_kw = remaining / (slots_left * dt * eff)
        laxity = np.clip(required_kw / fleet.charger_kw, 0.0, 2.0)

        solar_free = scenario.solar_kw[t] > 1e-6
        gate = here & ((laxity >= urgent_threshold) | cheap[t] | solar_free)

        desired = np.where(gate, fleet.charger_kw, 0.0)
        desired = np.minimum(desired, remaining / (dt * eff))
        p = _allocate_under_cap(desired, cfg.station_capacity_kw, weight=laxity)
        power[:, t] = p
        remaining = np.maximum(remaining - p * dt * eff, 0.0)

    power = decode_schedule(power, scenario)        # belt and braces feasibility
    return simulate(power, scenario, name=NAME, runtime_s=time.perf_counter() - t0)
