"""Hard constraint handling: feasible schedule decoding and validation.

A charging schedule is a matrix ``P`` of shape (n_evs, n_slots) holding the
STATION-SIDE power (kW) delivered to each EV in each slot.

Hard constraints (all enforced by construction in :func:`decode_schedule`,
never merely penalised):

  C1  P[i, t] = 0            outside [arrival_i, departure_i)
  C2  0 <= P[i, t] <= charger_rating_i
  C3  sum_i P[i, t] <= station_capacity_kw                 (shared limit)
  C4  cumulative battery energy of EV i <= requested_i     (no overcharge,
      therefore SOC_i(t) <= SOC_target_i <= 1.0)
  C5  only EVs that still need energy receive power

Assumptions (v1): constant charging efficiency, no vehicle-to-grid discharge,
no per-charger ramp limits.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

TOL = 1e-6


def _allocate_under_cap(desired: np.ndarray, limit: float,
                        weight: Optional[np.ndarray] = None) -> np.ndarray:
    """Scale a vector of desired powers down to respect a shared ``limit``.

    ``weight is None``  -> proportional (fair) curtailment.
    ``weight`` given    -> weighted water filling: capacity is handed out in
    proportion to the weights, individual demands act as caps and any capacity
    left over by a satisfied EV is redistributed to the remaining ones. This is
    the hook through which fuzzy priorities steer the physical allocation.
    """
    total = float(desired.sum())
    if total <= limit + TOL or total <= TOL:
        return desired
    if weight is None:
        return desired * (limit / total)

    alloc = np.zeros_like(desired)
    active = desired > TOL
    w = np.where(active, np.maximum(weight, 1e-3), 0.0)
    remaining = limit
    for _ in range(12):
        wsum = w[active].sum()
        if remaining <= TOL or not active.any() or wsum <= 0:
            break
        share = np.zeros_like(desired)
        share[active] = remaining * w[active] / wsum
        take = np.minimum(share, desired - alloc)
        take = np.where(active, take, 0.0)
        moved = float(take.sum())
        alloc += take
        remaining -= moved
        active = active & (alloc < desired - TOL)
        if moved <= TOL:
            break
    return alloc


def decode_schedule(desired: np.ndarray, scenario, priority: Optional[np.ndarray] = None
                    ) -> np.ndarray:
    """Map any (feasible or not) desired power matrix to a feasible schedule.

    Parameters
    ----------
    desired : (n, T) array of requested station-side power in kW.
    scenario : :class:`data.synthetic_data.Scenario`.
    priority : optional (n, T) array in [0, 1] used to break contention for the
        shared station capacity (higher = served first).

    Returns
    -------
    (n, T) feasible schedule in kW.
    """
    cfg, fleet = scenario.cfg, scenario.fleet
    n, T, dt = fleet.n_evs, cfg.n_slots, cfg.dt
    out = np.zeros((n, T), dtype=float)
    if n == 0:
        return out

    pmax = fleet.max_power()                       # C1 + C2
    desired = np.clip(np.asarray(desired, dtype=float), 0.0, None)
    desired = np.minimum(desired, pmax)
    remaining = fleet.requested_kwh.copy()         # battery-side kWh still wanted
    eff = cfg.charging_efficiency

    for t in range(T):
        d = desired[:, t]
        if not np.any(d > TOL):
            continue
        # C4 + C5: station-side power that exactly fills the remaining request
        cap = remaining / (dt * eff)
        d = np.minimum(d, np.maximum(cap, 0.0))
        # C3: shared station power limit
        w = None if priority is None else priority[:, t]
        p = _allocate_under_cap(d, cfg.station_capacity_kw, w)
        out[:, t] = p
        remaining -= p * dt * eff
        remaining = np.maximum(remaining, 0.0)
    return out


def validate_schedule(power: np.ndarray, scenario) -> dict:
    """Count violations of every hard constraint. All counts must be zero."""
    cfg, fleet = scenario.cfg, scenario.fleet
    dt, eff = cfg.dt, cfg.charging_efficiency
    tol = 1e-5
    if fleet.n_evs == 0:
        return {"negative_power": 0, "availability": 0, "charger_rating": 0,
                "station_capacity": 0, "overcharge": 0, "soc_above_one": 0,
                "total_violations": 0, "feasible": True}

    pmax = fleet.max_power()
    delivered = power.sum(axis=1) * dt * eff
    final_soc = fleet.soc_init + delivered / fleet.battery_kwh

    v = {
        "negative_power": int(np.sum(power < -tol)),
        "availability": int(np.sum((power > tol) & (~fleet.availability()))),
        "charger_rating": int(np.sum(power > pmax + tol)),
        "station_capacity": int(np.sum(power.sum(axis=0) > cfg.station_capacity_kw + tol)),
        "overcharge": int(np.sum(delivered > fleet.requested_kwh + tol)),
        "soc_above_one": int(np.sum(final_soc > 1.0 + tol)),
    }
    v["total_violations"] = int(sum(v.values()))
    v["feasible"] = v["total_violations"] == 0
    return v


# ---------------------------------------------------------------------------
# Population-level decoding (used by the GA for speed)
# ---------------------------------------------------------------------------
def _allocate_under_cap_batch(desired: np.ndarray, limit: float,
                              weight: Optional[np.ndarray] = None) -> np.ndarray:
    """Vectorised twin of :func:`_allocate_under_cap` for a whole population.

    ``desired`` has shape (P, n) - one row per individual. Produces exactly the
    same numbers as the per-individual function (verified in
    tests/test_algorithms.py).
    """
    total = desired.sum(axis=1)
    over = total > limit + TOL
    if not np.any(over):
        return desired
    if weight is None:
        scale = np.where(over, limit / np.maximum(total, TOL), 1.0)
        return desired * scale[:, None]

    alloc = np.where(over[:, None], 0.0, desired)
    active = over[:, None] & (desired > TOL)
    w = np.where(active, np.maximum(np.broadcast_to(weight, desired.shape), 1e-3), 0.0)
    remaining = np.where(over, limit, 0.0)
    for _ in range(12):
        wsum = w.sum(axis=1)
        live = (remaining > TOL) & (wsum > 0) & active.any(axis=1)
        if not np.any(live):
            break
        share = np.zeros_like(desired)
        np.divide(w * remaining[:, None], np.maximum(wsum, 1e-12)[:, None],
                  out=share, where=active)
        take = np.where(active, np.minimum(share, desired - alloc), 0.0)
        take = np.where(live[:, None], take, 0.0)
        moved = take.sum(axis=1)
        alloc += take
        remaining -= moved
        active = active & (alloc < desired - TOL)
        w = np.where(active, w, 0.0)
        if moved.max() <= TOL:
            break
    return alloc


def decode_population(genes: np.ndarray, scenario,
                      priority: Optional[np.ndarray] = None) -> np.ndarray:
    """Decode a whole GA population at once.

    ``genes`` has shape (P, n, T) with values in [0, 1]; returns (P, n, T)
    feasible station-side power in kW.
    """
    cfg, fleet = scenario.cfg, scenario.fleet
    P, n, T = genes.shape
    out = np.zeros((P, n, T))
    if n == 0:
        return out

    pmax = fleet.max_power()[None, :, :]
    desired = np.minimum(np.clip(genes, 0.0, 1.0) * fleet.charger_kw[None, :, None], pmax)
    remaining = np.tile(fleet.requested_kwh, (P, 1))
    dt, eff = cfg.dt, cfg.charging_efficiency

    for t in range(T):
        d = np.minimum(desired[:, :, t], remaining / (dt * eff))
        w = None if priority is None else priority[:, t]
        p = _allocate_under_cap_batch(d, cfg.station_capacity_kw, w)
        out[:, :, t] = p
        remaining = np.maximum(remaining - p * dt * eff, 0.0)
    return out
