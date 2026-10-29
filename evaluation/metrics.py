"""Evaluation metrics and the optimisation objective.

Every metric is computed from an actually simulated schedule - nothing is
hard-coded or assumed. Definitions:

cost_rs            Rs      sum over slots of grid_energy * tariff
grid_energy_kwh    kWh     energy imported from the grid
peak_grid_kw       kW      max over slots of grid power
avg_grid_kw        kW      mean grid power over the whole horizon
solar_to_ev_kwh    kWh     solar energy actually used for charging
solar_util_pct     %       solar_to_ev_kwh / available solar energy * 100
                           (availability is NOT utilisation; 0 solar -> 0 %)
delivered_kwh      kWh     battery-side energy stored in all EVs
requested_kwh      kWh     battery-side energy needed to reach target SOC
unmet_kwh          kWh     max(0, requested - delivered), summed over EVs
target_met_pct     %       share of EVs reaching target SOC before departure
avg_delay_h        h       energy-weighted mean delay between plug-in and the
                           moment energy is actually delivered
delay_norm         0..1    the same delay divided by each dwell time
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from simulation.constraints import validate_schedule

SOC_TOL = 1e-3


def compute_metrics(result, scenario) -> dict:
    cfg, fleet = scenario.cfg, scenario.fleet
    dt = cfg.dt
    n = fleet.n_evs

    requested = float(fleet.requested_kwh.sum()) if n else 0.0
    delivered = float(result.delivered_kwh.sum()) if n else 0.0
    unmet_per_ev = (np.maximum(fleet.requested_kwh - result.delivered_kwh, 0.0)
                    if n else np.zeros(0))
    unmet = float(unmet_per_ev.sum())

    target_met = (result.final_soc >= fleet.soc_target - SOC_TOL) if n else np.zeros(0, bool)
    available_solar = float(scenario.solar_kw.sum() * dt)
    solar_used = float(result.solar_to_ev_kw.sum() * dt)

    # energy-weighted charging delay
    energy = result.power * dt * cfg.charging_efficiency if n else np.zeros((0, cfg.n_slots))
    if n:
        slot_idx = np.arange(cfg.n_slots)[None, :]
        lag_h = np.maximum(slot_idx - fleet.arrival_slot[:, None], 0) * dt
        per_ev_energy = energy.sum(axis=1)
        active = per_ev_energy > 1e-9
        delay_h = np.zeros(n)
        delay_h[active] = (energy[active] * lag_h[active]).sum(axis=1) / per_ev_energy[active]
        dwell_h = np.maximum(fleet.dwell_slots() * dt, dt)
        delay_norm = float(np.mean(delay_h[active] / dwell_h[active])) if active.any() else 0.0
        avg_delay_h = float(delay_h[active].mean()) if active.any() else 0.0
    else:
        delay_norm = avg_delay_h = 0.0

    violations = validate_schedule(result.power, scenario)
    cost = float(result.cost_per_slot.sum())

    return {
        "algorithm": result.name,
        "cost_rs": cost,
        "grid_energy_kwh": float(result.grid_energy_kwh.sum()),
        "peak_grid_kw": float(result.grid_power_kw.max()) if cfg.n_slots else 0.0,
        "avg_grid_kw": float(result.grid_power_kw.mean()) if cfg.n_slots else 0.0,
        "solar_to_ev_kwh": solar_used,
        "available_solar_kwh": available_solar,
        "solar_util_pct": 100.0 * solar_used / available_solar if available_solar > 1e-9 else 0.0,
        "delivered_kwh": delivered,
        "requested_kwh": requested,
        "unmet_kwh": unmet,
        "unmet_pct": 100.0 * unmet / requested if requested > 1e-9 else 0.0,
        "target_met_pct": 100.0 * float(np.mean(target_met)) if n else 0.0,
        "target_met_count": int(np.sum(target_met)) if n else 0,
        "n_evs": n,
        "avg_delay_h": avg_delay_h,
        "delay_norm": delay_norm,
        "cost_per_kwh_delivered": cost / delivered if delivered > 1e-9 else 0.0,
        "peak_station_kw": float(result.total_power_kw.max()) if cfg.n_slots else 0.0,
        "violations": violations["total_violations"],
        "feasible": bool(violations["feasible"]),
        "violation_detail": violations,
        "runtime_s": float(result.runtime_s),
    }


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------
@dataclass
class Normalizers:
    """Scenario-level reference values that make the objective dimensionless.

    Cost and peak are normalised by the UNCONTROLLED baseline, so J is read as
    "fraction of the naive cost / peak". Unmet energy is normalised by the
    total requested energy and solar reward by the available solar energy.
    """
    cost_ref: float
    peak_ref: float
    requested_kwh: float
    solar_ref: float

    @staticmethod
    def from_result(result, scenario) -> "Normalizers":
        m = compute_metrics(result, scenario)
        return Normalizers(
            cost_ref=max(m["cost_rs"], 1e-6),
            peak_ref=max(m["peak_grid_kw"], 1e-6),
            requested_kwh=max(m["requested_kwh"], 1e-6),
            solar_ref=max(m["available_solar_kwh"], 1e-6),
        )


def objective(result, scenario, norms: Normalizers) -> dict:
    """Weighted multi-objective cost J (lower is better) and its breakdown.

        J = w1*cost/cost_ref + w2*peak/peak_ref + w3*unmet/requested
            + w4*delay_norm - w5*solar_used/solar_available

    The unmet-energy weight dominates by default so that the optimiser can
    never improve J simply by refusing to charge vehicles.
    """
    cfg = scenario.cfg
    m = compute_metrics(result, scenario)
    terms = {
        "cost": cfg.w_cost * m["cost_rs"] / norms.cost_ref,
        "peak": cfg.w_peak * m["peak_grid_kw"] / norms.peak_ref,
        "unmet": cfg.w_unmet * m["unmet_kwh"] / norms.requested_kwh,
        "delay": cfg.w_delay * m["delay_norm"],
        "solar_reward": -cfg.w_solar * m["solar_to_ev_kwh"] / norms.solar_ref,
    }
    terms["J"] = float(sum(terms.values()))
    terms["metrics"] = m
    return terms


def pct_change(value: float, baseline: float) -> float:
    """Percentage change of ``value`` relative to ``baseline`` (safe on zero)."""
    if abs(baseline) < 1e-9:
        return 0.0
    return 100.0 * (value - baseline) / baseline


# ---------------------------------------------------------------------------
# Batched objective (same formula, whole GA population at once)
# ---------------------------------------------------------------------------
def objective_batch(power_pop: np.ndarray, scenario, norms: Normalizers) -> np.ndarray:
    """Vectorised objective for a (P, n, T) stack of feasible schedules.

    Produces exactly the same numbers as :func:`objective` applied to each
    individual - this equivalence is asserted in tests/test_algorithms.py, so
    the GA always optimises the metric that the dashboard reports.
    """
    cfg, fleet = scenario.cfg, scenario.fleet
    dt, eff = cfg.dt, cfg.charging_efficiency
    P, n, T = power_pop.shape
    if n == 0:
        return np.zeros(P)

    total = power_pop.sum(axis=1)                              # (P, T)
    solar_to_ev = np.minimum(scenario.solar_kw[None, :], total)
    grid = np.maximum(total - solar_to_ev, 0.0)
    cost = (grid * dt * scenario.tariff[None, :]).sum(axis=1)
    peak = grid.max(axis=1)
    solar_used = solar_to_ev.sum(axis=1) * dt

    energy = power_pop * dt * eff                              # (P, n, T)
    delivered = energy.sum(axis=2)                             # (P, n)
    unmet = np.maximum(fleet.requested_kwh[None, :] - delivered, 0.0).sum(axis=1)

    lag_h = np.maximum(np.arange(T)[None, :] - fleet.arrival_slot[:, None], 0) * dt
    active = delivered > 1e-9
    with np.errstate(invalid="ignore", divide="ignore"):
        delay_h = np.where(active, (energy * lag_h[None, :, :]).sum(axis=2) /
                           np.maximum(delivered, 1e-12), 0.0)
    dwell_h = np.maximum(fleet.dwell_slots() * dt, dt)[None, :]
    n_active = active.sum(axis=1)
    delay_norm = np.where(n_active > 0,
                          (np.where(active, delay_h / dwell_h, 0.0)).sum(axis=1) /
                          np.maximum(n_active, 1), 0.0)

    return (cfg.w_cost * cost / norms.cost_ref
            + cfg.w_peak * peak / norms.peak_ref
            + cfg.w_unmet * unmet / norms.requested_kwh
            + cfg.w_delay * delay_norm
            - cfg.w_solar * solar_used / norms.solar_ref)
