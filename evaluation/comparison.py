"""Run every scheduling strategy on one scenario and compare them.

All strategies see the same fleet, solar profile, tariff, station capacity,
efficiency, constraint handling and evaluation formulas, and the GA variants
share the same objective normalizers, so the comparison is apples to apples.
"""

from __future__ import annotations

from typing import Callable, Optional

import pandas as pd

from algorithms import genetic_algorithm as ga
from algorithms import hybrid_optimizer, rule_based, uncontrolled
from evaluation.metrics import Normalizers, compute_metrics, pct_change

BASELINE = uncontrolled.NAME
STRATEGIES = [uncontrolled.NAME, rule_based.NAME, ga.NAME, hybrid_optimizer.NAME]

# Metrics where a LOWER value is better (used for the delta colouring).
LOWER_IS_BETTER = {"cost_rs", "peak_grid_kw", "grid_energy_kwh", "avg_grid_kw",
                   "unmet_kwh", "unmet_pct", "avg_delay_h", "runtime_s"}


def run_strategy(name: str, scenario, norms: Optional[Normalizers] = None,
                 progress: Optional[Callable] = None):
    """Run one strategy by name and return its SimulationResult."""
    if name == uncontrolled.NAME:
        return uncontrolled.run(scenario)
    if name == rule_based.NAME:
        return rule_based.run(scenario)
    if name == ga.NAME:
        return ga.run(scenario, priority=None, norms=norms, progress=progress)
    if name == hybrid_optimizer.NAME:
        return hybrid_optimizer.run(scenario, norms=norms, progress=progress)
    raise ValueError("Unknown strategy: %s" % name)


def run_all(scenario, progress: Optional[Callable] = None) -> dict:
    """Run all four strategies; returns {name: SimulationResult}."""
    base = uncontrolled.run(scenario)
    norms = Normalizers.from_result(base, scenario)
    results = {uncontrolled.NAME: base, rule_based.NAME: rule_based.run(scenario)}

    def sub(label):
        if progress is None:
            return None
        return lambda g, G, best: progress(label, g, G, best)

    results[ga.NAME] = ga.run(scenario, priority=None, norms=norms, progress=sub(ga.NAME))
    results[hybrid_optimizer.NAME] = hybrid_optimizer.run(
        scenario, norms=norms, progress=sub(hybrid_optimizer.NAME))
    return results


def metrics_frame(results: dict, scenario) -> pd.DataFrame:
    """Tidy metrics table, one row per strategy."""
    rows = []
    for name, res in results.items():
        m = compute_metrics(res, scenario)
        m.pop("violation_detail", None)
        m["algorithm"] = name
        rows.append(m)
    df = pd.DataFrame(rows).set_index("algorithm")
    return df.reindex([s for s in STRATEGIES if s in df.index])


def comparison_frame(results: dict, scenario, baseline: str = BASELINE) -> pd.DataFrame:
    """Metrics plus percentage change against a named baseline strategy."""
    df = metrics_frame(results, scenario)
    if baseline not in df.index:
        return df
    base = df.loc[baseline]
    for col in ["cost_rs", "peak_grid_kw", "grid_energy_kwh", "solar_util_pct",
                "unmet_kwh", "target_met_pct", "delivered_kwh"]:
        df[col + "_vs_base_pct"] = [pct_change(v, base[col]) for v in df[col]]
    return df


def summary_table(results: dict, scenario) -> pd.DataFrame:
    """Display-ready table for the dashboard and CSV export."""
    df = comparison_frame(results, scenario)
    out = pd.DataFrame({
        "Electricity cost (Rs)": df["cost_rs"].round(0),
        "Cost vs uncontrolled (%)": df["cost_rs_vs_base_pct"].round(1),
        "Peak grid demand (kW)": df["peak_grid_kw"].round(1),
        "Peak vs uncontrolled (%)": df["peak_grid_kw_vs_base_pct"].round(1),
        "Grid energy (kWh)": df["grid_energy_kwh"].round(1),
        "Solar utilisation (%)": df["solar_util_pct"].round(1),
        "Solar to EV (kWh)": df["solar_to_ev_kwh"].round(1),
        "Energy delivered (kWh)": df["delivered_kwh"].round(1),
        "Unmet energy (kWh)": df["unmet_kwh"].round(1),
        "Unmet energy (%)": df["unmet_pct"].round(2),
        "EVs meeting target (%)": df["target_met_pct"].round(1),
        "Avg charging delay (h)": df["avg_delay_h"].round(2),
        "Constraint violations": df["violations"],
        "Runtime (s)": df["runtime_s"].round(2),
    })
    return out
