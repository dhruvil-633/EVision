"""Markdown research report generated from an actual simulation run.

Nothing here is hard-coded: every number is read back from the schedules that
were simulated in this session.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from evaluation.comparison import BASELINE, summary_table
from evaluation.metrics import compute_metrics, pct_change

DISCLAIMER = (
    "All results come from a SYNTHETIC-DATA SIMULATION. Solar generation, "
    "tariffs and EV sessions are produced by configurable mathematical models "
    "and a seeded random number generator. They are not measurements of a real "
    "fleet and are not evidence of real-world savings."
)


def _fmt(x, nd=2):
    return ("%%.%df" % nd) % float(x)


def schedule_frame(result, scenario) -> pd.DataFrame:
    """Long-format charging schedule (one row per EV and active slot)."""
    from simulation.ev_models import hhmm
    cfg, fleet = scenario.cfg, scenario.fleet
    rows = []
    for i in range(fleet.n_evs):
        for t in range(cfg.n_slots):
            p = float(result.power[i, t])
            if p > 1e-6:
                rows.append({
                    "EV ID": fleet.ev_id[i],
                    "Slot": t,
                    "Time": hhmm(t * cfg.dt),
                    "Power (kW)": round(p, 3),
                    "Battery energy added (kWh)":
                        round(p * cfg.dt * cfg.charging_efficiency, 4),
                    "Tariff (Rs/kWh)": float(scenario.tariff[t]),
                })
    return pd.DataFrame(rows)


def fleet_result_frame(result, scenario) -> pd.DataFrame:
    """Per-EV outcome table."""
    cfg, fleet = scenario.cfg, scenario.fleet
    df = fleet.to_frame(cfg.dt)
    df["Delivered (kWh)"] = result.delivered_kwh.round(2)
    df["Final SOC (%)"] = (result.final_soc * 100).round(1)
    df["Unmet (kWh)"] = (fleet.requested_kwh - result.delivered_kwh).clip(min=0).round(2)
    df["Target met"] = result.final_soc >= fleet.soc_target - 1e-3
    return df


def build_report(scenario, results: dict, baseline: str = BASELINE) -> str:
    """Assemble the full Markdown report for the current simulation."""
    cfg = scenario.cfg
    table = summary_table(results, scenario)
    base_m = compute_metrics(results[baseline], scenario) if baseline in results else None
    lines = []
    a = lines.append

    a("# EVision - Intelligent EV Charging and Energy Optimization")
    a("")
    a("_Soft Computing based simulation study - generated %s_"
      % datetime.now().strftime("%Y-%m-%d %H:%M"))
    a("")
    a("> %s" % DISCLAIMER)
    a("")
    a("## 1. Problem statement")
    a("")
    a("A shared charging station with a hard %.0f kW grid connection and a "
      "%.0f kW on-site solar array must serve %d EV charging sessions over a "
      "%.0f hour horizon discretised into %d slots of %d minutes. The scheduler "
      "minimises electricity procurement cost and peak grid demand and maximises "
      "solar self-consumption, while delivering the requested energy before each "
      "departure."
      % (cfg.station_capacity_kw, cfg.solar_capacity_kw, scenario.fleet.n_evs,
         cfg.horizon_hours, cfg.n_slots, cfg.slot_minutes))
    a("")
    a("## 2. Experimental settings")
    a("")
    a("| Setting | Value |")
    a("| --- | --- |")
    for k, v in [
        ("Random seed", cfg.seed),
        ("Fleet size", scenario.fleet.n_evs),
        ("Time slots", "%d x %d min" % (cfg.n_slots, cfg.slot_minutes)),
        ("Station capacity", "%.0f kW" % cfg.station_capacity_kw),
        ("Solar capacity", "%.0f kW" % cfg.solar_capacity_kw),
        ("Charging efficiency", "%.2f" % cfg.charging_efficiency),
        ("Tariff off-peak / standard / peak",
         "%.2f / %.2f / %.2f Rs per kWh"
         % (cfg.tariff_offpeak, cfg.tariff_standard, cfg.tariff_peak)),
        ("Total requested energy", "%.1f kWh" % scenario.fleet.requested_kwh.sum()),
        ("Available solar energy", "%.1f kWh" % scenario.available_solar_kwh),
    ]:
        a("| %s | %s |" % (k, v))
    a("")
    a("## 3. Algorithm configuration")
    a("")
    a("| Parameter | Value |")
    a("| --- | --- |")
    for k, v in [
        ("GA population", cfg.ga_population), ("GA generations", cfg.ga_generations),
        ("Crossover probability", cfg.ga_crossover_prob),
        ("Mutation probability", cfg.ga_mutation_prob),
        ("Mutation sigma", cfg.ga_mutation_sigma), ("Elite individuals", cfg.ga_elite),
        ("Tournament size", cfg.ga_tournament),
        ("Objective weights cost / peak / unmet / delay / solar",
         "%.2f / %.2f / %.2f / %.2f / %.2f"
         % (cfg.w_cost, cfg.w_peak, cfg.w_unmet, cfg.w_delay, cfg.w_solar)),
    ]:
        a("| %s | %s |" % (k, v))
    a("")
    a("## 4. Results")
    a("")
    a(table.to_markdown())
    a("")
    return "\n".join(lines) + _results_discussion(scenario, results, baseline, base_m)


def _results_discussion(scenario, results, baseline, base_m) -> str:
    lines = [""]
    a = lines.append
    if base_m is not None:
        a("### Comparison against the %s baseline" % baseline)
        a("")
        for name, res in results.items():
            if name == baseline:
                continue
            m = compute_metrics(res, scenario)
            a("- **%s**: cost %s %%, peak %s %%, solar utilisation %s pp, "
              "unmet energy %s kWh (baseline %s kWh), EVs meeting target %s %% "
              "(baseline %s %%), runtime %s s."
              % (name,
                 _fmt(pct_change(m["cost_rs"], base_m["cost_rs"]), 1),
                 _fmt(pct_change(m["peak_grid_kw"], base_m["peak_grid_kw"]), 1),
                 _fmt(m["solar_util_pct"] - base_m["solar_util_pct"], 1),
                 _fmt(m["unmet_kwh"], 1), _fmt(base_m["unmet_kwh"], 1),
                 _fmt(m["target_met_pct"], 1), _fmt(base_m["target_met_pct"], 1),
                 _fmt(m["runtime_s"], 2)))
        a("")
    a("## 5. Metric definitions")
    a("")
    for item in [
        "**Electricity cost**: sum over slots of grid energy times the "
        "time-of-use tariff of that slot.",
        "**Peak grid demand**: maximum grid power over the horizon; on-site "
        "solar is self-consumed first.",
        "**Solar utilisation**: solar energy actually used for charging divided "
        "by the solar energy available over the same interval.",
        "**Unmet energy**: requested battery energy not delivered before "
        "departure, summed over all EVs.",
        "**EVs meeting target**: share of vehicles whose final SOC reaches the "
        "requested target SOC.",
        "**Average charging delay**: energy-weighted mean time between plug-in "
        "and actual energy delivery.",
    ]:
        a("- %s" % item)
    a("")
    a("## 6. Limitations")
    a("")
    for item in [
        "Synthetic inputs only; no measured fleet, irradiance or tariff data.",
        "Constant charging efficiency, no battery charging curve or thermal model.",
        "No vehicle-to-grid discharge and no battery degradation cost.",
        "Perfect foresight of arrivals, departures, solar output and tariffs.",
        "The GA is a stochastic heuristic: no global optimality is claimed and "
        "results vary with the random seed.",
        "Fuzzy urgency is evaluated from the plug-in state once per scenario "
        "rather than inside every fitness evaluation, for runtime reasons.",
    ]:
        a("- %s" % item)
    a("")
    a("## 7. Conclusion")
    a("")
    hybrid = "Hybrid Fuzzy + GA"
    if base_m is not None and hybrid in results:
        hm = compute_metrics(results[hybrid], scenario)
        a("On this scenario the hybrid Fuzzy + GA scheduler changed electricity "
          "cost by %s %% and peak grid demand by %s %% relative to uncontrolled "
          "charging, and moved solar utilisation from %s %% to %s %%. Unmet "
          "requested energy was %s kWh against %s kWh for the baseline, so part "
          "of the cost and peak gain is paid for in departure SOC; increasing "
          "the unmet-energy weight in the objective moves the operating point "
          "back towards full deadline compliance."
          % (_fmt(pct_change(hm["cost_rs"], base_m["cost_rs"]), 1),
             _fmt(pct_change(hm["peak_grid_kw"], base_m["peak_grid_kw"]), 1),
             _fmt(base_m["solar_util_pct"], 1), _fmt(hm["solar_util_pct"], 1),
             _fmt(hm["unmet_kwh"], 1), _fmt(base_m["unmet_kwh"], 1)))
    a("")
    a("_%s_" % DISCLAIMER)
    return "\n".join(lines)
