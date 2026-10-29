"""Metric definitions, percentage comparisons and edge cases."""

import numpy as np

from algorithms import uncontrolled
from data.synthetic_data import build_scenario, solar_profile, tariff_profile
from evaluation.comparison import run_all, summary_table
from evaluation.metrics import Normalizers, compute_metrics, objective, pct_change
from evaluation.report_generator import build_report, fleet_result_frame, schedule_frame
from simulation.energy_accounting import simulate


def test_hand_calculated_metrics(hand_scenario):
    s = hand_scenario
    r = simulate(np.array([[10.0, 10.0, 0.0, 0.0]]), s)
    m = compute_metrics(r, s)
    assert np.isclose(m["cost_rs"], 50.0)
    assert np.isclose(m["grid_energy_kwh"], 5.0)
    assert np.isclose(m["peak_grid_kw"], 10.0)
    assert np.isclose(m["avg_grid_kw"], 5.0)              # 20 kWh/h over 4 slots
    assert np.isclose(m["delivered_kwh"], 4.5)
    assert np.isclose(m["requested_kwh"], 5.0)
    assert np.isclose(m["unmet_kwh"], 0.5)
    assert np.isclose(m["unmet_pct"], 10.0)
    assert m["target_met_pct"] == 0.0                     # 0.59 < 0.60 target
    assert m["solar_util_pct"] == 0.0                     # no solar available


def test_solar_utilisation_is_usage_not_availability(scenario):
    s = scenario
    idle = compute_metrics(simulate(np.zeros((s.fleet.n_evs, s.cfg.n_slots)), s), s)
    assert idle["available_solar_kwh"] > 0.0
    assert idle["solar_to_ev_kwh"] == 0.0
    assert idle["solar_util_pct"] == 0.0                  # available != utilised

    busy = compute_metrics(uncontrolled.run(s), s)
    assert 0.0 < busy["solar_util_pct"] <= 100.0
    assert np.isclose(busy["solar_util_pct"],
                      100.0 * busy["solar_to_ev_kwh"] / busy["available_solar_kwh"])


def test_zero_solar_scenario_is_safe(small_cfg):
    s = build_scenario(small_cfg.with_(solar_capacity_kw=0.0))
    m = compute_metrics(uncontrolled.run(s), s)
    assert m["available_solar_kwh"] == 0.0
    assert m["solar_util_pct"] == 0.0                     # no division by zero


def test_pct_change_handles_zero_baseline():
    assert pct_change(5.0, 0.0) == 0.0
    assert np.isclose(pct_change(90.0, 100.0), -10.0)
    assert np.isclose(pct_change(120.0, 100.0), 20.0)


def test_target_met_counts_only_vehicles_reaching_target(scenario):
    m = compute_metrics(uncontrolled.run(scenario), scenario)
    assert 0 <= m["target_met_count"] <= m["n_evs"]
    assert np.isclose(m["target_met_pct"], 100.0 * m["target_met_count"] / m["n_evs"])


def test_delay_is_zero_when_charging_starts_immediately(hand_scenario):
    s = hand_scenario
    r = simulate(np.array([[10.0, 0.0, 0.0, 0.0]]), s)
    assert np.isclose(compute_metrics(r, s)["avg_delay_h"], 0.0)
    late = simulate(np.array([[0.0, 0.0, 0.0, 10.0]]), s)
    assert np.isclose(compute_metrics(late, s)["avg_delay_h"], 0.75)


def test_objective_punishes_refusing_to_charge(scenario):
    """An idle station is cheap but must score worse than real charging."""
    s = scenario
    base = uncontrolled.run(s)
    norms = Normalizers.from_result(base, s)
    idle = simulate(np.zeros((s.fleet.n_evs, s.cfg.n_slots)), s, name="idle")
    assert objective(idle, s, norms)["J"] > objective(base, s, norms)["J"]


def test_objective_terms_are_dimensionless_and_add_up(scenario):
    s = scenario
    norms = Normalizers.from_result(uncontrolled.run(s), s)
    terms = objective(uncontrolled.run(s), s, norms)
    parts = [v for k, v in terms.items() if k not in ("J", "metrics")]
    assert np.isclose(terms["J"], sum(parts))
    assert all(np.isfinite(p) for p in parts)


def test_summary_table_and_exports_are_consistent(scenario):
    results = run_all(scenario)
    table = summary_table(results, scenario)
    assert np.isclose(table.loc["Uncontrolled", "Cost vs uncontrolled (%)"], 0.0)
    assert table.notna().all().all()

    best = results["Hybrid Fuzzy + GA"]
    sched = schedule_frame(best, scenario)
    assert not sched.empty
    delivered = sched["Battery energy added (kWh)"].sum()
    assert np.isclose(delivered, best.delivered_kwh.sum(), atol=1e-2)

    fleet_df = fleet_result_frame(best, scenario)
    assert len(fleet_df) == scenario.fleet.n_evs
    assert fleet_df["Target met"].sum() == compute_metrics(best, scenario)["target_met_count"]


def test_report_contains_real_numbers_and_the_disclaimer(scenario):
    results = run_all(scenario)
    report = build_report(scenario, results)
    m = compute_metrics(results["Uncontrolled"], scenario)
    assert "SYNTHETIC-DATA SIMULATION" in report
    assert "%.0f" % m["cost_rs"] in report.replace(",", "")
    for name in results:
        assert name in report


def test_tariff_and_solar_profiles_have_the_expected_shape(small_cfg):
    tariff = tariff_profile(small_cfg)
    assert tariff.min() == small_cfg.tariff_offpeak
    assert tariff.max() == small_cfg.tariff_peak
    solar = solar_profile(small_cfg)
    hours = small_cfg.slot_hours()
    assert np.all(solar[hours < small_cfg.sunrise_hour] == 0.0)
    assert np.all(solar[hours > small_cfg.sunset_hour] == 0.0)
    assert solar.max() <= small_cfg.solar_capacity_kw + 1e-9
