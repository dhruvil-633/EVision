"""Energy accounting: hand calculation, efficiency, conservation, signs."""

import numpy as np

from simulation.constraints import decode_schedule
from simulation.energy_accounting import simulate

TOL = 1e-9


def test_hand_calculated_example(hand_scenario):
    s = hand_scenario
    # Charge at the full 10 kW for the first two 15-minute slots.
    power = np.array([[10.0, 10.0, 0.0, 0.0]])
    r = simulate(power, s)

    assert np.isclose(r.total_power_kw.sum() * s.cfg.dt, 5.0)        # 5 kWh station side
    assert np.isclose(r.grid_energy_kwh.sum(), 5.0)                  # no solar
    assert np.isclose(r.cost_per_slot.sum(), 50.0)                   # 5 kWh x 10 Rs
    assert np.isclose(r.delivered_kwh[0], 4.5)                       # 5 kWh x 0.90
    assert np.isclose(r.final_soc[0], 0.5 + 4.5 / 50.0)              # 0.59
    assert np.isclose(r.grid_power_kw.max(), 10.0)


def test_efficiency_applies_to_battery_side_only(hand_scenario):
    s = hand_scenario
    power = np.array([[8.0, 0.0, 0.0, 0.0]])
    r = simulate(power, s)
    station_kwh = 8.0 * s.cfg.dt
    assert np.isclose(r.grid_energy_kwh.sum(), station_kwh)
    assert np.isclose(r.delivered_kwh[0], station_kwh * s.cfg.charging_efficiency)


def test_grid_import_never_negative_and_solar_conserved(scenario):
    s = scenario
    power = decode_schedule(s.fleet.max_power(), s)
    r = simulate(power, s)
    assert np.all(r.grid_power_kw >= -TOL)
    # solar is split exactly once into used + curtailed
    assert np.allclose(r.solar_to_ev_kw + r.solar_curtailed_kw, s.solar_kw, atol=1e-9)
    # solar used can exceed neither the resource nor the demand
    assert np.all(r.solar_to_ev_kw <= s.solar_kw + TOL)
    assert np.all(r.solar_to_ev_kw <= r.total_power_kw + TOL)
    # station-side energy balance: charging = solar used + grid import
    assert np.allclose(r.total_power_kw, r.solar_to_ev_kw + r.grid_power_kw, atol=1e-9)


def test_battery_accumulation_matches_power(scenario):
    s = scenario
    power = decode_schedule(s.fleet.max_power() * 0.6, s)
    r = simulate(power, s)
    expected = power.sum(axis=1) * s.cfg.dt * s.cfg.charging_efficiency
    assert np.allclose(r.delivered_kwh, expected, atol=1e-9)
    assert np.allclose(r.battery_cum_kwh[:, -1], expected, atol=1e-9)
    assert np.all(np.diff(r.battery_cum_kwh, axis=1) >= -TOL)        # monotone


def test_zero_schedule_costs_nothing(scenario):
    r = simulate(np.zeros((scenario.fleet.n_evs, scenario.cfg.n_slots)), scenario)
    assert r.cost_per_slot.sum() == 0.0
    assert r.grid_energy_kwh.sum() == 0.0
    assert np.allclose(r.final_soc, scenario.fleet.soc_init)
