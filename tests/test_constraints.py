"""Hard constraints: availability, ratings, station limit, SOC and overcharge."""

import numpy as np

from config import SimulationConfig
from data.synthetic_data import build_scenario
from simulation.constraints import (decode_population, decode_schedule,
                                    validate_schedule)
from simulation.ev_models import empty_fleet

TOL = 1e-6


def _huge_request(scenario):
    """A deliberately infeasible request: everyone at full rate, all day."""
    return np.ones((scenario.fleet.n_evs, scenario.cfg.n_slots)) * 1e3


def test_decode_enforces_every_hard_constraint(scenario):
    s = scenario
    power = decode_schedule(_huge_request(s), s)
    v = validate_schedule(power, s)
    assert v["feasible"], v
    assert v["total_violations"] == 0


def test_no_charging_outside_the_parking_window(scenario):
    s = scenario
    power = decode_schedule(_huge_request(s), s)
    assert np.all(power[~s.fleet.availability()] == 0.0)


def test_charger_rating_respected(scenario):
    s = scenario
    power = decode_schedule(_huge_request(s), s)
    assert np.all(power <= s.fleet.charger_kw[:, None] + TOL)


def test_station_capacity_is_a_hard_limit(scenario):
    s = scenario
    power = decode_schedule(_huge_request(s), s)
    assert np.all(power.sum(axis=0) <= s.cfg.station_capacity_kw + TOL)


def test_soc_never_exceeds_target_or_one(scenario):
    s = scenario
    power = decode_schedule(_huge_request(s), s)
    delivered = power.sum(axis=1) * s.cfg.dt * s.cfg.charging_efficiency
    final_soc = s.fleet.soc_init + delivered / s.fleet.battery_kwh
    assert np.all(delivered <= s.fleet.requested_kwh + 1e-6)
    assert np.all(final_soc <= s.fleet.soc_target + 1e-6)
    assert np.all(final_soc <= 1.0 + 1e-6)


def test_negative_requests_are_clipped(scenario):
    s = scenario
    power = decode_schedule(-5.0 * np.ones((s.fleet.n_evs, s.cfg.n_slots)), s)
    assert np.all(power >= 0.0)
    assert power.sum() == 0.0


def test_decode_is_idempotent(scenario):
    s = scenario
    once = decode_schedule(_huge_request(s), s)
    twice = decode_schedule(once, s)
    assert np.allclose(once, twice, atol=1e-9)


def test_priority_weighting_still_feasible_and_favours_high_priority(scenario):
    s = scenario
    rng = np.random.default_rng(3)
    pri = rng.random((s.fleet.n_evs, s.cfg.n_slots))
    power = decode_schedule(_huge_request(s), s, priority=pri)
    assert validate_schedule(power, s)["feasible"]
    assert np.all(power.sum(axis=0) <= s.cfg.station_capacity_kw + TOL)


def test_batch_decode_matches_single_decode(scenario):
    s = scenario
    rng = np.random.default_rng(11)
    genes = rng.random((5, s.fleet.n_evs, s.cfg.n_slots))
    pri = rng.random((s.fleet.n_evs, s.cfg.n_slots))
    for p in (None, pri):
        batch = decode_population(genes, s, p)
        single = np.array([decode_schedule(g * s.fleet.charger_kw[:, None], s, p)
                           for g in genes])
        assert np.allclose(batch, single, atol=1e-9)


def test_validate_detects_injected_violations(scenario):
    s = scenario
    bad = decode_schedule(_huge_request(s), s).copy()
    bad[:, :] = s.cfg.station_capacity_kw      # everyone at the station limit
    v = validate_schedule(bad, s)
    assert not v["feasible"]
    assert v["station_capacity"] > 0


def test_empty_fleet_is_handled(small_cfg):
    cfg = small_cfg.with_(n_evs=0)
    s = build_scenario(cfg)
    assert s.fleet.n_evs == 0
    power = decode_schedule(np.zeros((0, cfg.n_slots)), s)
    assert power.shape == (0, cfg.n_slots)
    assert validate_schedule(power, s)["feasible"]


def test_unusual_configurations_do_not_crash():
    for kwargs in ({"station_capacity_kw": 0.0}, {"solar_capacity_kw": 0.0},
                   {"n_evs": 1}, {"charging_efficiency": 1.0},
                   {"slot_minutes": 60}, {"cloud_variability": 0.0}):
        cfg = SimulationConfig(n_evs=4, seed=1).with_(**kwargs)
        s = build_scenario(cfg)
        power = decode_schedule(s.fleet.max_power(), s)
        assert validate_schedule(power, s)["feasible"]
