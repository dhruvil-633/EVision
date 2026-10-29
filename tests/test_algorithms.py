"""Every strategy must produce a feasible, reproducible, honestly scored schedule."""

import numpy as np

from algorithms import genetic_algorithm as ga
from algorithms import hybrid_optimizer, rule_based, uncontrolled
from algorithms.fuzzy_controller import priority_matrix
from data.synthetic_data import build_scenario
from evaluation.comparison import STRATEGIES, run_all, run_strategy, summary_table
from evaluation.metrics import Normalizers, compute_metrics, objective, objective_batch
from simulation.constraints import decode_population, validate_schedule
from simulation.energy_accounting import simulate


def test_every_strategy_returns_a_feasible_schedule(scenario):
    for name in STRATEGIES:
        res = run_strategy(name, scenario)
        v = validate_schedule(res.power, scenario)
        assert v["feasible"], (name, v)
        assert res.power.shape == (scenario.fleet.n_evs, scenario.cfg.n_slots)
        assert np.all(np.isfinite(res.power))


def test_strategies_deliver_energy(scenario):
    for name in STRATEGIES:
        m = compute_metrics(run_strategy(name, scenario), scenario)
        assert m["delivered_kwh"] > 0.0
        assert m["delivered_kwh"] <= m["requested_kwh"] + 1e-6
        assert m["runtime_s"] >= 0.0


def test_uncontrolled_charges_immediately_on_arrival(scenario):
    res = uncontrolled.run(scenario)
    fleet = scenario.fleet
    for i in range(fleet.n_evs):
        if fleet.requested_kwh[i] > 1e-6:
            assert res.power[i, fleet.arrival_slot[i]] > 0.0


def test_rule_based_shifts_load_away_from_peak_tariff(scenario):
    base = uncontrolled.run(scenario)
    rb = rule_based.run(scenario)
    assert compute_metrics(rb, scenario)["cost_rs"] <= \
        compute_metrics(base, scenario)["cost_rs"] + 1e-6


def test_ga_is_reproducible_under_a_fixed_seed(scenario):
    a = ga.run(scenario, seed=123)
    b = ga.run(scenario, seed=123)
    assert np.allclose(a.power, b.power, atol=1e-12)
    assert a.extras["history"]["best"] == b.extras["history"]["best"]


def test_hybrid_is_reproducible_and_fuzzy_guided(scenario):
    a = hybrid_optimizer.run(scenario, seed=7)
    b = hybrid_optimizer.run(scenario, seed=7)
    assert np.allclose(a.power, b.power, atol=1e-12)
    assert a.extras["fuzzy_guided"] is True
    assert a.extras["priority"].shape == (scenario.fleet.n_evs, scenario.cfg.n_slots)


def test_ga_convergence_history_is_monotone_thanks_to_elitism(scenario):
    res = ga.run(scenario, seed=4)
    best = res.extras["history"]["best"]
    assert len(best) == scenario.cfg.ga_generations + 1
    assert np.all(np.diff(best) <= 1e-12)          # elitism never loses ground
    assert np.all(np.isfinite(best))


def test_ga_fitness_matches_the_reported_schedule(scenario):
    """The GA must score the schedule it actually returns, not a surrogate."""
    norms = ga.default_normalizers(scenario)
    res = ga.run(scenario, seed=9, norms=norms)
    recomputed = objective(simulate(res.power, scenario), scenario, norms)["J"]
    assert np.isclose(recomputed, res.extras["best_J"], atol=1e-9)


def test_batched_objective_equals_per_individual_objective(scenario):
    norms = ga.default_normalizers(scenario)
    rng = np.random.default_rng(2)
    genes = rng.random((6, scenario.fleet.n_evs, scenario.cfg.n_slots))
    pri = priority_matrix(scenario)
    for p in (None, pri):
        power = decode_population(genes, scenario, p)
        batch = objective_batch(power, scenario, norms)
        single = np.array([objective(simulate(x, scenario), scenario, norms)["J"]
                           for x in power])
        assert np.allclose(batch, single, atol=1e-9)


def test_optimisers_beat_the_seeded_baselines_on_the_objective(scenario):
    """Elitism plus baseline seeding: J can never be worse than the seeds."""
    norms = ga.default_normalizers(scenario)
    seeds = {n: objective(run_strategy(n, scenario), scenario, norms)["J"]
             for n in (uncontrolled.NAME, rule_based.NAME)}
    for name in (ga.NAME, hybrid_optimizer.NAME):
        j = objective(run_strategy(name, scenario, norms=norms), scenario, norms)["J"]
        assert j <= min(seeds.values()) + 1e-9, (name, j, seeds)


def test_run_all_covers_every_strategy_and_shares_the_scenario(scenario):
    results = run_all(scenario)
    assert set(results) == set(STRATEGIES)
    table = summary_table(results, scenario)
    assert list(table.index) == STRATEGIES
    assert (table["Constraint violations"] == 0).all()


def test_algorithms_handle_an_empty_fleet(small_cfg):
    s = build_scenario(small_cfg.with_(n_evs=0))
    for name in STRATEGIES:
        res = run_strategy(name, s)
        assert res.power.shape == (0, s.cfg.n_slots)
        m = compute_metrics(res, s)
        assert m["cost_rs"] == 0.0 and m["unmet_kwh"] == 0.0


def test_scenario_generation_is_reproducible(small_cfg):
    a, b = build_scenario(small_cfg), build_scenario(small_cfg)
    assert np.allclose(a.fleet.battery_kwh, b.fleet.battery_kwh)
    assert np.allclose(a.solar_kw, b.solar_kw)
    c = build_scenario(small_cfg.with_(seed=small_cfg.seed + 1))
    assert not np.allclose(a.fleet.battery_kwh, c.fleet.battery_kwh)
