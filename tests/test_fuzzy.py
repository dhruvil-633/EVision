"""Fuzzy controller: sanity, coverage, monotonicity and a scikit-fuzzy cross-check."""

import numpy as np
import pytest

from algorithms.fuzzy_controller import (INPUT_TERMS, OUTPUT_TERMS, RULES,
                                         FuzzyController, _mf, fuzzy_inputs,
                                         priority_matrix, trapmf, trimf)


def test_membership_functions_are_bounded_and_peak_at_one():
    x = np.linspace(0, 1, 201)
    for spec in list(INPUT_TERMS.values()) + list(OUTPUT_TERMS.values()):
        mu = _mf(spec, x)
        assert np.all(mu >= 0.0) and np.all(mu <= 1.0)
        assert mu.max() > 0.99                     # every term is reachable
    assert np.isclose(trimf(0.5, 0.0, 0.5, 1.0), 1.0)
    assert np.isclose(trapmf(0.5, 0.0, 0.2, 0.8, 1.0), 1.0)


def test_input_partition_covers_the_whole_universe():
    x = np.linspace(0, 1, 201)
    total = sum(_mf(spec, x) for spec in INPUT_TERMS.values())
    assert np.all(total > 0.0)                     # no gap in the partition


def test_outputs_are_finite_and_in_range_over_the_input_space():
    fc = FuzzyController()
    grid = np.linspace(0, 1, 7)
    u, p, s, h = np.meshgrid(grid, grid, grid, grid, indexing="ij")
    out = fc.evaluate(u.ravel(), p.ravel(), s.ravel(), h.ravel())
    assert np.all(np.isfinite(out))
    assert np.all(out >= 0.0) and np.all(out <= 1.0)
    assert out.std() > 0.05                        # the controller discriminates


def test_priority_increases_with_urgency():
    fc = FuzzyController()
    vals = [fc.evaluate_scalar(u, 0.5, 0.5, 0.5) for u in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert np.all(np.diff(vals) >= -1e-9)          # non-decreasing in urgency
    assert vals[-1] > vals[0] + 0.2


def test_expensive_slots_lower_priority_for_relaxed_evs():
    fc = FuzzyController()
    cheap = fc.evaluate_scalar(0.1, 0.0, 0.0, 0.8)
    pricey = fc.evaluate_scalar(0.1, 1.0, 0.0, 0.8)
    assert cheap > pricey


def test_urgent_ev_keeps_high_priority_even_with_low_headroom():
    fc = FuzzyController()
    assert fc.evaluate_scalar(1.0, 1.0, 0.0, 0.0) > 0.6


def test_explain_returns_a_consistent_trace():
    fc = FuzzyController()
    tr = fc.explain(0.9, 0.2, 0.8, 0.9)
    assert len(tr["rules"]) == len(RULES)
    assert any(r["active"] for r in tr["rules"])
    assert 0.0 <= tr["priority"] <= 1.0
    assert np.isclose(tr["priority"], fc.evaluate_scalar(0.9, 0.2, 0.8, 0.9))


def test_priority_matrix_is_zero_when_the_ev_is_absent(scenario):
    pri = priority_matrix(scenario)
    assert pri.shape == (scenario.fleet.n_evs, scenario.cfg.n_slots)
    assert np.all(pri[~scenario.fleet.availability()] == 0.0)
    assert np.all(np.isfinite(pri))


def test_fuzzy_inputs_are_normalised(scenario):
    sig = fuzzy_inputs(scenario)
    for key in ("urgency", "price", "solar", "headroom"):
        v = sig[key]
        assert np.all(v >= -1e-9) and np.all(v <= 1.0 + 1e-9), key


def test_empty_fleet_gives_empty_priority(small_cfg):
    from data.synthetic_data import build_scenario
    s = build_scenario(small_cfg.with_(n_evs=0))
    assert priority_matrix(s).shape == (0, s.cfg.n_slots)


# ---------------------------------------------------------------------------
# Cross-validation against scikit-fuzzy
# ---------------------------------------------------------------------------
skfuzzy = pytest.importorskip("skfuzzy", reason="scikit-fuzzy not installed")


def _ref_mf(spec, x):
    """Same membership function, built with scikit-fuzzy primitives."""
    kind, p = spec
    return skfuzzy.trimf(x, list(p)) if kind == "tri" else skfuzzy.trapmf(x, list(p))


def _reference_mamdani(u, p, s, h, universe):
    """Independent Mamdani pipeline implemented with scikit-fuzzy."""
    vals = {"urgency": u, "price": p, "solar": s, "headroom": h}
    mu = {(v, t): float(_ref_mf(spec, np.array([vals[v]]))[0])
          for v in vals for t, spec in INPUT_TERMS.items()}
    out_mf = {k: _ref_mf(spec, universe) for k, spec in OUTPUT_TERMS.items()}
    agg = np.zeros_like(universe)
    for ante, cons, _ in RULES:
        strength = min(mu[(v, t)] for v, t in ante.items())
        agg = np.fmax(agg, np.fmin(strength, out_mf[cons]))
    crisp = skfuzzy.defuzz(universe, agg, "centroid") if agg.sum() > 0 else 0.5
    return agg, crisp


def test_membership_functions_match_scikit_fuzzy():
    x = np.linspace(0, 1, 101)
    for spec in list(INPUT_TERMS.values()) + list(OUTPUT_TERMS.values()):
        assert np.allclose(_mf(spec, x), _ref_mf(spec, x), atol=1e-12)


def test_aggregation_matches_scikit_fuzzy_exactly():
    """Fuzzification, rule firing, implication and aggregation are identical."""
    fc = FuzzyController()
    rng = np.random.default_rng(5)
    for _ in range(25):
        u, p, s, h = rng.random(4)
        mu = fc.fuzzify(urgency=[u], price=[p], solar=[s], headroom=[h])
        strengths = fc.rule_strengths(mu, (1,))[:, 0]
        agg = np.zeros_like(fc.universe)
        for r, (_, cons, _) in enumerate(RULES):
            agg = np.fmax(agg, np.fmin(strengths[r], fc._out_mf[cons]))
        ref_agg, _ = _reference_mamdani(u, p, s, h, fc.universe)
        assert np.allclose(agg, ref_agg, atol=1e-12)


def test_defuzzified_output_matches_scikit_fuzzy():
    """Crisp outputs agree with the scikit-fuzzy centroid defuzzifier."""
    fc = FuzzyController()
    rng = np.random.default_rng(6)
    diffs = []
    for _ in range(60):
        u, p, s, h = rng.random(4)
        _, ref = _reference_mamdani(u, p, s, h, fc.universe)
        diffs.append(abs(fc.evaluate_scalar(u, p, s, h) - ref))
    assert max(diffs) < 1e-9, "max deviation %.2e" % max(diffs)
