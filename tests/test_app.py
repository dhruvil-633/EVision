"""End-to-end application test: the Streamlit app must start and simulate."""

import pathlib

import pytest

pytest.importorskip("streamlit.testing.v1", reason="Streamlit AppTest unavailable")

from streamlit.testing.v1 import AppTest   # noqa: E402


APP = str(pathlib.Path(__file__).resolve().parent.parent / "app.py")


def _app():
    at = AppTest.from_file(APP, default_timeout=300)
    at.session_state["cfg"] = _small_cfg()
    return at


def _small_cfg():
    from config import SimulationConfig
    return SimulationConfig(n_evs=6, seed=3, ga_population=8, ga_generations=4)


def test_app_starts_without_exceptions():
    at = _app().run()
    assert not at.exception, at.exception


def test_dashboard_runs_a_full_simulation():
    at = _app().run()
    assert not at.exception
    results = at.session_state["results"]
    assert set(results) >= {"Uncontrolled", "Rule-based",
                            "Genetic Algorithm", "Hybrid Fuzzy + GA"}
    assert any("Executive Dashboard" in m.value for m in at.markdown
               if isinstance(m.value, str))


def test_every_page_renders():
    pages = ["EV Fleet", "Energy Simulation", "Algorithm Laboratory",
             "Performance Comparison", "Soft Computing Explainability",
             "Research Report"]
    for page in pages:
        at = _app().run()
        at.radio[0].set_value(page).run()
        assert not at.exception, (page, at.exception)


def test_algorithm_laboratory_reruns_the_experiment():
    at = _app().run()
    at.radio[0].set_value("Algorithm Laboratory").run()
    assert not at.exception

    before = at.session_state["results"]["Hybrid Fuzzy + GA"].power.copy()
    # change the generation count, then submit the form
    gens = [s for s in at.slider if s.label == "Generations"][0]
    gens.set_value(8)
    at.button[0].click().run()

    assert not at.exception
    cfg = at.session_state["cfg"]
    assert cfg.ga_generations == 8
    after = at.session_state["results"]["Hybrid Fuzzy + GA"].power
    assert after.shape == before.shape
