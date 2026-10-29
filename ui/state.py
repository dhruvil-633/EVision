"""Session state: one scenario and one set of results per configuration.

The expensive work (the GA runs) happens only when the configuration changes
or the user asks for it - navigating between pages never re-optimises.
"""

from __future__ import annotations

import streamlit as st

from config import DEFAULT_CONFIG, SimulationConfig
from data.synthetic_data import build_scenario
from evaluation.comparison import STRATEGIES, run_strategy
from evaluation.metrics import Normalizers
from algorithms import uncontrolled

FAST = ("Uncontrolled", "Rule-based")


def init():
    ss = st.session_state
    ss.setdefault("cfg", DEFAULT_CONFIG)
    ss.setdefault("results", {})
    ss.setdefault("active", "Hybrid Fuzzy + GA")
    if "scenario" not in ss:
        ss.scenario = build_scenario(ss.cfg)


def cfg() -> SimulationConfig:
    return st.session_state.cfg


def scenario():
    return st.session_state.scenario


def results() -> dict:
    return st.session_state.results


def apply_config(new_cfg: SimulationConfig, rebuild_scenario: bool = True):
    """Store a new configuration and drop the results it invalidates."""
    ss = st.session_state
    ss.cfg = new_cfg
    if rebuild_scenario:
        ss.scenario = build_scenario(new_cfg)
        ss.results = {}
    else:
        ss.scenario = build_scenario(new_cfg)
        ss.results = {}


def normalizers() -> Normalizers:
    return Normalizers.from_result(uncontrolled.run(scenario()), scenario())


def ensure(names=STRATEGIES, force: bool = False, show_progress: bool = True):
    """Run the requested strategies if they are not cached yet."""
    ss = st.session_state
    todo = [n for n in names if force or n not in ss.results]
    if not todo:
        return ss.results
    norms = normalizers()
    bar = st.progress(0.0) if show_progress else None
    status = st.empty() if show_progress else None

    for k, name in enumerate(todo):
        def cb(gen, total, best, _name=name, _k=k):
            if bar is not None:
                frac = (_k + gen / max(total, 1)) / len(todo)
                bar.progress(min(frac, 1.0))
                status.caption("Optimising %s - generation %d/%d, best J = %.4f"
                               % (_name, gen, total, best))
        if show_progress and name in FAST:
            status.caption("Simulating %s ..." % name)
        ss.results[name] = run_strategy(name, scenario(), norms=norms,
                                        progress=cb if name not in FAST else None)
        if bar is not None:
            bar.progress((k + 1) / len(todo))
    if bar is not None:
        bar.empty()
        status.empty()
    return ss.results


def active_result():
    res = results()
    name = st.session_state.active
    return res.get(name) or res.get("Uncontrolled")


def empty_state(message: str):
    st.info(message, icon=":material/info:")
