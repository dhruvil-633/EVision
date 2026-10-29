"""Page 4 - Algorithm laboratory: tune the optimiser and run experiments."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from evaluation.comparison import STRATEGIES, summary_table
from ui import state, theme

PRESETS = {
    "Balanced (default)": dict(w_cost=1.0, w_peak=0.8, w_unmet=8.0, w_delay=0.3, w_solar=0.6),
    "Cost first": dict(w_cost=1.6, w_peak=0.5, w_unmet=4.0, w_delay=0.2, w_solar=0.8),
    "Deadline first": dict(w_cost=1.0, w_peak=0.8, w_unmet=40.0, w_delay=0.3, w_solar=0.6),
    "Peak shaving": dict(w_cost=0.8, w_peak=2.5, w_unmet=8.0, w_delay=0.3, w_solar=0.6),
}


def render():
    theme.hero("Algorithm Laboratory",
               "Tune the genetic algorithm, the objective weights and the site, "
               "then rerun the experiment")
    cfg = state.cfg()

    with st.form("algo_form", border=False):
        st.markdown("##### Genetic algorithm")
        a = st.columns(4)
        pop = a[0].slider("Population size", 8, 200, cfg.ga_population, 4)
        gens = a[1].slider("Generations", 5, 300, cfg.ga_generations, 5)
        cx = a[2].slider("Crossover probability", 0.0, 1.0, cfg.ga_crossover_prob, 0.05)
        mut = a[3].slider("Mutation probability", 0.0, 0.6, cfg.ga_mutation_prob, 0.01)
        b = st.columns(4)
        sigma = b[0].slider("Mutation sigma", 0.05, 0.8, cfg.ga_mutation_sigma, 0.05)
        elite = b[1].slider("Elite individuals", 1, 10, cfg.ga_elite, 1)
        tour = b[2].slider("Tournament size", 2, 8, cfg.ga_tournament, 1)
        b[3].caption("Chromosome: %d EVs x %d slots = %d genes"
                     % (state.scenario().fleet.n_evs, cfg.n_slots,
                        state.scenario().fleet.n_evs * cfg.n_slots))

        st.markdown("##### Objective weights")
        preset = st.selectbox("Weight preset", ["Custom"] + list(PRESETS), index=0,
                              help="Presets overwrite the sliders below when applied.")
        w = st.columns(5)
        w_cost = w[0].slider("Cost", 0.0, 3.0, cfg.w_cost, 0.1)
        w_peak = w[1].slider("Peak demand", 0.0, 3.0, cfg.w_peak, 0.1)
        w_unmet = w[2].slider("Unmet energy", 0.0, 60.0, cfg.w_unmet, 0.5)
        w_delay = w[3].slider("Charging delay", 0.0, 2.0, cfg.w_delay, 0.1)
        w_solar = w[4].slider("Solar reward", 0.0, 3.0, cfg.w_solar, 0.1)

        st.markdown("##### Site")
        s = st.columns(3)
        station = s[0].slider("Station capacity (kW)", 20, 500,
                              int(cfg.station_capacity_kw), 10)
        solar = s[1].slider("Solar capacity (kW)", 0, 400, int(cfg.solar_capacity_kw), 10)
        peak_tariff = s[2].slider("Peak tariff (Rs/kWh)", 5.0, 25.0,
                                  float(cfg.tariff_peak), 0.5)

        run = st.form_submit_button("Run simulation", type="primary",
                                    use_container_width=True)

    if run:
        weights = PRESETS.get(preset, dict(w_cost=w_cost, w_peak=w_peak, w_unmet=w_unmet,
                                           w_delay=w_delay, w_solar=w_solar))
        state.apply_config(cfg.with_(
            ga_population=int(pop), ga_generations=int(gens), ga_crossover_prob=float(cx),
            ga_mutation_prob=float(mut), ga_mutation_sigma=float(sigma),
            ga_elite=int(elite), ga_tournament=int(tour),
            station_capacity_kw=float(station), solar_capacity_kw=float(solar),
            tariff_peak=float(peak_tariff), **weights))
        st.toast("Configuration applied - rerunning all four strategies")

    if state.scenario().fleet.n_evs == 0:
        theme.note("The fleet is empty, so there is nothing to optimise.")
        return

    results = state.ensure()
    st.markdown("##### Objective value by generation")
    st.plotly_chart(_convergence_chart(results), use_container_width=True, theme=None)

    st.markdown("##### All strategies under the current scenario")
    st.dataframe(summary_table(results, state.scenario()), use_container_width=True)

    theme.info("The objective is J = w1*cost/cost_ref + w2*peak/peak_ref + "
               "w3*unmet/requested + w4*delay - w5*solar_used/solar_available, with "
               "cost_ref and peak_ref taken from the uncontrolled baseline of the same "
               "scenario. Lower J is better. The unmet-energy weight is deliberately "
               "large so that the optimiser cannot win by refusing to charge.")


def _convergence_chart(results):
    fig = go.Figure()
    for name in STRATEGIES:
        res = results.get(name)
        hist = (res.extras or {}).get("history") if res is not None else None
        if not hist or not hist.get("best"):
            continue
        gens = list(range(len(hist["best"])))
        color = theme.STRATEGY_COLORS[name]
        fig.add_trace(go.Scatter(x=gens, y=hist["best"], name="%s - best" % name,
                                 line=dict(color=color, width=2.5)))
        fig.add_trace(go.Scatter(x=gens, y=hist["mean"], name="%s - population mean" % name,
                                 line=dict(color=color, width=1, dash="dot"),
                                 opacity=0.6))
    if not fig.data:
        fig.add_annotation(text="No evolutionary run in the current results",
                           showarrow=False)
    fig.update_layout(height=360, xaxis_title="Generation",
                      yaxis_title="Objective J (lower is better)")
    return fig
