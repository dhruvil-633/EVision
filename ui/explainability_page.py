"""Page 6 - Soft computing explainability: fuzzy inference and GA internals."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from algorithms.fuzzy_controller import (INPUT_TERMS, OUTPUT_TERMS, RULES,
                                         FuzzyController, _mf, fuzzy_inputs,
                                         priority_matrix)
from ui import state, theme

HYBRID = "Hybrid Fuzzy + GA"


def render():
    theme.hero("Soft Computing Explainability",
               "What the fuzzy controller infers and what the genetic algorithm evolves")
    sc = state.scenario()
    tab_f, tab_g, tab_link = st.tabs(["Fuzzy logic controller",
                                      "Genetic algorithm",
                                      "How they interact"])
    with tab_f:
        _fuzzy_section(sc)
    with tab_g:
        _ga_section(sc)
    with tab_link:
        _interaction_section()


# ---------------------------------------------------------------------------
# Fuzzy
# ---------------------------------------------------------------------------
def _fuzzy_section(sc):
    fc = FuzzyController()
    st.markdown("##### Membership functions")
    a, b = st.columns(2)
    with a:
        st.plotly_chart(_mf_chart(INPUT_TERMS, "Input variables (urgency, price, "
                                               "solar, headroom)"),
                        use_container_width=True)
    with b:
        st.plotly_chart(_mf_chart(OUTPUT_TERMS, "Output variable (charging priority)"),
                        use_container_width=True)

    st.markdown("##### Live inference")
    c = st.columns(4)
    u = c[0].slider("Urgency", 0.0, 1.0, 0.8, 0.01)
    p = c[1].slider("Electricity price", 0.0, 1.0, 0.2, 0.01)
    s = c[2].slider("Solar availability", 0.0, 1.0, 0.7, 0.01)
    h = c[3].slider("Grid headroom", 0.0, 1.0, 0.6, 0.01)

    trace = fc.explain(u, p, s, h)
    k = st.columns([1, 3])
    with k[0]:
        theme.metric_card("Defuzzified priority", "%.3f" % trace["priority"],
                          hint="centroid of the aggregated output")
        active = sum(1 for r in trace["rules"] if r["active"])
        theme.metric_card("Active rules", "%d of %d" % (active, len(RULES)))
    with k[1]:
        st.plotly_chart(_aggregation_chart(fc, trace), use_container_width=True, theme=None)

    st.markdown("###### Rule base and firing strengths")
    rules_df = pd.DataFrame(trace["rules"])[
        ["rule", "antecedent", "consequent", "strength", "description"]]
    rules_df.columns = ["#", "IF", "THEN", "Firing strength", "Interpretation"]
    st.dataframe(rules_df.sort_values("Firing strength", ascending=False),
                 use_container_width=True, hide_index=True, height=330,
                 column_config={"Firing strength": st.column_config.ProgressColumn(
                     "Firing strength", min_value=0.0, max_value=1.0, format="%.3f")})

    mem = pd.DataFrame({"Term": list(trace["memberships"]),
                        "Membership": list(trace["memberships"].values())})
    with st.expander("Fuzzified input memberships"):
        st.dataframe(mem[mem["Membership"] > 0], use_container_width=True,
                     hide_index=True)

    if sc.fleet.n_evs:
        st.markdown("##### Fuzzy priority of the current fleet")
        pri = priority_matrix(sc)
        sig = fuzzy_inputs(sc)
        avail = sc.fleet.availability()
        mean_pri = np.divide(pri.sum(axis=1), np.maximum(avail.sum(axis=1), 1))
        rank = pd.DataFrame({
            "EV ID": sc.fleet.ev_id,
            "Requested (kWh)": sc.fleet.requested_kwh.round(2),
            "Dwell (h)": (sc.fleet.dwell_slots() * sc.cfg.dt).round(2),
            "Peak urgency": sig["urgency"].max(axis=1).round(3),
            "Mean fuzzy priority": mean_pri.round(3),
        }).sort_values("Mean fuzzy priority", ascending=False)
        a, b = st.columns([2, 3])
        with a:
            st.dataframe(rank, use_container_width=True, hide_index=True, height=360)
        with b:
            st.plotly_chart(_priority_heatmap(sc, pri), use_container_width=True, theme=None)


def _mf_chart(terms, title):
    x = np.linspace(0, 1, 201)
    fig = go.Figure()
    for name, spec in terms.items():
        fig.add_trace(go.Scatter(x=x, y=_mf(spec, x), name=name.replace("_", " "),
                                 mode="lines", line=dict(width=2.5)))
    fig.update_layout(height=330, title=dict(text=title, y=0.97), xaxis_title="Normalised value",
                      yaxis_title="Membership", yaxis=dict(range=[0, 1.05]))
    return fig


def _aggregation_chart(fc, trace):
    mu = fc.fuzzify(**{k: [v] for k, v in trace["inputs"].items()})
    strengths = fc.rule_strengths(mu, (1,))[:, 0]
    agg = np.zeros_like(fc.universe)
    for r, (_, cons, _) in enumerate(RULES):
        agg = np.maximum(agg, np.minimum(strengths[r], fc._out_mf[cons]))
    fig = go.Figure()
    for name, spec in OUTPUT_TERMS.items():
        fig.add_trace(go.Scatter(x=fc.universe, y=_mf(spec, fc.universe),
                                 name=name.replace("_", " "), mode="lines",
                                 line=dict(width=1, dash="dot"), opacity=0.5))
    fig.add_trace(go.Scatter(x=fc.universe, y=agg, name="Aggregated output",
                             fill="tozeroy", line=dict(color=theme.TEAL, width=2.5),
                             fillcolor="rgba(20,184,166,.25)"))
    fig.add_vline(x=trace["priority"], line=dict(color=theme.RED, dash="dash"),
                  annotation_text="centroid %.3f" % trace["priority"])
    fig.update_layout(height=320, title="Mamdani aggregation and defuzzification",
                      xaxis_title="Charging priority", yaxis_title="Membership")
    return fig


def _priority_heatmap(sc, pri):
    fig = go.Figure(go.Heatmap(z=pri, x=sc.cfg.slot_hours(),
                               y=list(sc.fleet.ev_id), colorscale="Teal",
                               colorbar=dict(title="Priority")))
    fig.update_layout(height=360, title="Fuzzy priority per EV and slot",
                      xaxis_title="Hour of day", yaxis_title="EV",
                      yaxis=dict(showticklabels=sc.fleet.n_evs <= 25))
    return fig


# ---------------------------------------------------------------------------
# Genetic algorithm
# ---------------------------------------------------------------------------
def _ga_section(sc):
    results = state.results()
    runs = {n: r for n, r in results.items() if (r.extras or {}).get("history", {}).get("best")}
    if not runs:
        theme.note("Run a simulation first (Algorithm Laboratory or Executive "
                   "Dashboard) to inspect the evolutionary search.")
        return

    name = st.selectbox("Optimiser run", list(runs))
    res = runs[name]
    hist, params = res.extras["history"], res.extras["params"]

    k = st.columns(4)
    with k[0]:
        theme.metric_card("Best objective J", "%.4f" % res.extras["best_J"],
                          hint="start %.4f" % hist["best"][0])
    with k[1]:
        theme.metric_card("Population x generations",
                          "%d x %d" % (params["population"], params["generations"]))
    with k[2]:
        theme.metric_card("Runtime", "%.2f s" % res.runtime_s,
                          hint="seed %s" % params["seed"])
    with k[3]:
        theme.metric_card("Fuzzy guided", "Yes" if res.extras["fuzzy_guided"] else "No",
                          hint="chromosome %d genes" % res.extras["best_genes"].size)

    st.plotly_chart(_history_chart(hist), use_container_width=True, theme=None)

    a, b = st.columns([2, 3])
    with a:
        st.markdown("###### Fitness breakdown of the best individual")
        br = res.extras["breakdown"]
        bdf = pd.DataFrame({
            "Term": ["cost", "peak", "unmet", "delay", "solar reward", "total J"],
            "Contribution to J": [round(br["cost"], 4), round(br["peak"], 4),
                                  round(br["unmet"], 4), round(br["delay"], 4),
                                  round(br["solar_reward"], 4), round(br["J"], 4)],
        })
        st.dataframe(bdf, use_container_width=True, hide_index=True)
        st.caption("Negative values are rewards. Terms are dimensionless: cost and "
                   "peak are divided by the uncontrolled baseline, unmet energy by "
                   "the requested energy and solar by the available solar energy.")
    with b:
        st.markdown("###### Best-found schedule (station-side power, kW)")
        st.plotly_chart(_schedule_heatmap(sc, res), use_container_width=True, theme=None)

    st.markdown("###### GA parameters")
    st.dataframe(pd.DataFrame([params]), use_container_width=True, hide_index=True)


def _history_chart(hist):
    gens = list(range(len(hist["best"])))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=gens, y=hist["worst"], name="Worst",
                             line=dict(color="#CBD5E1", width=1)))
    fig.add_trace(go.Scatter(x=gens, y=hist["mean"], name="Population mean",
                             line=dict(color=theme.BLUE, width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(x=gens, y=hist["best"], name="Best individual",
                             line=dict(color=theme.TEAL, width=3)))
    fig.update_layout(height=340, title="Convergence history",
                      xaxis_title="Generation", yaxis_title="Objective J")
    return fig


def _schedule_heatmap(sc, res):
    fig = go.Figure(go.Heatmap(z=res.power, x=sc.cfg.slot_hours(),
                               y=list(sc.fleet.ev_id), colorscale="Viridis",
                               colorbar=dict(title="kW")))
    fig.update_layout(height=360, xaxis_title="Hour of day", yaxis_title="EV",
                      yaxis=dict(showticklabels=sc.fleet.n_evs <= 25))
    return fig


# ---------------------------------------------------------------------------
# Interaction
# ---------------------------------------------------------------------------
def _interaction_section():
    st.markdown("""
##### How the fuzzy controller steers the genetic algorithm

The Mamdani controller turns four linguistic inputs into a priority
`pi[i, t]` in [0, 1] for every (EV, time slot) pair. That matrix enters the GA
in three concrete places:

1. **Population seeding** - initial chromosomes are Gaussian perturbations of
   the fuzzy priority matrix instead of uniform noise, so the search starts in
   a region that already respects deadline pressure, tariffs and solar.
2. **Constraint resolution** - when the decoded demand of a slot exceeds the
   shared station limit, capacity is distributed by weighted water filling
   using the fuzzy priorities as weights, instead of fair proportional
   curtailment. Urgent and solar-aligned vehicles keep their power.
3. **Guided mutation** - a share of the mutated genes is pulled half-way
   towards the fuzzy priority of that (EV, slot) pair, which anchors the search
   to domain knowledge.

The objective function, the scenario and the hard constraints are identical
for the plain GA and the hybrid, so the comparison isolates the effect of the
fuzzy guidance.

##### What is *not* claimed

- Neither optimiser is proven globally optimal; both are stochastic heuristics
  and results vary with the random seed.
- Fuzzy urgency is computed from the plug-in state once per scenario rather
  than re-evaluated inside every fitness evaluation, which keeps the dashboard
  interactive.
- No neural network or trained model is involved anywhere in the system.
    """)
