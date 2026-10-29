"""Page 2 - EV fleet table and session statistics."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from evaluation.comparison import STRATEGIES
from evaluation.report_generator import fleet_result_frame
from ui import state, theme


def render():
    theme.hero("EV Fleet",
               "Synthetic charging sessions, requested energy and per-vehicle outcome")
    cfg, sc = state.cfg(), state.scenario()

    if sc.fleet.n_evs == 0:
        theme.note("No vehicles in the fleet. Increase <b>Fleet size</b> in the "
                   "sidebar and press <b>Apply and regenerate</b>.")
        return

    results = state.ensure()
    c1, c2, c3 = st.columns([2, 2, 3])
    with c1:
        strategy = st.selectbox("Outcome for strategy", STRATEGIES,
                                index=STRATEGIES.index(st.session_state.active))
        st.session_state.active = strategy
    with c2:
        only_missed = st.checkbox("Only vehicles that missed their target", value=False)
    with c3:
        query = st.text_input("Search EV ID", placeholder="e.g. EV-007")

    df = fleet_result_frame(results[strategy], sc)
    if only_missed:
        df = df[~df["Target met"]]
    if query:
        df = df[df["EV ID"].str.contains(query.strip(), case=False, na=False)]

    k = st.columns(4)
    with k[0]:
        theme.metric_card("Vehicles shown", "%d of %d" % (len(df), sc.fleet.n_evs))
    with k[1]:
        theme.metric_card("Requested energy", "%.0f kWh" % sc.fleet.requested_kwh.sum())
    with k[2]:
        theme.metric_card("Mean dwell time",
                          "%.1f h" % (sc.fleet.dwell_slots().mean() * cfg.dt))
    with k[3]:
        theme.metric_card("Targets met",
                          "%d / %d" % (int(df["Target met"].sum()), len(df))
                          if len(df) else "0 / 0")

    st.dataframe(df, use_container_width=True, hide_index=True, height=420,
                 column_config={
                     "Final SOC (%)": st.column_config.ProgressColumn(
                         "Final SOC (%)", min_value=0, max_value=100, format="%.1f"),
                     "Target met": st.column_config.CheckboxColumn("Target met"),
                 })

    st.download_button("Download fleet CSV", df.to_csv(index=False).encode(),
                       file_name="evision_fleet_%s.csv" % strategy.replace(" ", "_").lower(),
                       mime="text/csv")

    st.markdown("##### Session distributions")
    a, b = st.columns(2)
    with a:
        st.plotly_chart(_arrival_chart(sc), use_container_width=True, theme=None)
    with b:
        st.plotly_chart(_energy_chart(df), use_container_width=True, theme=None)

    theme.info("Fleet data is synthetic: battery capacity, SOC, arrival, dwell time "
               "and charger rating are drawn from the configured distributions using "
               "the random seed shown in the sidebar. The same seed always rebuilds "
               "the same fleet.")


def _arrival_chart(sc):
    hours = sc.fleet.arrival_slot * sc.cfg.dt
    fig = px.histogram(x=hours, nbins=24, labels={"x": "Arrival hour"},
                       title="Arrival time distribution")
    fig.update_traces(marker_color=theme.TEAL)
    fig.update_layout(height=320, yaxis_title="Vehicles", bargap=0.05,
                      xaxis=dict(dtick=3))
    return fig


def _energy_chart(df):
    fig = px.scatter(df, x="Dwell (h)", y="Requested (kWh)", color="Target met",
                     hover_name="EV ID", size="Charger (kW)",
                     color_discrete_map={True: theme.GREEN, False: theme.RED},
                     title="Requested energy vs available dwell time")
    fig.update_layout(height=320)
    return fig
