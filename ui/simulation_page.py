"""Page 3 - Energy simulation: power, tariff, battery and station utilisation."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from evaluation.comparison import STRATEGIES
from simulation.ev_models import hhmm
from ui import state, theme


def render():
    theme.hero("Energy Simulation",
               "Slot-by-slot power balance, tariffs, battery accumulation and utilisation")
    cfg, sc = state.cfg(), state.scenario()

    if sc.fleet.n_evs == 0:
        theme.note("Generate a fleet first (sidebar) to simulate energy flows.")
        return

    results = state.ensure()
    strategy = st.selectbox("Strategy", STRATEGIES,
                            index=STRATEGIES.index(st.session_state.active))
    st.session_state.active = strategy
    res = results[strategy]
    hours = cfg.slot_hours()

    st.plotly_chart(_power_chart(sc, res, hours), use_container_width=True, theme=None)

    a, b = st.columns(2)
    with a:
        st.plotly_chart(_tariff_chart(sc, res, hours), use_container_width=True, theme=None)
    with b:
        st.plotly_chart(_battery_chart(sc, res, hours), use_container_width=True, theme=None)

    st.markdown("##### Station utilisation")
    st.plotly_chart(_utilisation_chart(sc, res, hours), use_container_width=True, theme=None)

    st.markdown("##### Time-slot inspector")
    slot = st.slider("Slot", 0, cfg.n_slots - 1, int(np.argmax(res.total_power_kw)),
                     help="Pick a 15-minute slot to inspect the allocation in detail.")
    _slot_detail(sc, res, slot)

    table = res.slot_frame(sc)
    st.download_button("Download slot-level schedule CSV",
                       table.to_csv(index=False).encode(),
                       file_name="evision_slots_%s.csv" % strategy.replace(" ", "_").lower(),
                       mime="text/csv")
    with st.expander("Full slot table"):
        st.dataframe(table, use_container_width=True, hide_index=True, height=360)


def _power_chart(sc, res, hours):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hours, y=sc.solar_kw, name="Solar available",
                             line=dict(color=theme.GREEN, width=1),
                             fill="tozeroy", fillcolor="rgba(34,197,94,.15)"))
    fig.add_trace(go.Scatter(x=hours, y=res.solar_to_ev_kw, name="Solar used by EVs",
                             line=dict(color=theme.GREEN, width=2)))
    fig.add_trace(go.Scatter(x=hours, y=res.grid_power_kw, name="Grid import",
                             stackgroup="supply", line=dict(width=0),
                             fillcolor="rgba(71,85,105,.35)"))
    fig.add_trace(go.Scatter(x=hours, y=res.total_power_kw, name="Total EV charging",
                             line=dict(color=theme.TEAL, width=2.5)))
    fig.add_hline(y=sc.cfg.station_capacity_kw, line=dict(color=theme.RED, dash="dash"),
                  annotation_text="Station capacity")
    fig.update_layout(height=400, title="Solar generation vs EV charging demand",
                      xaxis_title="Hour of day", yaxis_title="Power (kW)",
                      xaxis=dict(dtick=3, range=[0, sc.cfg.horizon_hours]))
    return fig


def _tariff_chart(sc, res, hours):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=hours, y=sc.tariff, name="Tariff (Rs/kWh)",
                         marker_color="rgba(59,130,246,.35)"), secondary_y=False)
    fig.add_trace(go.Scatter(x=hours, y=res.grid_power_kw, name="Grid import (kW)",
                             line=dict(color=theme.SLATE, width=2)), secondary_y=True)
    fig.update_yaxes(title_text="Tariff (Rs/kWh)", secondary_y=False)
    fig.update_yaxes(title_text="Grid power (kW)", secondary_y=True)
    fig.update_layout(height=340, title="Tariff and grid import",
                      xaxis_title="Hour of day", xaxis=dict(dtick=4))
    return fig


def _battery_chart(sc, res, hours):
    total = res.battery_cum_kwh.sum(axis=0) if res.battery_cum_kwh.size else np.zeros(len(hours))
    requested = float(sc.fleet.requested_kwh.sum())
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hours, y=total, name="Energy stored in batteries",
                             line=dict(color=theme.TEAL, width=2.5),
                             fill="tozeroy", fillcolor="rgba(20,184,166,.15)"))
    fig.add_hline(y=requested, line=dict(color=theme.AMBER, dash="dash"),
                  annotation_text="Total requested")
    fig.update_layout(height=340, title="Cumulative battery energy",
                      xaxis_title="Hour of day", yaxis_title="Energy (kWh)",
                      xaxis=dict(dtick=4))
    return fig


def _utilisation_chart(sc, res, hours):
    util = 100.0 * res.total_power_kw / max(sc.cfg.station_capacity_kw, 1e-9)
    connected = (sc.fleet.availability()).sum(axis=0)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=hours, y=util, name="Station utilisation (%)",
                         marker_color="rgba(20,184,166,.55)"), secondary_y=False)
    fig.add_trace(go.Scatter(x=hours, y=connected, name="EVs connected",
                             line=dict(color=theme.NAVY, width=2)), secondary_y=True)
    fig.update_yaxes(title_text="Utilisation (%)", range=[0, 105], secondary_y=False)
    fig.update_yaxes(title_text="EVs connected", secondary_y=True)
    fig.update_layout(height=320, xaxis_title="Hour of day", xaxis=dict(dtick=3))
    return fig


def _slot_detail(sc, res, slot):
    import pandas as pd
    cfg = sc.cfg
    k = st.columns(5)
    with k[0]:
        theme.metric_card("Time", hhmm(slot * cfg.dt))
    with k[1]:
        theme.metric_card("Tariff", "Rs %.2f" % sc.tariff[slot], hint="per kWh")
    with k[2]:
        theme.metric_card("Solar used", "%.1f kW" % res.solar_to_ev_kw[slot],
                          hint="%.1f kW available" % sc.solar_kw[slot])
    with k[3]:
        theme.metric_card("Grid import", "%.1f kW" % res.grid_power_kw[slot],
                          hint="Rs %.2f this slot" % res.cost_per_slot[slot])
    with k[4]:
        theme.metric_card("Charging power", "%.1f kW" % res.total_power_kw[slot],
                          hint="%.0f%% of station limit"
                               % (100 * res.total_power_kw[slot] / max(cfg.station_capacity_kw, 1e-9)))

    active = np.where(res.power[:, slot] > 1e-6)[0]
    if active.size == 0:
        st.caption("No vehicle is charging in this slot.")
        return
    share = res.power[active, slot] / max(res.total_power_kw[slot], 1e-9)
    df = pd.DataFrame({
        "EV ID": sc.fleet.ev_id[active],
        "Power (kW)": res.power[active, slot].round(2),
        "Share of slot (%)": (share * 100).round(1),
        "Solar share (kW)": (share * res.solar_to_ev_kw[slot]).round(2),
        "Grid share (kW)": (share * res.grid_power_kw[slot]).round(2),
        "Battery gain (kWh)": (res.power[active, slot] * cfg.dt
                               * cfg.charging_efficiency).round(3),
    }).sort_values("Power (kW)", ascending=False)
    st.dataframe(df, use_container_width=True, hide_index=True, height=260)
    st.caption("Solar and grid shares split the slot supply pro rata across the "
               "vehicles charging in that slot; on-site solar is self-consumed first.")
