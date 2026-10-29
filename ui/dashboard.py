"""Page 1 - Executive dashboard."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from evaluation.comparison import STRATEGIES
from evaluation.metrics import compute_metrics, pct_change
from ui import state, theme


def render():
    theme.hero("Executive Dashboard",
               "Fleet, cost, grid and solar position for the selected charging strategy")
    cfg, sc = state.cfg(), state.scenario()

    if sc.fleet.n_evs == 0:
        theme.note("The fleet is empty. Increase <b>Fleet size</b> in the sidebar "
                   "and press <b>Apply and regenerate</b> to run a simulation.")
        return

    results = state.ensure()
    left, right = st.columns([3, 1])
    with right:
        active = st.selectbox("Strategy", STRATEGIES,
                              index=STRATEGIES.index(st.session_state.active))
        st.session_state.active = active
    with left:
        st.markdown("#### %s" % active)
        st.caption(_strategy_blurb(active))

    res = results[active]
    m = compute_metrics(res, sc)
    base = compute_metrics(results["Uncontrolled"], sc)

    c = st.columns(4)
    with c[0]:
        theme.metric_card("Fleet size", "%d EVs" % m["n_evs"],
                          hint="%.0f kWh requested" % m["requested_kwh"])
    with c[1]:
        d = pct_change(m["cost_rs"], base["cost_rs"])
        theme.metric_card("Projected electricity cost", "Rs %s" % f"{m['cost_rs']:,.0f}",
                          delta="%+.1f%% vs uncontrolled" % d if active != "Uncontrolled" else "",
                          good=theme.delta_direction(d), hint="Grid energy only")
    with c[2]:
        d = pct_change(m["peak_grid_kw"], base["peak_grid_kw"])
        theme.metric_card("Peak grid demand", "%.1f kW" % m["peak_grid_kw"],
                          delta="%+.1f%% vs uncontrolled" % d if active != "Uncontrolled" else "",
                          good=theme.delta_direction(d),
                          hint="Station limit %.0f kW" % cfg.station_capacity_kw)
    with c[3]:
        d = m["solar_util_pct"] - base["solar_util_pct"]
        theme.metric_card("Solar utilisation", "%.1f%%" % m["solar_util_pct"],
                          delta="%+.1f pp vs uncontrolled" % d if active != "Uncontrolled" else "",
                          good=theme.delta_direction(d, lower_is_better=False),
                          hint="%.0f of %.0f kWh used"
                               % (m["solar_to_ev_kwh"], m["available_solar_kwh"]))

    c = st.columns(4)
    with c[0]:
        theme.metric_card("EVs meeting target SOC", "%.1f%%" % m["target_met_pct"],
                          hint="%d of %d vehicles" % (m["target_met_count"], m["n_evs"]))
    with c[1]:
        theme.metric_card("Energy delivered", "%.0f kWh" % m["delivered_kwh"],
                          hint="%.1f kWh unmet (%.1f%%)" % (m["unmet_kwh"], m["unmet_pct"]))
    with c[2]:
        saving = base["cost_rs"] - m["cost_rs"]
        theme.metric_card("Estimated saving", "Rs %s" % f"{saving:,.0f}",
                          hint="versus uncontrolled charging",
                          good="down" if saving > 0 else "flat")
    with c[3]:
        theme.metric_card("Schedule feasibility",
                          "OK" if m["feasible"] else "%d violations" % m["violations"],
                          hint="Runtime %.2f s" % m["runtime_s"],
                          good="down" if m["feasible"] else "up")

    st.markdown("")
    left, right = st.columns([2, 1])
    with left:
        st.markdown("##### Grid demand, solar generation and charging power")
        st.plotly_chart(_overview_chart(sc, res), use_container_width=True, theme=None)
    with right:
        st.markdown("##### Simulation configuration")
        st.dataframe(_config_frame(cfg, sc), hide_index=True,
                     use_container_width=True, height=380)

    if m["unmet_pct"] > base["unmet_pct"] + 0.5:
        theme.note(
            "This strategy leaves <b>%.1f%%</b> of the requested energy unmet "
            "against <b>%.1f%%</b> for uncontrolled charging: part of the cost "
            "and peak reduction is paid for in departure SOC. Raise the "
            "<i>unmet energy</i> weight in the Algorithm Laboratory to move the "
            "operating point back towards full deadline compliance."
            % (m["unmet_pct"], base["unmet_pct"]))


def _strategy_blurb(name: str) -> str:
    return {
        "Uncontrolled": "Every EV charges at full rate from plug-in; the only "
                        "limit is the shared station capacity. Reference baseline.",
        "Rule-based": "Crisp heuristic: charge when the deadline presses, when "
                      "energy is cheap, or when on-site solar is available.",
        "Genetic Algorithm": "Evolutionary search over the charging schedule "
                             "without any fuzzy guidance.",
        "Hybrid Fuzzy + GA": "Proposed method: a Mamdani fuzzy controller "
                             "prioritises every (EV, slot) pair and steers the "
                             "GA seeding, mutation and capacity allocation.",
    }[name]


def _overview_chart(sc, res):
    hours = sc.cfg.slot_hours()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hours, y=sc.solar_kw, name="Solar available (kW)",
                             fill="tozeroy", line=dict(color=theme.GREEN, width=1),
                             fillcolor="rgba(34,197,94,.18)"))
    fig.add_trace(go.Scatter(x=hours, y=res.total_power_kw, name="EV charging (kW)",
                             line=dict(color=theme.TEAL, width=2)))
    fig.add_trace(go.Scatter(x=hours, y=res.grid_power_kw, name="Grid import (kW)",
                             line=dict(color=theme.SLATE, width=2, dash="dot")))
    fig.add_hline(y=sc.cfg.station_capacity_kw, line=dict(color=theme.RED, dash="dash"),
                  annotation_text="Station limit", annotation_position="top left")
    fig.update_layout(height=380, xaxis_title="Hour of day", yaxis_title="Power (kW)",
                      xaxis=dict(dtick=3, range=[0, sc.cfg.horizon_hours]))
    return fig


def _config_frame(cfg, sc):
    import pandas as pd
    rows = [
        ("Random seed", str(cfg.seed)),
        ("Horizon", "%.0f h / %d slots" % (cfg.horizon_hours, cfg.n_slots)),
        ("Slot length", "%d min" % cfg.slot_minutes),
        ("Station capacity", "%.0f kW" % cfg.station_capacity_kw),
        ("Solar capacity", "%.0f kW" % cfg.solar_capacity_kw),
        ("Charging efficiency", "%.2f" % cfg.charging_efficiency),
        ("Tariff off-peak", "Rs %.2f /kWh" % cfg.tariff_offpeak),
        ("Tariff standard", "Rs %.2f /kWh" % cfg.tariff_standard),
        ("Tariff peak", "Rs %.2f /kWh" % cfg.tariff_peak),
        ("Available solar energy", "%.0f kWh" % sc.available_solar_kwh),
        ("GA population / generations", "%d / %d" % (cfg.ga_population, cfg.ga_generations)),
        ("Objective weights", "cost %.1f, peak %.1f, unmet %.1f, delay %.1f, solar %.1f"
         % (cfg.w_cost, cfg.w_peak, cfg.w_unmet, cfg.w_delay, cfg.w_solar)),
    ]
    return pd.DataFrame(rows, columns=["Setting", "Value"])
