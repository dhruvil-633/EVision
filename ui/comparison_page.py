"""Page 5 - Side-by-side performance comparison of all four strategies."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from evaluation.comparison import BASELINE, STRATEGIES, comparison_frame, summary_table
from ui import state, theme

PANELS = [
    ("cost_rs", "Electricity cost (Rs)", True),
    ("peak_grid_kw", "Peak grid demand (kW)", True),
    ("solar_util_pct", "Solar utilisation (%)", False),
    ("target_met_pct", "EVs meeting target SOC (%)", False),
    ("unmet_kwh", "Unmet energy (kWh)", True),
    ("runtime_s", "Optimisation runtime (s)", True),
]


def render():
    theme.hero("Performance Comparison",
               "All four scheduling strategies on the same scenario, "
               "constraints and metric definitions")
    sc = state.scenario()
    if sc.fleet.n_evs == 0:
        theme.note("Generate a fleet first to compare strategies.")
        return

    results = state.ensure()
    df = comparison_frame(results, sc)

    st.plotly_chart(_panel_grid(df), use_container_width=True, theme=None)

    st.markdown("##### Absolute values and change against the %s baseline" % BASELINE)
    table = summary_table(results, sc)
    st.dataframe(table, use_container_width=True)
    st.download_button("Download comparison CSV", table.to_csv().encode(),
                       file_name="evision_comparison.csv", mime="text/csv")

    st.markdown("##### Cost versus deadline compliance")
    st.plotly_chart(_tradeoff_chart(df), use_container_width=True, theme=None)

    st.markdown("##### Reading the trade-offs")
    base = df.loc[BASELINE]
    bullets = []
    for name in STRATEGIES:
        if name == BASELINE or name not in df.index:
            continue
        r = df.loc[name]
        bullets.append(
            "- **%s** - cost %+.1f%%, peak %+.1f%%, solar utilisation %+.1f pp, "
            "EVs meeting target %.1f%% (baseline %.1f%%), runtime %.2f s."
            % (name, r["cost_rs_vs_base_pct"], r["peak_grid_kw_vs_base_pct"],
               r["solar_util_pct"] - base["solar_util_pct"], r["target_met_pct"],
               base["target_met_pct"], r["runtime_s"]))
    st.markdown("\n".join(bullets))

    theme.info(
        "Percentages are computed against the uncontrolled baseline of this exact "
        "scenario. Smart strategies buy cost and peak reductions by moving energy "
        "into cheap, sunny or quiet slots; where a vehicle leaves before that energy "
        "can be delivered, the saving is partly paid for in departure SOC, which is "
        "why unmet energy and target completion are shown next to every cost figure. "
        "The genetic algorithm and the hybrid are stochastic heuristics - neither is "
        "a proven global optimum.")


def _panel_grid(df):
    fig = make_subplots(rows=2, cols=3, subplot_titles=[p[1] for p in PANELS],
                        vertical_spacing=0.18, horizontal_spacing=0.08)
    names = [n for n in STRATEGIES if n in df.index]
    colors = [theme.STRATEGY_COLORS[n] for n in names]
    for k, (col, title, lower_better) in enumerate(PANELS):
        r, c = divmod(k, 3)
        vals = [df.loc[n, col] for n in names]
        fig.add_trace(go.Bar(x=names, y=vals, marker_color=colors, showlegend=False,
                             text=["%.1f" % v for v in vals], textposition="outside",
                             hovertemplate="%{x}<br>" + title + ": %{y:.2f}<extra></extra>"),
                      row=r + 1, col=c + 1)
    fig.update_layout(height=620, margin=dict(t=60, b=40))
    fig.update_xaxes(tickangle=-15, tickfont=dict(size=10))
    return fig


def _tradeoff_chart(df):
    names = [n for n in STRATEGIES if n in df.index]
    fig = go.Figure()
    for n in names:
        fig.add_trace(go.Scatter(
            x=[df.loc[n, "cost_rs"]], y=[df.loc[n, "target_met_pct"]],
            mode="markers+text", text=[n], textposition="top center",
            marker=dict(size=18, color=theme.STRATEGY_COLORS[n]), name=n,
            hovertemplate="%s<br>Cost Rs %%{x:,.0f}<br>Targets met %%{y:.1f}%%"
                          "<extra></extra>" % n))
    fig.update_layout(height=380, xaxis_title="Electricity cost (Rs, lower is better)",
                      yaxis_title="EVs meeting target SOC (%, higher is better)",
                      showlegend=False, hovermode="closest")
    return fig
