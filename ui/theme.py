"""Design system for the EVision dashboard: colours, CSS and shared widgets."""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

NAVY = "#0E1A2B"
NAVY_SOFT = "#16263D"
TEAL = "#14B8A6"
GREEN = "#22C55E"
AMBER = "#F59E0B"
RED = "#EF4444"
BLUE = "#3B82F6"
SLATE = "#475569"
INK = "#0F172A"
PAPER = "#FFFFFF"
CANVAS = "#F5F7FA"
BORDER = "#E2E8F0"

STRATEGY_COLORS = {
    "Uncontrolled": SLATE,
    "Rule-based": BLUE,
    "Genetic Algorithm": AMBER,
    "Hybrid Fuzzy + GA": TEAL,
}

CSS = """
<style>
.block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1500px;}
section[data-testid="stSidebar"] {background: %(navy)s;}
section[data-testid="stSidebar"] * {color: #E2E8F0;}
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {color: #FFFFFF;}
section[data-testid="stSidebar"] .stSlider label, section[data-testid="stSidebar"] label
 {color: #CBD5E1 !important; font-size: 0.85rem;}
.stApp {background: %(canvas)s;}
h1, h2, h3, h4 {color: %(ink)s; letter-spacing: -0.01em;}
.ev-hero {background: linear-gradient(100deg, %(navy)s 0%%, #1E3A5F 100%%);
  color: #fff; padding: 1.4rem 1.6rem; border-radius: 14px; margin-bottom: 1.2rem;}
.ev-hero h1 {color: #fff; margin: 0 0 .3rem 0; font-size: 1.6rem;}
.ev-hero p {margin: 0; color: #B6C6DA; font-size: .92rem;}
.ev-card {background: %(paper)s; border: 1px solid %(border)s; border-radius: 12px;
  padding: .95rem 1.05rem; height: 100%%; box-shadow: 0 1px 2px rgba(15,23,42,.04);}
.ev-card .label {font-size: .74rem; text-transform: uppercase; letter-spacing: .06em;
  color: #64748B; font-weight: 600;}
.ev-card .value {font-size: 1.5rem; font-weight: 700; color: %(ink)s; line-height: 1.5;}
.ev-card .delta {font-size: .8rem; font-weight: 600;}
.ev-card .hint {font-size: .74rem; color: #94A3B8;}
.ev-up {color: %(red)s;} .ev-down {color: %(green)s;} .ev-flat {color: #94A3B8;}
.ev-note {background: #FFFBEB; border-left: 3px solid %(amber)s; padding: .7rem .9rem;
  border-radius: 6px; font-size: .85rem; color: #78350F;}
.ev-info {background: #F0FDFA; border-left: 3px solid %(teal)s; padding: .7rem .9rem;
  border-radius: 6px; font-size: .85rem; color: #134E4A;}
.ev-pill {display: inline-block; padding: .18rem .6rem; border-radius: 999px;
  font-size: .72rem; font-weight: 600; background: #E2E8F0; color: #334155;
  margin-right: .35rem;}
div[data-testid="stDataFrame"] {border-radius: 10px;}
@media (max-width: 900px) {.ev-card .value {font-size: 1.2rem;}}
</style>
""" % {"navy": NAVY, "canvas": CANVAS, "paper": PAPER, "border": BORDER,
       "ink": INK, "red": RED, "green": GREEN, "amber": AMBER, "teal": TEAL}


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str):
    st.markdown(
        '<div class="ev-hero"><h1>%s</h1><p>%s</p></div>' % (title, subtitle),
        unsafe_allow_html=True)


def metric_card(label: str, value: str, delta: str = "", hint: str = "",
                good: str = "flat"):
    cls = {"up": "ev-up", "down": "ev-down"}.get(good, "ev-flat")
    delta_html = '<div class="delta %s">%s</div>' % (cls, delta) if delta else ""
    hint_html = '<div class="hint">%s</div>' % hint if hint else ""
    st.markdown(
        '<div class="ev-card"><div class="label">%s</div>'
        '<div class="value">%s</div>%s%s</div>' % (label, value, delta_html, hint_html),
        unsafe_allow_html=True)


def note(text: str):
    st.markdown('<div class="ev-note">%s</div>' % text, unsafe_allow_html=True)


def info(text: str):
    st.markdown('<div class="ev-info">%s</div>' % text, unsafe_allow_html=True)


def register_template():
    """A restrained Plotly template matching the dashboard."""
    tpl = go.layout.Template()
    tpl.layout = go.Layout(
        font=dict(family="Inter, Segoe UI, sans-serif", size=12, color=INK),
        paper_bgcolor=PAPER, plot_bgcolor=PAPER,
        colorway=[TEAL, BLUE, AMBER, SLATE, GREEN, RED],
        margin=dict(l=55, r=20, t=74, b=45),
        xaxis=dict(gridcolor="#EEF2F6", zerolinecolor="#E2E8F0"),
        yaxis=dict(gridcolor="#EEF2F6", zerolinecolor="#E2E8F0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.04, x=0,
                    bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
    )
    pio.templates["evision"] = tpl
    pio.templates.default = "evision"


def delta_direction(value: float, lower_is_better: bool = True) -> str:
    if abs(value) < 0.05:
        return "flat"
    improving = value < 0 if lower_is_better else value > 0
    return "down" if improving else "up"
