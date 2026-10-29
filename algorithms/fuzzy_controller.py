"""Mamdani fuzzy inference system for EV charging priority.

Implementation note (honest substitution)
-----------------------------------------
scikit-fuzzy provides a Mamdani engine, but its ControlSystemSimulation
evaluates ONE input tuple per call, and this project needs one inference per
(EV, time slot) pair. We therefore implement the SAME Mamdani pipeline in
vectorised numpy:

    fuzzification -> min(AND) rule firing -> Mamdani min implication
    -> max aggregation -> centroid defuzzification

tests/test_fuzzy.py cross-checks this engine against scikit-fuzzy using the
same membership functions and rule base: membership functions and rule
aggregation must match exactly and the defuzzified output to within 1e-9, so
the substitution is verified rather than assumed.

Linguistic variables (all crisp inputs are normalised to [0, 1])
----------------------------------------------------------------
urgency   how hard the EV must push to meet its departure deadline
          (required average power / charger rating)
price     normalised time-of-use tariff of the slot
solar     normalised on-site solar availability in the slot
headroom  remaining shared station capacity in the slot
output    charging priority in [0, 1]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------
# Membership functions
# --------------------------------------------------------------------------
def trimf(x, a, b, c):
    """Triangular membership function."""
    x = np.asarray(x, dtype=float)
    left = np.where(b > a, (x - a) / max(b - a, 1e-12), np.where(x >= b, 1.0, 0.0))
    right = np.where(c > b, (c - x) / max(c - b, 1e-12), np.where(x <= b, 1.0, 0.0))
    return np.clip(np.minimum(left, right), 0.0, 1.0)


def trapmf(x, a, b, c, d):
    """Trapezoidal membership function."""
    x = np.asarray(x, dtype=float)
    left = np.where(b > a, (x - a) / max(b - a, 1e-12), np.where(x >= b, 1.0, 0.0))
    right = np.where(d > c, (d - x) / max(d - c, 1e-12), np.where(x <= c, 1.0, 0.0))
    return np.clip(np.minimum(np.minimum(left, right), 1.0), 0.0, 1.0)


# Three-term partition shared by every input variable.
INPUT_TERMS = {
    "low": ("trap", (0.0, 0.0, 0.15, 0.40)),
    "medium": ("tri", (0.25, 0.50, 0.75)),
    "high": ("trap", (0.60, 0.85, 1.0, 1.0)),
}

# Five-term partition of the priority output.
OUTPUT_TERMS = {
    "very_low": ("trap", (0.0, 0.0, 0.05, 0.22)),
    "low": ("tri", (0.10, 0.28, 0.46)),
    "medium": ("tri", (0.34, 0.50, 0.66)),
    "high": ("tri", (0.54, 0.72, 0.90)),
    "very_high": ("trap", (0.78, 0.95, 1.0, 1.0)),
}

INPUT_VARS = ("urgency", "price", "solar", "headroom")


def _mf(spec, x):
    kind, p = spec
    return trimf(x, *p) if kind == "tri" else trapmf(x, *p)


# --------------------------------------------------------------------------
# Rule base: (antecedents, consequent, plain-language description).
# Inputs that a rule does not list are treated as DO-NOT-CARE.
# --------------------------------------------------------------------------
RULES = [
    ({"urgency": "high", "headroom": "high"}, "very_high",
     "Urgent EV and plenty of station capacity -> charge now"),
    ({"urgency": "high", "solar": "high"}, "very_high",
     "Urgent EV and abundant solar -> charge now on free energy"),
    ({"urgency": "high", "headroom": "medium"}, "high",
     "Urgent EV, moderate capacity -> still prioritise"),
    ({"urgency": "high", "headroom": "low"}, "high",
     "Critically low capacity is overridden by an urgent deadline"),
    ({"urgency": "high", "price": "high"}, "high",
     "Expensive energy does not justify missing a deadline"),
    ({"urgency": "medium", "solar": "high"}, "high",
     "Moderate urgency with high solar -> elevated priority"),
    ({"urgency": "medium", "price": "low", "headroom": "high"}, "high",
     "Cheap slot with free capacity -> pull charging forward"),
    ({"urgency": "medium", "price": "low"}, "high",
     "Cheap energy and moderate urgency -> charge"),
    ({"urgency": "medium", "price": "medium", "solar": "medium"}, "medium",
     "Everything average -> average priority"),
    ({"urgency": "medium", "price": "high", "solar": "low"}, "low",
     "Expensive grid energy, no solar, no deadline pressure -> defer"),
    ({"urgency": "medium", "headroom": "low"}, "low",
     "Scarce station capacity -> yield to more urgent vehicles"),
    ({"urgency": "low", "price": "low"}, "medium",
     "No pressure but cheap -> opportunistic charging"),
    ({"urgency": "low", "solar": "high"}, "medium",
     "No pressure but surplus solar -> absorb it"),
    ({"urgency": "low", "price": "medium", "solar": "medium"}, "low",
     "Relaxed EV in an average slot -> low priority"),
    ({"urgency": "low", "price": "high", "solar": "low"}, "very_low",
     "Relaxed EV, expensive slot, no solar -> wait"),
    ({"urgency": "low", "headroom": "low"}, "very_low",
     "Relaxed EV must yield when the station is nearly saturated"),
    ({"urgency": "low", "headroom": "high", "price": "medium"}, "low",
     "Spare capacity slightly raises a relaxed EV above the floor"),
    ({"solar": "high", "price": "high", "urgency": "low"}, "medium",
     "Self-consumption is worthwhile even at a high grid tariff"),
]


@dataclass
class FuzzyController:
    """Vectorised Mamdani controller: min implication, max aggregation and
    centroid defuzzification over a discretised output universe."""
    resolution: int = 101

    def __post_init__(self):
        self.universe = np.linspace(0.0, 1.0, self.resolution)
        self._out_mf = {k: _mf(v, self.universe) for k, v in OUTPUT_TERMS.items()}

    # -- fuzzification ---------------------------------------------------
    def fuzzify(self, **inputs) -> dict:
        out = {}
        for var in INPUT_VARS:
            x = np.clip(np.asarray(inputs[var], dtype=float), 0.0, 1.0)
            for term, spec in INPUT_TERMS.items():
                out[(var, term)] = _mf(spec, x)
        return out

    def rule_strengths(self, mu: dict, shape) -> np.ndarray:
        """(n_rules, ...) firing strength of every rule (AND = min)."""
        strengths = np.ones((len(RULES),) + tuple(shape), dtype=float)
        for r, (ante, _, _) in enumerate(RULES):
            for var, term in ante.items():
                strengths[r] = np.minimum(strengths[r], mu[(var, term)])
        return strengths

    # -- inference + defuzzification -------------------------------------
    def evaluate(self, urgency, price, solar, headroom) -> np.ndarray:
        """Crisp priority in [0, 1] for broadcastable array-like inputs."""
        urgency = np.atleast_1d(np.asarray(urgency, dtype=float))
        shape = urgency.shape
        price = np.broadcast_to(np.asarray(price, dtype=float), shape)
        solar = np.broadcast_to(np.asarray(solar, dtype=float), shape)
        headroom = np.broadcast_to(np.asarray(headroom, dtype=float), shape)
        if urgency.size == 0:
            return np.zeros(shape)

        mu = self.fuzzify(urgency=urgency, price=price, solar=solar, headroom=headroom)
        strengths = self.rule_strengths(mu, shape)

        agg = np.zeros(shape + (self.resolution,), dtype=float)
        for r, (_, cons, _) in enumerate(RULES):
            clipped = np.minimum(strengths[r][..., None], self._out_mf[cons][None, :])
            np.maximum(agg, clipped, out=agg)

        # Exact centroid of the piecewise-linear aggregated membership
        # function (same integration scheme as scikit-fuzzy).
        x0, x1 = self.universe[:-1], self.universe[1:]
        y0, y1 = agg[..., :-1], agg[..., 1:]
        dx = x1 - x0
        area = (0.5 * (y0 + y1) * dx).sum(axis=-1)
        moment = (dx / 6.0 * (x0 * (2 * y0 + y1) + x1 * (y0 + 2 * y1))).sum(axis=-1)
        # If no rule fires at all, fall back to the neutral middle.
        return np.where(area > 1e-12, moment / np.maximum(area, 1e-12), 0.5)

    def evaluate_scalar(self, urgency, price, solar, headroom) -> float:
        return float(self.evaluate([urgency], [price], [solar], [headroom])[0])

    def explain(self, urgency, price, solar, headroom) -> dict:
        """Full inference trace for one input tuple (explainability page)."""
        inp = dict(urgency=urgency, price=price, solar=solar, headroom=headroom)
        mu = self.fuzzify(**{k: [v] for k, v in inp.items()})
        memberships = {"%s.%s" % (v, t): float(mu[(v, t)][0])
                       for v in INPUT_VARS for t in INPUT_TERMS}
        strengths = self.rule_strengths(mu, (1,))[:, 0]
        rules = [
            {"rule": i + 1,
             "antecedent": " AND ".join("%s is %s" % (v, t) for v, t in ante.items()),
             "consequent": "priority is %s" % cons,
             "description": label,
             "strength": float(strengths[i]),
             "active": bool(strengths[i] > 1e-6)}
            for i, (ante, cons, label) in enumerate(RULES)
        ]
        return {"inputs": inp, "memberships": memberships, "rules": rules,
                "priority": self.evaluate_scalar(**inp)}


# --------------------------------------------------------------------------
# Scenario -> fuzzy input signals -> priority matrix
# --------------------------------------------------------------------------
def fuzzy_inputs(scenario) -> dict:
    """Build the four normalised fuzzy input signals.

    urgency is (n_evs, n_slots); price, solar and headroom are (n_slots,).

    Urgency uses the energy requested at plug in and the time left before
    departure, i.e. a static (pre dispatch) estimate of deadline pressure.
    This keeps ONE fuzzy inference pass per scenario instead of one per GA
    fitness evaluation; the limitation is documented in the README.
    """
    cfg, fleet = scenario.cfg, scenario.fleet
    n, T, dt, eff = fleet.n_evs, cfg.n_slots, cfg.dt, cfg.charging_efficiency
    if n == 0:
        return {"urgency": np.zeros((0, T)), "price": np.zeros(T),
                "solar": np.zeros(T), "headroom": np.zeros(T)}

    slots = np.arange(T)[None, :]
    slots_left = np.maximum(fleet.departure_slot[:, None] - slots, 1)
    required_kw = fleet.requested_kwh[:, None] / (slots_left * dt * eff)
    urgency = np.clip(required_kw / fleet.charger_kw[:, None], 0.0, 1.0)
    urgency = urgency * fleet.availability()

    span = float(scenario.tariff.max() - scenario.tariff.min())
    if span > 1e-9:
        price = (scenario.tariff - scenario.tariff.min()) / span
    else:
        price = np.zeros(T)

    solar = np.clip(scenario.solar_kw / max(cfg.solar_capacity_kw, 1e-9), 0.0, 1.0)
    connected = (fleet.availability() * fleet.charger_kw[:, None]).sum(axis=0)
    headroom = np.clip(1.0 - connected / max(cfg.station_capacity_kw, 1e-9), 0.0, 1.0)

    return {"urgency": urgency, "price": price, "solar": solar, "headroom": headroom}


def priority_matrix(scenario, controller: FuzzyController = None) -> np.ndarray:
    """(n_evs, n_slots) defuzzified charging priority; 0 when unavailable."""
    controller = controller or FuzzyController()
    cfg, fleet = scenario.cfg, scenario.fleet
    n, T = fleet.n_evs, cfg.n_slots
    out = np.zeros((n, T))
    if n == 0:
        return out

    sig = fuzzy_inputs(scenario)
    idx = np.where(fleet.availability())
    if idx[0].size == 0:
        return out
    out[idx] = controller.evaluate(
        sig["urgency"][idx],
        np.broadcast_to(sig["price"], (n, T))[idx],
        np.broadcast_to(sig["solar"], (n, T))[idx],
        np.broadcast_to(sig["headroom"], (n, T))[idx],
    )
    return out
