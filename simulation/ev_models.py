"""EV fleet data model.

The fleet is stored as a struct-of-arrays (numpy) so that every downstream
computation can be vectorised over vehicles and time slots.

Slot convention: ``arrival_slot`` is inclusive, ``departure_slot`` is
exclusive, i.e. the EV may charge during slots [arrival_slot, departure_slot).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Fleet:
    ev_id: np.ndarray            # (n,)  str
    battery_kwh: np.ndarray      # (n,)  kWh
    soc_init: np.ndarray         # (n,)  0..1
    soc_target: np.ndarray       # (n,)  0..1
    arrival_slot: np.ndarray     # (n,)  int, inclusive
    departure_slot: np.ndarray   # (n,)  int, exclusive
    charger_kw: np.ndarray       # (n,)  kW
    n_slots: int

    @property
    def n_evs(self) -> int:
        return int(self.battery_kwh.shape[0])

    @property
    def requested_kwh(self) -> np.ndarray:
        """Energy the battery must absorb to reach the target SOC (kWh)."""
        return np.maximum(0.0, (self.soc_target - self.soc_init) * self.battery_kwh)

    @property
    def headroom_kwh(self) -> np.ndarray:
        """Energy to 100 percent SOC (kWh) - the physical battery limit."""
        return np.maximum(0.0, (1.0 - self.soc_init) * self.battery_kwh)

    def availability(self) -> np.ndarray:
        """Boolean (n, T) mask: True when the EV is plugged in."""
        t = np.arange(self.n_slots)[None, :]
        return (t >= self.arrival_slot[:, None]) & (t < self.departure_slot[:, None])

    def max_power(self) -> np.ndarray:
        """(n, T) kW upper bound from charger rating and availability."""
        return self.availability() * self.charger_kw[:, None]

    def dwell_slots(self) -> np.ndarray:
        return np.maximum(0, self.departure_slot - self.arrival_slot)

    def to_frame(self, dt: float) -> pd.DataFrame:
        return pd.DataFrame({
            "EV ID": self.ev_id,
            "Battery (kWh)": self.battery_kwh.round(1),
            "Initial SOC (%)": (self.soc_init * 100).round(1),
            "Target SOC (%)": (self.soc_target * 100).round(1),
            "Arrival": [hhmm(s * dt) for s in self.arrival_slot],
            "Departure": [hhmm(s * dt) for s in self.departure_slot],
            "Dwell (h)": (self.dwell_slots() * dt).round(2),
            "Charger (kW)": self.charger_kw,
            "Requested (kWh)": self.requested_kwh.round(2),
        })


def hhmm(hours: float) -> str:
    """Format a clock time in hours as HH:MM (wrapping at 24 h)."""
    hours = float(hours) % 24.0
    h = int(hours)
    m = int(round((hours - h) * 60))
    if m == 60:
        h, m = (h + 1) % 24, 0
    return "%02d:%02d" % (h, m)


def empty_fleet(n_slots: int) -> Fleet:
    """A valid, zero-vehicle fleet (used for empty-state handling and tests)."""
    z = np.zeros(0)
    return Fleet(
        ev_id=np.array([], dtype=object),
        battery_kwh=z, soc_init=z, soc_target=z,
        arrival_slot=np.zeros(0, dtype=int), departure_slot=np.zeros(0, dtype=int),
        charger_kw=z, n_slots=n_slots,
    )
