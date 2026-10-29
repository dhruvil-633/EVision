"""Physical energy accounting for a charging schedule.

Per time slot of duration dt (hours), for a schedule P[i, t] in kW:

    charging_power(t)   = sum_i P[i, t]                             [kW]
    solar_to_ev(t)      = min(solar(t), charging_power(t))          [kW]
    grid_power(t)       = charging_power(t) - solar_to_ev(t) >= 0   [kW]
    grid_energy(t)      = grid_power(t) * dt                        [kWh]
    cost(t)             = grid_energy(t) * tariff(t)                [Rs]
    battery_gain(i, t)  = P[i, t] * dt * efficiency                 [kWh]

Solar is used on site first (self-consumption); any excess is curtailed. The
station-side energy drawn is therefore split exactly once into a solar part
and a grid part - no double counting. Battery-side energy is always the
station-side energy times the (constant) charging efficiency.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class SimulationResult:
    name: str
    power: np.ndarray             # (n, T) kW station-side, per EV
    total_power_kw: np.ndarray    # (T,)
    solar_to_ev_kw: np.ndarray    # (T,)
    solar_curtailed_kw: np.ndarray
    grid_power_kw: np.ndarray     # (T,)
    grid_energy_kwh: np.ndarray   # (T,)
    cost_per_slot: np.ndarray     # (T,) Rs
    battery_cum_kwh: np.ndarray   # (n, T) cumulative battery-side energy
    delivered_kwh: np.ndarray     # (n,)
    final_soc: np.ndarray         # (n,)
    runtime_s: float = 0.0
    extras: dict = field(default_factory=dict)

    def slot_frame(self, scenario) -> pd.DataFrame:
        cfg = scenario.cfg
        from simulation.ev_models import hhmm
        from data.synthetic_data import tariff_band
        return pd.DataFrame({
            "Slot": np.arange(cfg.n_slots),
            "Time": [hhmm(h) for h in cfg.slot_hours()],
            "Tariff (Rs/kWh)": scenario.tariff,
            "Band": tariff_band(cfg),
            "Solar available (kW)": scenario.solar_kw.round(2),
            "EV charging (kW)": self.total_power_kw.round(2),
            "Solar to EV (kW)": self.solar_to_ev_kw.round(2),
            "Grid import (kW)": self.grid_power_kw.round(2),
            "Grid energy (kWh)": self.grid_energy_kwh.round(3),
            "Slot cost (Rs)": self.cost_per_slot.round(2),
            "Station utilisation (%)":
                (100 * self.total_power_kw / max(cfg.station_capacity_kw, 1e-9)).round(1),
        })


def simulate(power: np.ndarray, scenario, name: str = "schedule",
             runtime_s: float = 0.0, extras: dict = None) -> SimulationResult:
    """Run the energy accounting for a (already feasible) schedule."""
    cfg, fleet = scenario.cfg, scenario.fleet
    dt, eff = cfg.dt, cfg.charging_efficiency
    power = np.asarray(power, dtype=float)
    if power.size == 0:
        power = power.reshape(max(fleet.n_evs, 0), cfg.n_slots)

    total = power.sum(axis=0) if power.shape[0] else np.zeros(cfg.n_slots)
    solar_to_ev = np.minimum(scenario.solar_kw, total)
    grid_power = np.maximum(total - solar_to_ev, 0.0)
    grid_energy = grid_power * dt
    cost = grid_energy * scenario.tariff

    battery_gain = power * dt * eff
    battery_cum = np.cumsum(battery_gain, axis=1) if power.shape[0] else np.zeros((0, cfg.n_slots))
    delivered = battery_gain.sum(axis=1) if power.shape[0] else np.zeros(0)
    final_soc = (fleet.soc_init + delivered / fleet.battery_kwh) if fleet.n_evs else np.zeros(0)

    return SimulationResult(
        name=name, power=power, total_power_kw=total,
        solar_to_ev_kw=solar_to_ev,
        solar_curtailed_kw=np.maximum(scenario.solar_kw - solar_to_ev, 0.0),
        grid_power_kw=grid_power, grid_energy_kwh=grid_energy, cost_per_slot=cost,
        battery_cum_kwh=battery_cum, delivered_kwh=delivered, final_soc=final_soc,
        runtime_s=runtime_s, extras=extras or {},
    )
