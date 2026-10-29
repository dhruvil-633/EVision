"""Synthetic scenario generation: solar, tariffs and the EV fleet.

Everything here is produced by deterministic mathematical formulas plus a
seeded RNG. No external data set, API or trained model is involved - the
numbers are SIMULATED INPUTS, not measurements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from config import SimulationConfig
from simulation.ev_models import Fleet, empty_fleet


# ---------------------------------------------------------------------------
# Solar
# ---------------------------------------------------------------------------
def solar_profile(cfg: SimulationConfig, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Bell-shaped daylight curve with optional cloud variability (kW).

        P_solar(t) = P_cap * sin(pi * (t - sunrise) / (sunset - sunrise)) ** 1.6

    Cloud variability multiplies the clear-sky curve by a smoothed random
    factor; with cloud_variability = 0 the profile is fully deterministic.
    """
    t = cfg.slot_hours()
    day = max(cfg.sunset_hour - cfg.sunrise_hour, 1e-9)
    x = (t - cfg.sunrise_hour) / day
    clear = np.where((x > 0) & (x < 1), np.sin(np.pi * np.clip(x, 0, 1)) ** 1.6, 0.0)
    profile = cfg.solar_capacity_kw * clear

    if cfg.cloud_variability > 0 and rng is not None:
        noise = rng.normal(0.0, 1.0, size=profile.size)
        noise = np.convolve(noise, np.ones(5) / 5.0, mode="same")   # smooth clouds
        factor = 1.0 - cfg.cloud_variability * np.clip(noise, 0.0, 3.0)
        profile = profile * np.clip(factor, 0.0, 1.0)
    return np.maximum(profile, 0.0)


# ---------------------------------------------------------------------------
# Tariff
# ---------------------------------------------------------------------------
def tariff_profile(cfg: SimulationConfig) -> np.ndarray:
    """Time-of-use tariff (Rs/kWh) per slot."""
    t = cfg.slot_hours()
    tariff = np.full(t.size, float(cfg.tariff_standard))
    for lo, hi in cfg.offpeak_windows:
        tariff[(t >= lo) & (t < hi)] = cfg.tariff_offpeak
    for lo, hi in cfg.peak_windows:
        tariff[(t >= lo) & (t < hi)] = cfg.tariff_peak
    return tariff


def tariff_band(cfg: SimulationConfig) -> list:
    """Human readable tariff band per slot."""
    t = cfg.slot_hours()
    band = ["Standard"] * t.size
    for i, h in enumerate(t):
        for lo, hi in cfg.offpeak_windows:
            if lo <= h < hi:
                band[i] = "Off-peak"
        for lo, hi in cfg.peak_windows:
            if lo <= h < hi:
                band[i] = "Peak"
    return band


# ---------------------------------------------------------------------------
# Fleet
# ---------------------------------------------------------------------------
def generate_fleet(cfg: SimulationConfig, rng: Optional[np.random.Generator] = None) -> Fleet:
    """Synthetic EV charging sessions drawn from controlled distributions."""
    rng = rng if rng is not None else np.random.default_rng(cfg.seed)
    n, T, dt = cfg.n_evs, cfg.n_slots, cfg.dt
    if n <= 0:
        return empty_fleet(T)

    battery = rng.uniform(*cfg.battery_kwh_range, size=n)
    soc_i = rng.uniform(*cfg.soc_init_range, size=n)
    soc_t = rng.uniform(*cfg.soc_target_range, size=n)
    soc_t = np.minimum(np.maximum(soc_t, soc_i + 0.05), 1.0)   # always a real request

    weights = np.array(cfg.charger_kw_weights, dtype=float)
    charger = rng.choice(np.array(cfg.charger_kw_choices, dtype=float),
                         size=n, p=weights / weights.sum())

    # Bimodal arrivals: morning workplace peak + evening residential peak.
    peaks = np.array(cfg.arrival_peak_hours, dtype=float)
    which = rng.integers(0, peaks.size, size=n)
    arrival_h = np.clip(rng.normal(peaks[which], cfg.arrival_sigma_hours),
                        0.0, cfg.horizon_hours - dt)
    dwell_h = rng.uniform(*cfg.dwell_hours_range, size=n)

    arrival = np.floor(arrival_h / dt).astype(int)
    departure = np.ceil((arrival_h + dwell_h) / dt).astype(int)
    departure = np.clip(departure, arrival + 1, T)      # truncated at the horizon

    return Fleet(
        ev_id=np.array(["EV-%03d" % (i + 1) for i in range(n)], dtype=object),
        battery_kwh=battery, soc_init=soc_i, soc_target=soc_t,
        arrival_slot=arrival, departure_slot=departure,
        charger_kw=np.asarray(charger, dtype=float), n_slots=T,
    )


@dataclass
class Scenario:
    """Everything that is shared by every algorithm within one experiment."""
    cfg: SimulationConfig
    fleet: Fleet
    solar_kw: np.ndarray
    tariff: np.ndarray

    @property
    def available_solar_kwh(self) -> float:
        return float(self.solar_kw.sum() * self.cfg.dt)


def build_scenario(cfg: SimulationConfig) -> Scenario:
    """Build a reproducible scenario (fleet + solar + tariff) from a config."""
    rng = np.random.default_rng(cfg.seed)
    fleet = generate_fleet(cfg, rng)
    return Scenario(cfg=cfg, fleet=fleet,
                    solar_kw=solar_profile(cfg, rng), tariff=tariff_profile(cfg))
