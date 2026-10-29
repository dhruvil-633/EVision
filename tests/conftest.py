"""Shared pytest fixtures and helpers."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SimulationConfig                      # noqa: E402
from data.synthetic_data import Scenario, build_scenario  # noqa: E402
from simulation.ev_models import Fleet                   # noqa: E402

TOL = 1e-6


@pytest.fixture
def small_cfg():
    """Small, fast scenario used by most tests."""
    return SimulationConfig(n_evs=8, seed=7, ga_population=8, ga_generations=4)


@pytest.fixture
def scenario(small_cfg):
    return build_scenario(small_cfg)


@pytest.fixture
def hand_scenario():
    """A hand-checkable scenario: 1 EV, 4 slots, no solar, flat tariff.

    battery 50 kWh, SOC 0.50 -> 0.60  => requested 5.0 kWh (battery side)
    charger 10 kW, dt 0.25 h, efficiency 0.90, tariff 10 Rs/kWh
    """
    cfg = SimulationConfig(n_evs=1, horizon_hours=1.0, slot_minutes=15,
                           station_capacity_kw=20.0, charging_efficiency=0.90,
                           solar_capacity_kw=0.0, cloud_variability=0.0,
                           tariff_offpeak=10.0, tariff_standard=10.0,
                           tariff_peak=10.0, peak_windows=(), offpeak_windows=())
    fleet = Fleet(ev_id=np.array(["EV-001"], dtype=object),
                  battery_kwh=np.array([50.0]), soc_init=np.array([0.5]),
                  soc_target=np.array([0.6]), arrival_slot=np.array([0]),
                  departure_slot=np.array([4]), charger_kw=np.array([10.0]),
                  n_slots=cfg.n_slots)
    return Scenario(cfg=cfg, fleet=fleet, solar_kw=np.zeros(cfg.n_slots),
                    tariff=np.full(cfg.n_slots, 10.0))
