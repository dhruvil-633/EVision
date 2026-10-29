"""Central configuration for the EVision simulation platform.

All quantities use SI-consistent engineering units:
    power    -> kW
    energy   -> kWh
    time     -> hours
    tariff   -> INR per kWh (Rs/kWh)

A ``SimulationConfig`` fully determines a scenario together with its random
seed, which makes every experiment reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict, replace


@dataclass(frozen=True)
class SimulationConfig:
    # ---- time grid -------------------------------------------------------
    horizon_hours: float = 24.0
    slot_minutes: int = 15

    # ---- fleet -----------------------------------------------------------
    n_evs: int = 40
    seed: int = 42
    battery_kwh_range: tuple = (30.0, 75.0)
    soc_init_range: tuple = (0.15, 0.55)
    soc_target_range: tuple = (0.75, 0.95)
    charger_kw_choices: tuple = (7.4, 11.0, 22.0)
    charger_kw_weights: tuple = (0.45, 0.35, 0.20)
    arrival_peak_hours: tuple = (8.5, 18.0)   # bimodal workplace / evening peaks
    arrival_sigma_hours: float = 1.8
    dwell_hours_range: tuple = (3.0, 11.0)

    # ---- station ---------------------------------------------------------
    station_capacity_kw: float = 150.0
    charging_efficiency: float = 0.93

    # ---- solar -----------------------------------------------------------
    solar_capacity_kw: float = 120.0
    sunrise_hour: float = 6.5
    sunset_hour: float = 18.5
    cloud_variability: float = 0.12          # 0 = perfectly clear sky

    # ---- tariff (time-of-use, Rs/kWh) ------------------------------------
    tariff_offpeak: float = 5.0
    tariff_standard: float = 8.0
    tariff_peak: float = 12.5
    peak_windows: tuple = ((7.0, 10.0), (18.0, 22.0))
    offpeak_windows: tuple = ((0.0, 6.0), (23.0, 24.0))

    # ---- objective weights ----------------------------------------------
    w_cost: float = 1.0
    w_peak: float = 0.8
    w_unmet: float = 8.0          # dominant: never "optimise" by not charging
    w_delay: float = 0.3
    w_solar: float = 0.6          # subtracted reward

    # ---- genetic algorithm ----------------------------------------------
    ga_population: int = 40
    ga_generations: int = 40
    ga_crossover_prob: float = 0.8
    ga_mutation_prob: float = 0.15
    ga_mutation_sigma: float = 0.25
    ga_elite: int = 2
    ga_tournament: int = 3

    @property
    def dt(self) -> float:
        """Slot duration in hours."""
        return self.slot_minutes / 60.0

    @property
    def n_slots(self) -> int:
        return int(round(self.horizon_hours / self.dt))

    def slot_hours(self):
        """Clock time (hours) at the start of each slot."""
        import numpy as np
        return np.arange(self.n_slots) * self.dt

    def with_(self, **kwargs) -> "SimulationConfig":
        return replace(self, **kwargs)

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_CONFIG = SimulationConfig()
