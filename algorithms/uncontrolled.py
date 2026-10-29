"""Baseline 1 - uncontrolled ("dumb") charging.

Every EV starts charging at its full charger rating the moment it plugs in and
keeps charging until its target SOC is reached. The only thing that limits it
is the shared station power limit, which is shared by fair proportional
curtailment when the station is over-subscribed.

This is the reference against which cost and peak savings are reported.
"""

from __future__ import annotations

import time

from simulation.constraints import decode_schedule
from simulation.energy_accounting import simulate

NAME = "Uncontrolled"


def run(scenario):
    t0 = time.perf_counter()
    desired = scenario.fleet.max_power()          # charge as fast as possible, always
    power = decode_schedule(desired, scenario)
    return simulate(power, scenario, name=NAME, runtime_s=time.perf_counter() - t0)
