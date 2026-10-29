# EVision - Intelligent EV Charging and Energy Optimization

_Soft Computing based simulation study - generated 2026-10-29 23:54_

> All results come from a SYNTHETIC-DATA SIMULATION. Solar generation, tariffs and EV sessions are produced by configurable mathematical models and a seeded random number generator. They are not measurements of a real fleet and are not evidence of real-world savings.

## 1. Problem statement

A shared charging station with a hard 150 kW grid connection and a 120 kW on-site solar array must serve 40 EV charging sessions over a 24 hour horizon discretised into 96 slots of 15 minutes. The scheduler minimises electricity procurement cost and peak grid demand and maximises solar self-consumption, while delivering the requested energy before each departure.

## 2. Experimental settings

| Setting | Value |
| --- | --- |
| Random seed | 42 |
| Fleet size | 40 |
| Time slots | 96 x 15 min |
| Station capacity | 150 kW |
| Solar capacity | 120 kW |
| Charging efficiency | 0.93 |
| Tariff off-peak / standard / peak | 5.00 / 8.00 / 12.50 Rs per kWh |
| Total requested energy | 1096.2 kWh |
| Available solar energy | 757.5 kWh |

## 3. Algorithm configuration

| Parameter | Value |
| --- | --- |
| GA population | 40 |
| GA generations | 40 |
| Crossover probability | 0.8 |
| Mutation probability | 0.15 |
| Mutation sigma | 0.25 |
| Elite individuals | 2 |
| Tournament size | 3 |
| Objective weights cost / peak / unmet / delay / solar | 1.00 / 0.80 / 8.00 / 0.30 / 0.60 |

## 4. Results

| algorithm         |   Electricity cost (Rs) |   Cost vs uncontrolled (%) |   Peak grid demand (kW) |   Peak vs uncontrolled (%) |   Grid energy (kWh) |   Solar utilisation (%) |   Solar to EV (kWh) |   Energy delivered (kWh) |   Unmet energy (kWh) |   Unmet energy (%) |   EVs meeting target (%) |   Avg charging delay (h) |   Constraint violations |   Runtime (s) |
|:------------------|------------------------:|---------------------------:|------------------------:|---------------------------:|--------------------:|------------------------:|--------------------:|-------------------------:|---------------------:|-------------------:|-------------------------:|-------------------------:|------------------------:|--------------:|
| Uncontrolled      |                    8068 |                        0   |                   125   |                        0   |               710.7 |                    55.5 |               420.6 |                   1052.1 |                 44   |               4.02 |                     85   |                     1.29 |                       0 |          0    |
| Rule-based        |                    7551 |                       -6.4 |                   150   |                       20   |               710.7 |                    55.5 |               420.6 |                   1052.1 |                 44   |               4.02 |                     85   |                     1.62 |                       0 |          0.01 |
| Genetic Algorithm |                    6089 |                      -24.5 |                    84.1 |                      -32.7 |               576.9 |                    68   |               514.9 |                   1015.3 |                 80.8 |               7.37 |                     75   |                     2.03 |                       0 |          0.97 |
| Hybrid Fuzzy + GA |                    5437 |                      -32.6 |                    88.1 |                      -29.5 |               525.1 |                    74   |               560.4 |                   1009.4 |                 86.8 |               7.91 |                     77.5 |                     2.6  |                       0 |          1    |

### Comparison against the Uncontrolled baseline

- **Rule-based**: cost -6.4 %, peak 20.0 %, solar utilisation 0.0 pp, unmet energy 44.0 kWh (baseline 44.0 kWh), EVs meeting target 85.0 % (baseline 85.0 %), runtime 0.01 s.
- **Genetic Algorithm**: cost -24.5 %, peak -32.7 %, solar utilisation 12.4 pp, unmet energy 80.8 kWh (baseline 44.0 kWh), EVs meeting target 75.0 % (baseline 85.0 %), runtime 0.97 s.
- **Hybrid Fuzzy + GA**: cost -32.6 %, peak -29.5 %, solar utilisation 18.5 pp, unmet energy 86.8 kWh (baseline 44.0 kWh), EVs meeting target 77.5 % (baseline 85.0 %), runtime 1.00 s.

## 5. Metric definitions

- **Electricity cost**: sum over slots of grid energy times the time-of-use tariff of that slot.
- **Peak grid demand**: maximum grid power over the horizon; on-site solar is self-consumed first.
- **Solar utilisation**: solar energy actually used for charging divided by the solar energy available over the same interval.
- **Unmet energy**: requested battery energy not delivered before departure, summed over all EVs.
- **EVs meeting target**: share of vehicles whose final SOC reaches the requested target SOC.
- **Average charging delay**: energy-weighted mean time between plug-in and actual energy delivery.

## 6. Limitations

- Synthetic inputs only; no measured fleet, irradiance or tariff data.
- Constant charging efficiency, no battery charging curve or thermal model.
- No vehicle-to-grid discharge and no battery degradation cost.
- Perfect foresight of arrivals, departures, solar output and tariffs.
- The GA is a stochastic heuristic: no global optimality is claimed and results vary with the random seed.
- Fuzzy urgency is evaluated from the plug-in state once per scenario rather than inside every fitness evaluation, for runtime reasons.

## 7. Conclusion

On this scenario the hybrid Fuzzy + GA scheduler changed electricity cost by -32.6 % and peak grid demand by -29.5 % relative to uncontrolled charging, and moved solar utilisation from 55.5 % to 74.0 %. Unmet requested energy was 86.8 kWh against 44.0 kWh for the baseline, so part of the cost and peak gain is paid for in departure SOC; increasing the unmet-energy weight in the objective moves the operating point back towards full deadline compliance.

_All results come from a SYNTHETIC-DATA SIMULATION. Solar generation, tariffs and EV sessions are produced by configurable mathematical models and a seeded random number generator. They are not measurements of a real fleet and are not evidence of real-world savings._