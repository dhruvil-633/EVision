"""Reproduce the example simulation output shipped with the project.

    python -m scripts.run_example

Writes example_output.md (full report) and example_comparison.csv using the
default configuration and seed, so the numbers in the README can be checked.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from config import DEFAULT_CONFIG                     # noqa: E402
from data.synthetic_data import build_scenario        # noqa: E402
from evaluation.comparison import run_all, summary_table   # noqa: E402
from evaluation.report_generator import build_report  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent


def main(cfg=DEFAULT_CONFIG):
    scenario = build_scenario(cfg)
    print("Scenario: %d EVs, %.0f kWh requested, %.0f kWh solar available"
          % (scenario.fleet.n_evs, scenario.fleet.requested_kwh.sum(),
             scenario.available_solar_kwh))

    def progress(name, g, total, best):
        if g == total or g % 10 == 0:
            print("  %-20s generation %3d/%d  best J = %.4f"
                  % (name, g, total, best))

    results = run_all(scenario, progress=progress)
    table = summary_table(results, scenario)
    print(table.to_string())

    (ROOT / "example_output.md").write_text(build_report(scenario, results),
                                            encoding="utf-8")
    table.to_csv(ROOT / "example_comparison.csv")
    print("\nWrote example_output.md and example_comparison.csv")
    return results


if __name__ == "__main__":
    main()
