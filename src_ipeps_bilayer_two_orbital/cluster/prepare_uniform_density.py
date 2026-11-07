#!/usr/bin/env python3
"""Prepare the matched D3 screen with uniform initial site densities."""

import argparse
import copy
import json
from pathlib import Path

from prepare_opposite_txz import d3_cases


SEEDS = ("diagonal_minus", "diagonal_plus", "stripe_x", "stripe_y", "stripe_pi2_y")
CHARGE_PATTERNS = ("uniform_x", "uniform_y")
CAMPAIGN = "opposite_txz_uniform_density_d3_20261107"


def uniform_density_cases():
    template = d3_cases()[0]
    steps = sum(stage["Steps"] for stage in template["algorithm"]["Schedule"])
    cases = []
    for seed in SEEDS:
        for charge_pattern in CHARGE_PATTERNS:
            case = copy.deepcopy(template)
            case.update(id=f"table_{seed}_s0_{charge_pattern}_c0",
                        point="table_opposite_txz_uniform_density")
            case["algorithm"].update(
                InitState=seed, SpinOffset=0, LayerAlignment=-1,
                ChargePattern=charge_pattern, ChargeOffset=0,
                Chi=32, CTMTolerance=1e-5, MeasureInitial=True,
                MeasureStageEnds=False, MeasureEvery=steps + 1,
            )
            cases.append(case)
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    cases = uniform_density_cases()
    manifest = {
        "schema": "lno.cent.followup.v1",
        "campaign": CAMPAIGN,
        "description": (
            "Original-table N-only D3 runs at T=2.95 and Chi=32 with opposite x/y hopping. "
            "Each spin seed uses two intralayer dimer orientations with initial 1.5 electrons per Ni."
        ),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open("x") as output:
            json.dump(manifest, output, indent=2)
            output.write("\n")
    except FileExistsError:
        parser.error("Refusing to replace existing manifest")
    print(f"Prepared {len(cases)} cases for {CAMPAIGN}")


if __name__ == "__main__":
    main()
