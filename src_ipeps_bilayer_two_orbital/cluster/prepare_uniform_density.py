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


def uniform_density_cases(checkerboard=False):
    template = d3_cases()[0]
    steps = sum(stage["Steps"] for stage in template["algorithm"]["Schedule"])
    seeds = SEEDS[:4] if checkerboard else SEEDS
    patterns = ("uniform_type_x",) if checkerboard else CHARGE_PATTERNS
    cases = []
    for seed in seeds:
        for charge_pattern in patterns:
            case = copy.deepcopy(template)
            case.update(id=f"table_{seed}_s0_{charge_pattern}_c0",
                        point="table_opposite_txz_uniform_type" if checkerboard
                        else "table_opposite_txz_uniform_density")
            case["algorithm"].update(
                InitState=seed, SpinOffset=0, LayerAlignment=-1,
                ChargePattern=charge_pattern, ChargeOffset=0,
                Chi=32, CTMTolerance=1e-5, MeasureInitial=True,
                MeasureStageEnds=False, MeasureEvery=steps + 1,
            )
            if checkerboard:
                case["algorithm"]["SweepOrder"] = "bond_type"
            cases.append(case)
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--checkerboard", action="store_true",
                        help="Compare four magnetic seeds with checkerboard dimers and grouped bond sweeps")
    args = parser.parse_args()
    cases = uniform_density_cases(args.checkerboard)
    campaign = "opposite_txz_uniform_type_d3_20261107" if args.checkerboard else CAMPAIGN
    manifest = {
        "schema": "lno.cent.followup.v1",
        "campaign": campaign,
        "description": (
            "Original-table N-only D3 runs at T=2.95 and Chi=32 with opposite x/y hopping. "
            + ("Each spin seed uses x-directed checkerboard dimers and grouped H1/H2/V1/V2 sweeps."
               if args.checkerboard else
               "Each spin seed uses two intralayer dimer orientations with initial 1.5 electrons per Ni.")
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
    print(f"Prepared {len(cases)} cases for {campaign}")


if __name__ == "__main__":
    main()
