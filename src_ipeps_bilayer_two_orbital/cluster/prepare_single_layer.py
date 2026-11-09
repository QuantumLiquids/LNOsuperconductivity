#!/usr/bin/env python3
"""Prepare the matched zero-interlayer single-layer iPEPS comparison."""

import argparse
import copy
import json
from pathlib import Path

from cent_followup import SINGLE_LAYER_EXECUTABLE, validate_manifest
from prepare_opposite_txz import checked_physics, d3_cases


POINTS = {
    "table": ("table_opposite_txz", {}),
    "r030": ("boundary_r030_s100_opposite_txz", {"Jxz1": .0585, "Jxz2": .01755}),
}

# Each pair is (x-covering seed, phase, rotated y-covering seed, phase).
# (row, col) -> (col+1, row) flips anchor parity, takes H1/H2 to V1/V2,
# and changes the plus-diagonal phase by -1 and the minus phase by negation.
PRIMARY_PAIRS = (
    ("diagonal_minus", 0, "diagonal_minus", 0),
    ("diagonal_plus", 0, "diagonal_plus", 3),
    ("diagonal_minus", 1, "diagonal_minus", 3),
    ("stripe_y", 0, "stripe_x", 0),
)


def prepare_campaign(campaign, dimensions, points):
    template = d3_cases()[0]
    settings = template["algorithm"]
    settings.pop("LayerAlignment")
    steps = sum(stage["Steps"] for stage in settings["Schedule"])
    settings.update(Chi=32, CTMTolerance=1e-5, MeasureInitial=True,
                    MeasureEvery=steps + 1)
    cases = []
    for dimension in dimensions:
        for point in points:
            point_id, exchange = POINTS[point]
            physics = checked_physics(dict(template["physics"], **exchange))
            pairs = list(PRIMARY_PAIRS)
            if point == "table":
                pairs.extend((("stripe_pi2_y", 0, "stripe_pi2_x", 0),
                              ("fm", 0, None, None), ("neel", 0, None, None)))
            for seed, spin, rotated_seed, rotated_spin in pairs:
                case = {
                    "id": f"{point}_d{dimension}_{seed}_s{spin}_x",
                    "kind": "evolve",
                    "point": point_id,
                    "executable": SINGLE_LAYER_EXECUTABLE,
                    "start_total_time": 0.,
                    "physics": physics,
                    "algorithm": dict(settings, Dmax=dimension, InitState=seed, SpinOffset=spin,
                                      ChargePattern="uniform_type_x", ChargeOffset=0,
                                      SweepOrder="bond_type"),
                }
                cases.append(case)
                if rotated_seed is not None:
                    rotated = copy.deepcopy(case)
                    rotated["id"] = f"{point}_d{dimension}_{rotated_seed}_s{rotated_spin}_y"
                    rotated["rotated_from"] = case["id"]
                    rotated["algorithm"].update(
                        InitState=rotated_seed, SpinOffset=rotated_spin,
                        ChargePattern="uniform_type_y", ChargeOffset=1,
                        SweepOrder="bond_type_y_first",
                    )
                    cases.append(rotated)
    return validate_manifest({
        "schema": "lno.cent.followup.v1",
        "campaign": campaign,
        "description": (
            "Projected single-layer N-only iPEPS at 1.5 electrons/Ni with opposite x/y hopping. "
            "Original table and constrained exchange comparison at T=2.95 and Chi=32. "
            "Paired starts rotate the magnetic texture, uniform-density dimer covering and gate sequence."
        ),
        "cases": cases,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--d", action="append", type=int, help="Bond-dimension cap; repeat to select several")
    parser.add_argument("--point", action="append", choices=POINTS, help="Parameter point; repeat to select both")
    parser.add_argument("--campaign", default="single_layer_opposite_txz_20261109")
    args = parser.parse_args()
    dimensions = sorted(set(args.d or (3, 4)))
    if any(d < 2 for d in dimensions):
        parser.error("The uniform-density dimer seed requires D >= 2")
    points = list(dict.fromkeys(args.point or POINTS))
    manifest = prepare_campaign(args.campaign, dimensions, points)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open("x") as stream:
            json.dump(manifest, stream, indent=2)
            stream.write("\n")
    except FileExistsError:
        parser.error("Refusing to replace existing manifest")
    print(f"Prepared {len(manifest['cases'])} cases for {args.campaign}")


if __name__ == "__main__":
    main()
