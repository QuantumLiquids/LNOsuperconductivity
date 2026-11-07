#!/usr/bin/env python3
"""Prepare fresh opposite-x/y hopping screens and matched D4 cold starts."""

import argparse
import copy
import json
from pathlib import Path


COMPLETE = {"complete", "complete_ctm_recovered"}
MAIN_Q = ("pi2_minus_pi2", "pi2_pi2", "0_pi", "pi_0")


def checked_physics(physics):
    if physics.get("TxzYSign") != -1:
        raise ValueError("This campaign requires explicit TxzYSign=-1")
    if min(physics["Jx1"], physics["Jx2"]) <= physics["Jxz1"] or \
            physics["Jxz2"] / physics["Jxz1"] <= .25:
        raise ValueError("Exchange parameters violate the collaborator's bounds")
    return physics


def d3_cases():
    source = Path(__file__).resolve().parents[1]
    physics = checked_physics(json.loads((source/"params/physics_table.json").read_text())["CaseParams"])
    algorithm = json.loads((source/"params/candidate_d2.json").read_text())["CaseParams"]
    algorithm.update(Dmax=3, ThreadNum=1, Chi=16, CTMTolerance=1e-6,
                     MaxCTMSweeps=200, MinCTMSweeps=2, MeasureInitial=False,
                     MeasureStageEnds=False, CheckpointEvery=5,
                     Schedule=[{"Tau": .1, "Steps": 25}, {"Tau": .03, "Steps": 15}])
    algorithm.pop("GateCacheDir", None)
    # The first eight reproduce the controlled initializer coverage, followed
    # by other independent charge registries and magnetic competitors.
    starts = [(seed, spin, "stripe_x", 0) for seed in ("diagonal_minus", "diagonal_plus") for spin in (0, 1)]
    starts += [(seed, 0, charge, 0) for seed in ("stripe_x", "stripe_y") for charge in ("stripe_x", "stripe_y")]
    for seed in ("diagonal_minus", "diagonal_plus"):
        for spin in (0, 1):
            for charge in ("stripe_x", "stripe_y"):
                for offset in (0, 1):
                    item = (seed, spin, charge, offset)
                    if item not in starts:
                        starts.append(item)
    starts += [(seed, 0, charge, 1) for seed in ("stripe_x", "stripe_y") for charge in ("stripe_x", "stripe_y")]
    starts += [(seed, 0, charge, 0) for seed in ("fm", "neel") for charge in ("stripe_x", "stripe_y")]
    starts += [("stripe_pi2_x", 0, "stripe_y", 0), ("stripe_pi2_x", 1, "stripe_x", 0),
               ("stripe_pi2_y", 0, "stripe_x", 0), ("stripe_pi2_y", 1, "stripe_y", 0)]
    assert len(starts) == len(set(starts)) == 32
    cases = []
    for seed, spin, charge, offset in starts:
        settings = copy.deepcopy(algorithm)
        settings.update(InitState=seed, SpinOffset=spin, ChargePattern=charge,
                        ChargeOffset=offset, LayerAlignment=-1)
        cases.append({"id": f"table_{seed}_s{spin}_{charge}_c{offset}",
                      "kind": "evolve", "point": "table_opposite_txz", "start_total_time": 0,
                      "physics": copy.deepcopy(physics), "algorithm": settings})
    return cases


def d4_cases(document):
    records = document["records"]
    if len(records) != 32 or any(r["status"] not in COMPLETE for r in records):
        raise ValueError("D4 selection requires all 32 D3 measurements, including competitors")
    if len({r["case"]["id"] for r in records}) != 32:
        raise ValueError("D3 selection contains duplicate cases")
    best = {}
    reference_physics = records[0]["case"]["physics"]
    for record in records:
        case, measurement = record["case"], record["measurement"]
        checked_physics(case["physics"])
        if case["physics"] != reference_physics or case["point"] != "table_opposite_txz" or \
                case["kind"] != "evolve" or case["start_total_time"] != 0 or case.get("restart_checkpoint") or \
                case["algorithm"]["Schedule"] != [{"Tau": .1, "Steps": 25}, {"Tau": .03, "Steps": 15}] or \
                case["algorithm"]["Dmax"] != 3 or abs(record["planned_total_time"] - 2.95) > 1e-12 or \
                measurement["chi"] != 16:
            raise ValueError("Expected the matched cold D3 T2.95 Chi16 screen")
        q = record["dominant_q"]
        if q in MAIN_Q and (q not in best or measurement["physical_energy_per_ni"] < best[q]["measurement"]["physical_energy_per_ni"]):
            best[q] = record
    if set(best) != set(MAIN_Q):
        raise ValueError("One or more requested branches did not retain their wavevector")
    cases = []
    for q in MAIN_Q:
        record = best[q]
        case = copy.deepcopy(record["case"])
        case["id"] = "table_d4_" + q
        case["algorithm"]["Dmax"] = 4
        case["selected_d3_case"] = record["case"]["id"]
        case["selection"] = "Same cold product initializer and schedule as the best measured D3 start at this q; no checkpoint warm start."
        cases.append(case)
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--d4-from", type=Path)
    args = parser.parse_args()
    cases = d4_cases(json.loads(args.d4_from.read_text())) if args.d4_from else d3_cases()
    dimension = 4 if args.d4_from else 3
    result = {"schema": "lno.cent.followup.v1",
              "campaign": f"opposite_txz_table_d{dimension}_20261107",
              "description": "Confirmed opposite x/y interorbital hopping; fresh N-only cold starts, original table exchanges, T2.95 Chi16. Previous same-sign states/caches are not reused.",
              "cases": cases}
    if args.output.exists():
        parser.error("Refusing to replace existing manifest")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(f"Prepared {len(cases)} D{dimension} cases; TxzYSign=-1")


if __name__ == "__main__":
    main()
