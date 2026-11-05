#!/usr/bin/env python3
"""Prepare the 2026-11-05 N-only convergence and constrained-exchange study.

The prior result JSON supplies immutable physics, algorithms and checkpoint
names. All output manifests are new; run with cent_followup.py/Slurm.
"""

import argparse
import copy
import json
from pathlib import Path


EXCHANGES = ("Jx1", "Jx2", "Jxz1", "Jxz2")


def manifest(campaign, description, cases):
    return {"schema": "lno.cent.followup.v1", "campaign": campaign,
            "description": description, "cases": cases}


def checkpoint(deployment, prior, record):
    return str(deployment / "runs" / prior["manifest"]["campaign"] /
               record["case"]["id"] / "state/checkpoint_step00000020")


def algorithm(record, **changes):
    result = copy.deepcopy(record["case"]["algorithm"])
    result.update(ThreadNum=1, CTMTolerance=1e-6, MaxCTMSweeps=200,
                  MinCTMSweeps=2, MeasureInitial=False, MeasureStageEnds=False,
                  CheckpointEvery=5)
    result.pop("GateCacheDir", None)
    result.update(changes)
    return result


def prepare(prior, deployment):
    records = prior["records"]
    if len(records) != 32 or any(r["status"] != "complete" for r in records):
        raise ValueError("Expected all 32 completed cases of the prior campaign")
    table = records[0]["case"]["physics"]
    legacy = next(r["case"]["physics"] for r in records if r["case"]["point"] == "candidate")

    # Index 0 is a Chi16 pilot for the most compact axial D4 checkpoint.
    chosen_d4 = [records[i] for i in (5, 2, 8, 23, 20, 26)]
    chi_cases = []
    for chi in (16, 8, 12):
        for record in chosen_d4:
            old = record["case"]
            point = "table" if old["point"] == "table" else "legacy_candidate"
            chi_cases.append({"id": f"{point}_{old['algorithm']['InitState']}_d4_chi{chi}",
                              "kind": "measure", "point": point,
                              "physics": copy.deepcopy(old["physics"]),
                              "algorithm": algorithm(record, Chi=chi, Schedule=[]),
                              "start_total_time": 1.3,
                              "checkpoint": checkpoint(deployment, prior, record)})

    time_cases = []
    # Continue both diagonal orientations and the same axial reference at D3/D4.
    for index in (1, 4, 7, 19, 22, 25, 2, 5, 8, 20, 23, 26):
        record = records[index]
        old = record["case"]
        point = "table" if old["point"] == "table" else "legacy_candidate"
        time_cases.append({"id": f"{point}_d{old['algorithm']['Dmax']}_{old['algorithm']['InitState']}_t210",
                           "kind": "evolve", "point": point,
                           "physics": copy.deepcopy(old["physics"]),
                           "algorithm": algorithm(record, Schedule=[{"Tau": .03, "Steps": 20},
                                                                    {"Tau": .01, "Steps": 20}]),
                           "start_total_time": 1.3,
                           "restart_checkpoint": checkpoint(deployment, prior, record)})

    points = []
    for fraction in (0., .05, .10):
        for scale in (1., .5):
            physics = copy.deepcopy(table)
            for key in EXCHANGES:
                physics[key] = scale * ((1-fraction)*table[key] + fraction*legacy[key])
            points.append((f"near_lam{round(100*fraction):03d}_s{round(100*scale):03d}", physics,
                           {"interpolation_fraction": fraction, "af_scale": scale}))
    for scale in (1., .5):
        physics = copy.deepcopy(table)
        physics.update(Jx1=.065*scale, Jx2=.0633*scale,
                       Jxz1=.0585*scale, Jxz2=.01755*scale)
        points.append((f"boundary_r030_s{round(100*scale):03d}", physics,
                       {"jxz1_over_jx1": .9, "jxz2_over_jxz1": .3, "af_scale": scale}))
    starts = [(seed, spin, "stripe_x") for seed in ("diagonal_minus", "diagonal_plus")
              for spin in (0, 1)]
    starts += [(seed, 0, charge) for seed in ("stripe_x", "stripe_y")
               for charge in ("stripe_x", "stripe_y")]
    scan_cases = []
    for name, physics, coordinates in points:
        if not (min(physics["Jx1"], physics["Jx2"]) > physics["Jxz1"] and
                physics["Jxz2"]/physics["Jxz1"] > .25):
            raise ValueError("Exchange constraints violated: " + name)
        for seed, spin, charge in starts:
            scan_cases.append({"id": f"{name}_{seed}_s{spin}_{charge}",
                               "kind": "evolve", "point": name,
                               "physics": copy.deepcopy(physics),
                               "algorithm": algorithm(records[1], Dmax=3, InitState=seed,
                                                      SpinOffset=spin, ChargePattern=charge,
                                                      Schedule=[{"Tau": .1, "Steps": 25},
                                                                {"Tau": .03, "Steps": 15}]),
                               "start_total_time": 0., "scan_coordinates": coordinates})
    return {
        "chi_manifest.json": manifest("nonly_chi_followup_20261105",
            "Frozen T1.3 D4 states: Chi8,12,16 at fixed tolerance1e-6; no evolution. Index0 is Chi16 pilot.", chi_cases),
        "time_manifest.json": manifest("nonly_time_followup_20261105",
            "Matched D3/D4 continuation T1.3->2.1, Chi8; table and legacy candidate are convergence controls.", time_cases),
        "constrained_manifest.json": manifest("nonly_constrained_j_d3_20261105",
            "Eight allowed exchange points, D3 cold T2.95, Chi8. Both diagonal phases and both axial charge orientations. TB/V/Hund fixed.", scan_cases)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prior_results", type=Path)
    parser.add_argument("deployment", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepared = prepare(json.loads(args.prior_results.read_text()), args.deployment)
    if any((args.output/name).exists() for name in prepared):
        parser.error("Refusing to replace an existing follow-up manifest")
    args.output.mkdir(parents=True, exist_ok=True)
    for name, value in prepared.items():
        (args.output/name).write_text(json.dumps(value, indent=2)+"\n")
        print(name, len(value["cases"]), "cases")


if __name__ == "__main__":
    main()
