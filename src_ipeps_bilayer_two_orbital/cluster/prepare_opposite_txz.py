#!/usr/bin/env python3
"""Prepare fresh opposite-x/y hopping screens and matched D4 cold starts."""

import argparse
import copy
import json
from pathlib import Path


COMPLETE = {"complete", "complete_ctm_recovered"}
MAIN_Q = ("pi2_minus_pi2", "pi2_pi2", "0_pi", "pi_0")
EXCHANGE_KEYS = ("Jx1", "Jx2", "Jxz1", "Jxz2")
SCHEDULE = [{"Tau": .1, "Steps": 25}, {"Tau": .03, "Steps": 15}]


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
                     Schedule=SCHEDULE)
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


def d4_cases(document, best_axial_only=False):
    records = document["records"]
    if len(records) != 32 or len({r["case"]["id"] for r in records}) != 32:
        raise ValueError("D4 selection requires all 32 original D3 starts")
    for record in records:
        if record["status"] in COMPLETE:
            continue
        # A missing axial competitor cannot improve the diagonal's gap
        # against an already measured axial branch. Still keep it missing.
        if not (best_axial_only and record["status"] == "failed" and record.get("checkpoint_step") == 40
                and record["case"]["algorithm"]["InitState"] in {"stripe_x", "stripe_y"}):
            raise ValueError("Unresolved D3 start: " + record["case"]["id"])
    measured = [r for r in records if r["status"] in COMPLETE]
    best = {}
    reference_physics = measured[0]["case"]["physics"]
    chi = measured[0]["measurement"]["chi"]
    ctm_tolerance = measured[0]["case"]["algorithm"]["CTMTolerance"]
    for record in measured:
        case, measurement = record["case"], record["measurement"]
        cold = case.get("source_cold_case", case)
        checked_physics(cold["physics"])
        if case["physics"] != reference_physics or case["point"] != "table_opposite_txz" or \
                cold["physics"] != case["physics"] or cold["kind"] != "evolve" or \
                cold["start_total_time"] != 0 or cold.get("restart_checkpoint") or \
                cold["algorithm"]["Schedule"] != SCHEDULE or \
                case["algorithm"]["Dmax"] != 3 or abs(record["planned_total_time"] - 2.95) > 1e-12 or \
                case["algorithm"]["Chi"] != chi or measurement["chi"] != chi or \
                case["algorithm"]["CTMTolerance"] != ctm_tolerance:
            raise ValueError("Expected matched cold D3 T2.95 states measured at one common Chi")
        q = record["dominant_q"]
        if q in MAIN_Q and (q not in best or measurement["physical_energy_per_ni"] < best[q]["measurement"]["physical_energy_per_ni"]):
            best[q] = record
    axial = [q for q in ("0_pi", "pi_0") if q in best]
    if best_axial_only and axial:
        selected_q = MAIN_Q[:2] + (min(axial, key=lambda q: best[q]["measurement"]["physical_energy_per_ni"]),)
    else:
        selected_q = MAIN_Q
    if any(q not in best for q in selected_q):
        raise ValueError("One or more requested branches did not retain their wavevector")
    cases = []
    for q in selected_q:
        record = best[q]
        case = copy.deepcopy(record["case"].get("source_cold_case", record["case"]))
        case["id"] = "table_d4_" + q
        case["algorithm"].update(Dmax=4, Chi=chi, CTMTolerance=ctm_tolerance)
        case["selected_d3_case"] = record["case"]["id"]
        case["selection"] = "Same cold product initializer and schedule as the best measured D3 start at this q; no checkpoint warm start."
        cases.append(case)
    return cases


def frozen_cases(document, chi):
    """Measure the existing final states at one common Chi, with no new SU."""
    records = document["records"]
    signs = {r["case"]["physics"].get("TxzYSign", 1) for r in records}
    if signs == {-1} and len(records) == 32:
        sign = -1
    elif signs == {1} and len(records) == 8:
        sign = 1
    else:
        raise ValueError("Expected all 32 corrected or 8 matched legacy source cases")
    if len({r["case"]["id"] for r in records}) != len(records):
        raise ValueError("Frozen cohort contains duplicate source cases")
    cases = []
    for record in records:
        old = record["case"]
        checkpoint = record.get("checkpoint", "")
        identity = record.get("checkpoint_identity", {})
        if record["status"] not in COMPLETE | {"failed"} or \
                not checkpoint.endswith("/checkpoint_step00000040") or \
                identity.get("path") != checkpoint or not identity.get("sha256") or \
                old["algorithm"]["Dmax"] != 3 or abs(record["planned_total_time"]-2.95) > 1e-12:
            raise ValueError("Missing final D3 T2.95 checkpoint: " + old["id"])
        case = copy.deepcopy(old)
        if sign == -1:
            cold = old.get("source_cold_case", old)
            if cold["kind"] != "evolve" or cold["algorithm"]["Schedule"] != SCHEDULE or \
                    cold["start_total_time"] != 0 or cold.get("restart_checkpoint"):
                raise ValueError("Unmatched original evolution history: " + old["id"])
            case["source_cold_case"] = copy.deepcopy(cold)
        case["physics"]["TxzYSign"] = sign
        case.update(id=old["id"]+f"_frozen_chi{chi}", kind="measure", start_total_time=2.95,
                    checkpoint=checkpoint, source_last_tau=.03,
                    expected_checkpoint_sha256=identity["sha256"], source_case_id=old["id"])
        case.pop("restart_checkpoint", None)
        case["algorithm"].update(Chi=chi, Schedule=[], CTMTolerance=1e-6, MaxCTMSweeps=200)
        cases.append(case)
    return cases, sign


def exchange_cases(document, best_axial_only=False):
    """Hold the selected table initializers fixed while varying allowed J."""
    selected = d4_cases(document, best_axial_only)
    half = {key: .5*selected[0]["physics"][key] for key in EXCHANGE_KEYS}
    points = [
        ("near_lam010_s100_opposite_txz", dict(zip(EXCHANGE_KEYS, (.060125, .0585525, .05465, .02479))),
         {"interpolation_fraction": .10, "af_scale": 1.}),
        ("boundary_r030_s100_opposite_txz", {"Jxz1": .0585, "Jxz2": .01755},
         {"jxz2_over_jxz1": .30, "jxz1_over_jx1": .9, "af_scale": 1.}),
        ("near_lam000_s050_opposite_txz", half, {"interpolation_fraction": 0., "af_scale": .5}),
    ]
    cases = []
    for point, exchanges, coordinates in points:
        for source in selected:
            q = source["id"][len("table_d4_"):]
            case = copy.deepcopy(source)
            case.update(id=point+"_"+q, point=point, scan_coordinates=coordinates,
                        selection="Cold start with the best original-table D3 initializer at the observed q; only the AF exchanges are changed.")
            case["algorithm"]["Dmax"] = 3
            case["physics"].update(exchanges)
            checked_physics(case["physics"])
            cases.append(case)
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--d4-from", type=Path)
    selection.add_argument("--exchange-from", type=Path)
    selection.add_argument("--freeze-from", type=Path)
    parser.add_argument("--chi", type=int, default=24, help="New contraction Chi for --freeze-from")
    parser.add_argument("--best-axial-only", action="store_true",
                        help="Compare both diagonals with the best measured axial branch; allow a retained but unmeasured axial checkpoint")
    args = parser.parse_args()
    if args.freeze_from:
        if args.chi < 1:
            parser.error("Chi must be positive")
        cases, sign = frozen_cases(json.loads(args.freeze_from.read_text()), args.chi)
        prefix = "opposite_txz" if sign == -1 else "legacy_same_sign"
        campaign = f"{prefix}_table_d3_chi{args.chi}_frozen_20261107"
        description = f"Frozen D3 T2.95 checkpoints remeasured at common Chi{args.chi}, TxzYSign={sign}; no additional SU."
    elif args.exchange_from:
        cases = exchange_cases(json.loads(args.exchange_from.read_text()), args.best_axial_only)
        campaign = "opposite_txz_exchange_d3_20261107"
        description = f"Opposite x/y hopping; 3 constrained AF points with {len(cases)//3} matched cold D3 branch starts each, T2.95 Chi{cases[0]['algorithm']['Chi']}."
    else:
        cases = d4_cases(json.loads(args.d4_from.read_text()), args.best_axial_only) if args.d4_from else d3_cases()
        dimension = 4 if args.d4_from else 3
        campaign = f"opposite_txz_table_d{dimension}_20261107"
        description = f"Confirmed opposite x/y interorbital hopping; fresh N-only cold starts, original table exchanges, T2.95 Chi{cases[0]['algorithm']['Chi']}. Previous same-sign states/caches are not reused."
    result = {"schema": "lno.cent.followup.v1",
              "campaign": campaign, "description": description,
              "cases": cases}
    if args.output.exists():
        parser.error("Refusing to replace existing manifest")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(f"Prepared {len(cases)} cases for {campaign}; TxzYSign={cases[0]['physics']['TxzYSign']}")


if __name__ == "__main__":
    main()
